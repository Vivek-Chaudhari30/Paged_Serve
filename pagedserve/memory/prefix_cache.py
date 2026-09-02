"""Block-aligned prefix caching: a hash chain plus an LRU pool.

In a chat or few-shot workload many requests share a long identical prefix — a
system prompt, tool schemas, examples. Prefilling that prefix once per request
is pure waste, and prefill is what dominates time-to-first-token.

Why the hash is chained
-----------------------
``hash(block_i) = H(hash(block_{i-1}), tokens_of_block_i)``

Hashing each block's tokens independently would be **wrong**, and wrong in the
worst way: it would produce plausible output rather than a crash.

A cached block holds K and V, and those values are not a function of the block's
own tokens. Attention makes every key and value depend on the entire context
preceding it. The tokens ``["the", "cat"]`` appearing at position 32 of one
prompt and position 32 of a different prompt have *different* K and V, because
everything before them differs. An independent hash would match those two
blocks, hand the second request the first one's KV, and generate fluent text
conditioned on a context that never existed.

Chaining makes a hit mean "identical tokens all the way back to position zero",
which is the only condition under which sharing KV is sound.

Why only whole blocks are cached
--------------------------------
A partially filled block is *mutable* — the next token generated writes into it.
Sharing a mutable block would mean copy-on-write on every append, which is
constant copying. Restricting the cache to sealed, full blocks makes every
shared block immutable and therefore free to share. The cost is up to
``block_size - 1`` tokens of missed reuse at the tail of a prefix; the gain is a
scheme that cannot corrupt another request's state.

Why blocks linger after their last user leaves
----------------------------------------------
When a sequence finishes, its blocks drop to refcount zero — but a finished
request's system-prompt blocks are the *most* likely blocks to be wanted again a
second later. Returning them to the free list immediately throws away the best
entries in the cache. They go into an LRU pool instead and are only reclaimed
under real memory pressure. Keeping them until something needs the space is the
entire point of a cache.

Why not a radix tree
--------------------
SGLang's RadixAttention stores prefixes in a radix tree, giving automatic
longest-prefix matching and clean sharing across a branching conversation tree.
It is strictly more capable and meaningfully more complex. A flat hash chain
captures the dominant real case — an identical system prompt or few-shot block
across many requests — for a fraction of the work. Knowing why the simpler thing
was chosen is worth more than having built the complex one.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["PrefixCache", "PrefixCacheStats", "block_hashes", "chain_hash"]


def chain_hash(parent: int | None, tokens: Sequence[int]) -> int:
    """One link in the chain: the parent's hash plus this block's tokens.

    Uses blake2b rather than Python's ``hash()``. Built-in hashing of ``str``
    and ``bytes`` is salted per process, so a cache keyed on it would silently
    stop matching across a restart — and worse, would appear to work in any
    single-process test. Integer hashing happens not to be salted today, which
    makes the bug even easier to miss. An explicit digest removes the question.

    Truncated to 64 bits. A collision would serve one request another's KV, so
    it is worth being precise about the risk: at 2^64, reaching even a
    one-in-a-billion chance of a single collision takes on the order of 10^5
    distinct cached blocks... times far more than any cache will hold. The
    birthday bound at a million cached blocks is about 3e-8.
    """
    digest = hashlib.blake2b(digest_size=8)
    if parent is not None:
        digest.update(parent.to_bytes(8, "little"))
    # Fixed-width little-endian, so token 1 followed by token 256 cannot encode
    # to the same bytes as some other pair.
    for token in tokens:
        digest.update(int(token).to_bytes(8, "little", signed=True))
    return int.from_bytes(digest.digest(), "little")


def block_hashes(
    token_ids: Sequence[int], block_size: int, *, parent: int | None = None
) -> list[int]:
    """Chained hashes for each *complete* block of ``token_ids``.

    A trailing partial block produces no hash, because it is not cacheable.

    Args:
        token_ids: The sequence's tokens, from position zero.
        block_size: Tokens per block.
        parent: Hash of the block preceding ``token_ids``, for hashing a
            continuation rather than a fresh prompt.

    Returns:
        One hash per full block, in order. Element *i* identifies the entire
        prefix through the end of block *i*, never block *i* alone.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    hashes: list[int] = []
    current = parent
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        current = chain_hash(current, token_ids[start : start + block_size])
        hashes.append(current)
    return hashes


@dataclass
class PrefixCacheStats:
    """What the cache is actually doing, for the result file.

    A hit rate without ``tokens_saved`` is not interpretable: hits on a 16-token
    block and on a 2000-token prefix count the same and mean nothing alike.
    """

    lookups: int = 0
    hits: int = 0
    blocks_reused: int = 0
    tokens_saved: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float | None:
        """Fraction of block lookups that hit, or ``None`` if none were made."""
        if self.lookups == 0:
            return None
        return self.hits / self.lookups

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": self.hit_rate,
            "blocks_reused": self.blocks_reused,
            "tokens_saved": self.tokens_saved,
            "evictions": self.evictions,
        }


class PrefixCache:
    """Maps prefix hashes to physical blocks, and keeps unused ones warm.

    Owns no blocks itself. It is an index over blocks the ``BlockManager``
    allocated, plus an eviction order for the ones nobody currently references.
    The allocator remains the single authority on what is free.
    """

    def __init__(self, block_size: int, *, enabled: bool = True) -> None:
        self.block_size = block_size
        self.enabled = enabled
        self.stats = PrefixCacheStats()

        # hash -> physical block id, for blocks that are cached, referenced or not.
        self._by_hash: dict[int, int] = {}
        # physical block id -> hash, so a block can be un-indexed when reclaimed.
        self._by_block: dict[int, int] = {}
        # Reclaimable blocks in least-recently-used order. OrderedDict rather
        # than a deque: eviction is by LRU order but removal on a cache hit is
        # by block id, and only a mapping makes both cheap.
        self._pool: OrderedDict[int, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._by_hash)

    @property
    def num_reclaimable(self) -> int:
        """Cached blocks nobody currently references."""
        return len(self._pool)

    def lookup(self, block_hash: int) -> int | None:
        """The physical block for this prefix, or ``None``.

        Does not change refcounts — the caller owns that, because acquiring a
        block and accounting for it must not be two operations that can be
        interleaved.
        """
        if not self.enabled:
            return None
        self.stats.lookups += 1
        block_id = self._by_hash.get(block_hash)
        if block_id is None:
            return None
        self.stats.hits += 1
        # Referenced again, so no longer a candidate for eviction.
        self._pool.pop(block_id, None)
        return block_id

    def match(self, hashes: Sequence[int]) -> list[int]:
        """Physical blocks for the longest matching *leading* run of hashes.

        Stops at the first miss and does not look further. A later block whose
        hash happens to be present cannot be used, because the sequence's own
        blocks must be contiguous from position zero — a gap would mean
        attending over KV for tokens the request never had.
        """
        matched: list[int] = []
        for block_hash in hashes:
            block_id = self.lookup(block_hash)
            if block_id is None:
                break
            matched.append(block_id)
        if matched:
            self.stats.blocks_reused += len(matched)
            self.stats.tokens_saved += len(matched) * self.block_size
        return matched

    def insert(self, block_hash: int, block_id: int) -> None:
        """Index a freshly filled block under its prefix hash.

        An existing entry for the same hash wins. The two blocks hold identical
        KV by construction, and keeping the incumbent avoids invalidating
        references other sequences already hold.
        """
        if not self.enabled:
            return
        if block_hash in self._by_hash:
            return
        self._by_hash[block_hash] = block_id
        self._by_block[block_id] = block_hash

    def reclaimable_blocks(self) -> list[int]:
        """Cached blocks nobody references, in least-recently-used order."""
        return list(self._pool)

    def is_cached(self, block_id: int) -> bool:
        """Whether this block is indexed under some prefix hash."""
        return block_id in self._by_block

    def release(self, block_id: int) -> None:
        """Note that nothing references this block any more.

        It stays cached and becomes the newest eviction candidate rather than
        being reclaimed. A finished request's system-prompt blocks are the most
        likely blocks to be wanted next, and freeing them on the spot discards
        the best entries in the cache.
        """
        if block_id in self._by_block:
            self._pool[block_id] = None
            self._pool.move_to_end(block_id)

    def evict(self) -> int | None:
        """Reclaim the least recently used unreferenced block.

        Returns the block id, now un-indexed and safe to hand out, or ``None``
        when nothing can be reclaimed.
        """
        if not self._pool:
            return None
        block_id, _ = self._pool.popitem(last=False)
        block_hash = self._by_block.pop(block_id, None)
        if block_hash is not None:
            self._by_hash.pop(block_hash, None)
        self.stats.evictions += 1
        return block_id

    def forget(self, block_id: int) -> None:
        """Drop a block from the index entirely.

        For copy-on-write, where a block's contents are about to diverge from
        the prefix its hash claims to identify.
        """
        block_hash = self._by_block.pop(block_id, None)
        if block_hash is not None:
            self._by_hash.pop(block_hash, None)
        self._pool.pop(block_id, None)

    def reset(self) -> None:
        self._by_hash.clear()
        self._by_block.clear()
        self._pool.clear()
        self.stats = PrefixCacheStats()
