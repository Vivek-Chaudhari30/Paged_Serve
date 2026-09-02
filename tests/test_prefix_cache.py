"""Unit tests for the prefix cache: hashing, matching, and eviction.

Pure Python, no torch. The correctness question this module has to answer is
"does a hit mean these two requests really share this prefix", and that is
answerable without a model.
"""

from __future__ import annotations

import pytest

from pagedserve.memory.prefix_cache import (
    PrefixCache,
    PrefixCacheStats,
    block_hashes,
    chain_hash,
)

BLOCK_SIZE = 4


class TestChainHash:
    def test_is_deterministic(self):
        assert chain_hash(None, [1, 2, 3]) == chain_hash(None, [1, 2, 3])

    def test_is_stable_across_processes(self):
        """Not Python's hash(), which is salted per process.

        A cache keyed on a salted hash stops matching after a restart, and looks
        perfectly correct in any single-process test.
        """
        import subprocess
        import sys

        expected = chain_hash(None, [7, 8, 9])
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pagedserve.memory.prefix_cache import chain_hash;"
                "print(chain_hash(None, [7, 8, 9]))",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert int(result.stdout.strip()) == expected

    def test_different_tokens_give_different_hashes(self):
        assert chain_hash(None, [1, 2, 3]) != chain_hash(None, [1, 2, 4])

    def test_order_matters(self):
        assert chain_hash(None, [1, 2]) != chain_hash(None, [2, 1])

    def test_parent_changes_the_result(self):
        """The whole point: identical tokens under a different prefix differ."""
        assert chain_hash(None, [5, 6]) != chain_hash(99, [5, 6])
        assert chain_hash(1, [5, 6]) != chain_hash(2, [5, 6])

    def test_token_boundaries_are_unambiguous(self):
        # Fixed-width encoding, so [1, 256] cannot collide with some other pair
        # by concatenating to the same bytes.
        assert chain_hash(None, [1, 256]) != chain_hash(None, [257, 0])

    def test_handles_negative_tokens(self):
        assert chain_hash(None, [-1]) != chain_hash(None, [1])


class TestBlockHashes:
    def test_one_hash_per_full_block(self):
        tokens = list(range(12))
        assert len(block_hashes(tokens, 4)) == 3

    def test_trailing_partial_block_is_not_hashed(self):
        """Only sealed blocks are cacheable, because partial ones are mutable."""
        assert len(block_hashes(list(range(10)), 4)) == 2  # 10 = 2 blocks + 2
        assert block_hashes(list(range(3)), 4) == []

    def test_a_shared_prefix_produces_shared_hashes(self):
        shared = list(range(8))
        a = block_hashes(shared + [100, 101, 102, 103], 4)
        b = block_hashes(shared + [200, 201, 202, 203], 4)
        assert a[:2] == b[:2]
        assert a[2] != b[2]

    def test_the_chain_diverges_permanently_after_one_difference(self):
        """A late difference must not leave later blocks matching.

        Identical tokens at the same offset still have different K and V if
        anything before them differed, so a hash that matched there would hand
        one request another's context.
        """
        a = block_hashes([0, 0, 0, 0] + [1, 1, 1, 1] + [9, 9, 9, 9], 4)
        b = block_hashes([0, 0, 0, 1] + [1, 1, 1, 1] + [9, 9, 9, 9], 4)
        assert a[0] != b[0]
        assert a[1] != b[1], "identical tokens under a different prefix must differ"
        assert a[2] != b[2]

    def test_parent_continues_an_existing_chain(self):
        whole = block_hashes(list(range(8)), 4)
        first_half = block_hashes(list(range(4)), 4)
        continued = block_hashes(list(range(4, 8)), 4, parent=first_half[0])
        assert continued == [whole[1]]

    def test_rejects_a_nonsense_block_size(self):
        with pytest.raises(ValueError, match="block_size must be positive"):
            block_hashes([1, 2], 0)


class TestLookupAndMatch:
    def test_miss_on_an_empty_cache(self):
        assert PrefixCache(BLOCK_SIZE).lookup(123) is None

    def test_hit_after_insert(self):
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(123, block_id=7)
        assert cache.lookup(123) == 7

    def test_match_returns_the_leading_run(self):
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.insert(11, 1)
        assert cache.match([10, 11, 12]) == [0, 1]

    def test_match_stops_at_the_first_miss(self):
        """A later hit cannot be used across a gap.

        The sequence's blocks must be contiguous from position zero; a gap would
        mean attending over KV for tokens the request never had.
        """
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.insert(12, 2)  # present, but block 1 is missing
        assert cache.match([10, 11, 12]) == [0]

    def test_match_on_empty_input(self):
        assert PrefixCache(BLOCK_SIZE).match([]) == []

    def test_disabled_cache_never_hits(self):
        cache = PrefixCache(BLOCK_SIZE, enabled=False)
        cache.insert(123, 7)
        assert cache.lookup(123) is None
        assert cache.match([123]) == []

    def test_insert_keeps_the_incumbent(self):
        # Both blocks hold identical KV by construction, and keeping the first
        # avoids invalidating references other sequences already hold.
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.insert(10, 5)
        assert cache.lookup(10) == 0


class TestEviction:
    def test_released_blocks_stay_cached(self):
        """A finished request's blocks are the most likely to be wanted next."""
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.release(0)
        assert cache.lookup(10) == 0
        assert len(cache) == 1

    def test_evicts_least_recently_used_first(self):
        cache = PrefixCache(BLOCK_SIZE)
        for i in range(3):
            cache.insert(10 + i, i)
            cache.release(i)
        assert cache.evict() == 0
        assert cache.evict() == 1
        assert cache.evict() == 2

    def test_a_hit_refreshes_recency(self):
        cache = PrefixCache(BLOCK_SIZE)
        for i in range(3):
            cache.insert(10 + i, i)
            cache.release(i)
        cache.lookup(10)  # block 0 referenced again
        cache.release(0)  # and released, so it is now the newest candidate
        assert cache.evict() == 1

    def test_a_referenced_block_is_not_evictable(self):
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        # Never released, so nothing is reclaimable even though it is cached.
        assert cache.num_reclaimable == 0
        assert cache.evict() is None

    def test_evicting_removes_the_index_entry(self):
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.release(0)
        assert cache.evict() == 0
        assert cache.lookup(10) is None
        assert len(cache) == 0

    def test_evict_on_empty_pool(self):
        assert PrefixCache(BLOCK_SIZE).evict() is None

    def test_forget_removes_a_block_entirely(self):
        """For copy-on-write: the contents are about to stop matching the hash."""
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.release(0)
        cache.forget(0)
        assert cache.lookup(10) is None
        assert cache.num_reclaimable == 0

    def test_forget_is_safe_on_an_unknown_block(self):
        PrefixCache(BLOCK_SIZE).forget(999)

    def test_release_is_safe_on_an_uncached_block(self):
        cache = PrefixCache(BLOCK_SIZE)
        cache.release(999)
        assert cache.num_reclaimable == 0


class TestStats:
    def test_hit_rate_is_none_before_any_lookup(self):
        # Not 0.0 -- nothing has been asked, so there is no rate to report.
        assert PrefixCacheStats().hit_rate is None

    def test_counts_hits_and_tokens_saved(self):
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.insert(11, 1)
        cache.match([10, 11, 12])
        assert cache.stats.lookups == 3
        assert cache.stats.hits == 2
        assert cache.stats.blocks_reused == 2
        # Tokens saved is what makes a hit rate interpretable: hits on a
        # 16-token block and on a 2000-token prefix count the same otherwise.
        assert cache.stats.tokens_saved == 2 * BLOCK_SIZE
        assert cache.stats.hit_rate == pytest.approx(2 / 3)

    def test_counts_evictions(self):
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.release(0)
        cache.evict()
        assert cache.stats.evictions == 1

    def test_serializes_for_a_result_file(self):
        import json

        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.match([10])
        assert json.loads(json.dumps(cache.stats.to_dict()))["hits"] == 1

    def test_reset_clears_everything(self):
        cache = PrefixCache(BLOCK_SIZE)
        cache.insert(10, 0)
        cache.match([10])
        cache.reset()
        assert len(cache) == 0
        assert cache.stats.hits == 0


class TestRealisticSharing:
    def test_a_shared_system_prompt_matches_across_requests(self):
        """The case the whole scheme exists for."""
        system = list(range(100, 116))  # 4 blocks at block_size 4
        cache = PrefixCache(BLOCK_SIZE)

        first = block_hashes(system + [1, 2, 3, 4], BLOCK_SIZE)
        for index, block_hash in enumerate(first):
            cache.insert(block_hash, index)

        second = block_hashes(system + [9, 9, 9, 9], BLOCK_SIZE)
        matched = cache.match(second)
        assert matched == [0, 1, 2, 3]
        assert cache.stats.tokens_saved == 16

    def test_a_different_system_prompt_shares_nothing(self):
        cache = PrefixCache(BLOCK_SIZE)
        for index, block_hash in enumerate(block_hashes(list(range(16)), BLOCK_SIZE)):
            cache.insert(block_hash, index)
        other = block_hashes([999] + list(range(1, 16)), BLOCK_SIZE)
        assert cache.match(other) == []
