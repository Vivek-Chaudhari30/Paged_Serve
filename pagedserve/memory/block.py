"""Physical blocks and per-sequence block tables.

The OS analogy this project is named for, made concrete:

===========================  ==========================================
Virtual memory               PagedServe
===========================  ==========================================
Process                      Sequence (one request)
Page frame                   KV block (``block_size`` token slots)
Page table                   ``BlockTable`` (logical index -> block id)
Page fault -> allocate frame Sequence crosses a block boundary
Shared library page          Shared prefix block (Phase 5)
Reference counting           ``PhysicalBlock.ref_count``
Copy-on-write                Fork a shared block when writers diverge
===========================  ==========================================

The payoff is a bound on internal fragmentation. A contiguous cache wastes up
to ``max_seq_len - actual_len`` tokens per sequence; a paged one wastes at most
``block_size - 1``, and only ever in the last block. With ``block_size=16`` that
is 15 tokens instead of roughly 1900.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BlockTable", "PhysicalBlock"]


@dataclass
class PhysicalBlock:
    """One fixed-size chunk of the KV cache.

    ``ref_count`` is here from Phase 2 even though nothing shares blocks until
    Phase 5. Retrofitting reference counts into an allocator that assumed
    exclusive ownership means auditing every path that frees anything; adding
    them now costs an integer.
    """

    block_id: int
    ref_count: int = 0

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


class BlockTable:
    """A sequence's logical-to-physical block mapping.

    Logical token *i* lives in physical block ``blocks[i // block_size]`` at
    offset ``i % block_size``. That is the whole indirection: the sequence sees
    a contiguous run of positions, while the blocks behind it can be scattered
    anywhere in the cache and can be handed out one at a time from a free list.
    """

    __slots__ = ("block_size", "blocks")

    def __init__(self, block_size: int, blocks: list[int] | None = None) -> None:
        self.block_size = block_size
        self.blocks: list[int] = list(blocks) if blocks else []

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockTable):
            return NotImplemented
        return self.block_size == other.block_size and self.blocks == other.blocks

    def __repr__(self) -> str:
        return f"BlockTable(block_size={self.block_size}, blocks={self.blocks})"

    @property
    def capacity(self) -> int:
        """Token slots this table can address, used or not."""
        return len(self.blocks) * self.block_size

    def blocks_needed(self, num_tokens: int) -> int:
        """Blocks required to hold ``num_tokens``: ``ceil(n / block_size)``."""
        if num_tokens <= 0:
            return 0
        return -(-num_tokens // self.block_size)

    def slot(self, logical_position: int) -> int:
        """Flat physical slot index for a logical token position.

        The flat index is what the write path scatters into, so this is the one
        place the block-to-slot arithmetic lives.
        """
        if logical_position < 0:
            raise IndexError(f"negative logical position {logical_position}")
        block_index = logical_position // self.block_size
        if block_index >= len(self.blocks):
            raise IndexError(
                f"logical position {logical_position} needs block {block_index}, "
                f"but this table has only {len(self.blocks)}"
            )
        return self.blocks[block_index] * self.block_size + (logical_position % self.block_size)

    def slots(self, start: int, count: int) -> list[int]:
        """Flat slot indices for ``count`` consecutive logical positions."""
        return [self.slot(start + i) for i in range(count)]

    def append(self, block_id: int) -> None:
        self.blocks.append(block_id)

    def copy(self) -> BlockTable:
        """A shallow copy sharing the same physical blocks.

        Forking a sequence copies this list of integers and bumps refcounts; the
        KV itself is shared until the two writers diverge. That is why ``n>1``
        sampling costs almost nothing in Phase 6.
        """
        return BlockTable(self.block_size, self.blocks)
