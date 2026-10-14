"""CUDA graph decode runner: bucket selection and padded inputs on CPU, replay on GPU."""

from __future__ import annotations

import pytest
import torch

from pagedserve.memory.block_manager import BlockManager
from pagedserve.sequence import Sequence
from pagedserve.worker.cuda_graph import build_decode_inputs, default_batch_sizes


def test_default_buckets():
    assert default_batch_sizes(1) == [1]
    assert default_batch_sizes(6) == [1, 2, 4, 6]
    assert default_batch_sizes(32) == [1, 2, 4, 8, 16, 24, 32]
    assert default_batch_sizes(20) == [1, 2, 4, 8, 16, 20]


def _decoding(manager: BlockManager, seq_id: int, prompt_len: int) -> Sequence:
    sequence = Sequence(seq_id=seq_id, prompt_token_ids=list(range(prompt_len)), max_tokens=8)
    manager.allocate(seq_id, prompt_len + 1)
    sequence.num_computed_tokens = prompt_len
    sequence.append_token(100 + seq_id)
    return sequence


def test_build_decode_inputs_pads_into_the_trash_block():
    manager = BlockManager(16, 4, watermark=0.0)
    a = _decoding(manager, 0, 5)
    b = _decoding(manager, 1, 2)
    host = build_decode_inputs([a, b], manager, bucket=4, max_blocks=3)
    trash = manager.trash_block_id
    assert host["input_ids"].tolist() == [[a.all_token_ids[5], b.all_token_ids[2], 0, 0]]
    assert host["positions"].tolist() == [[5, 2, 0, 0]]
    assert host["seq_lens"].tolist() == [6, 3, 1, 1]
    assert host["seq_lens"].dtype == torch.int32
    assert host["slot_mapping"].tolist()[:2] == [
        manager.slots(0, 5, 1)[0],
        manager.slots(1, 2, 1)[0],
    ]
    assert host["slot_mapping"].tolist()[2:] == [trash * 4] * 2
    table_a = list(manager.block_table(0))
    assert host["block_tables"][0, : len(table_a)].tolist() == table_a
    assert host["block_tables"][2:].eq(trash).all()


def test_build_decode_inputs_rejects_overflow():
    manager = BlockManager(16, 4, watermark=0.0)
    seqs = [_decoding(manager, i, 3) for i in range(3)]
    with pytest.raises(ValueError):
        build_decode_inputs(seqs, manager, bucket=2, max_blocks=4)
