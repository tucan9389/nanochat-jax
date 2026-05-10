"""Smoke tests for nanochat_jax.dataloader (BOS-aligned best-fit packing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat_jax import dataset as dataset_module
from nanochat_jax.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)


class _FakeTokenizer:
    """Minimal tokenizer protocol used by the dataloader."""

    BOS = 0

    def get_bos_token_id(self):
        return self.BOS

    def encode(self, texts, prepend=None, num_threads=None):
        # Each character becomes a token (id = ord); BOS is prepended.
        out = []
        for t in texts:
            ids = [self.BOS] if prepend is not None else []
            ids.extend([1 + (ord(c) % 100) for c in t])
            out.append(ids)
        return out


@pytest.fixture
def fake_data(tmp_path: Path, monkeypatch):
    """Create a tiny parquet 'val' dataset with three short documents."""
    table = pa.table({"text": pa.array(["alpha", "beta gamma", "delta"])})
    pq.write_table(table, str(tmp_path / "shard_00000.parquet"), row_group_size=2)
    monkeypatch.setattr(dataset_module, "DATA_DIR", str(tmp_path))
    return tmp_path


def test_dataloader_yields_correct_shapes(fake_data):
    tok = _FakeTokenizer()
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, B=2, T=8, split="val", buffer_size=2,
    )
    inputs, targets = next(loader)
    assert inputs.shape == (2, 8)
    assert targets.shape == (2, 8)
    assert inputs.dtype == np.int64
    assert targets.dtype == np.int64


def test_dataloader_targets_are_inputs_shifted_by_one(fake_data):
    tok = _FakeTokenizer()
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tok, B=2, T=8, split="val", buffer_size=2,
    )
    inputs, targets = next(loader)
    # targets[t] == inputs[t+1] by construction (row shifted by 1).
    # Reconstruct the (B, T+1) row: cat(inputs[:, :], targets[:, -1:]).
    full = np.concatenate([inputs, targets[:, -1:]], axis=1)
    np.testing.assert_array_equal(targets, full[:, 1:])


def test_dataloader_state_includes_pq_rg_epoch(fake_data):
    tok = _FakeTokenizer()
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tok, B=1, T=8, split="val", buffer_size=2,
    )
    _, _, state = next(loader)
    assert set(state.keys()) == {"pq_idx", "rg_idx", "epoch"}
    assert state["epoch"] >= 1


def test_dataloader_rejects_unknown_split(fake_data):
    tok = _FakeTokenizer()
    with pytest.raises(AssertionError, match="split must be"):
        next(tokenizing_distributed_data_loader_bos_bestfit(
            tok, B=1, T=8, split="test",
        ))
