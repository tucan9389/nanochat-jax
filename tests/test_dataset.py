"""Smoke tests for nanochat_jax.dataset."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat_jax import dataset as dataset_module
from nanochat_jax.dataset import (
    MAX_SHARD,
    index_to_filename,
    list_parquet_files,
    parquets_iter_batched,
)


def test_index_to_filename_zero_padded():
    assert index_to_filename(0) == "shard_00000.parquet"
    assert index_to_filename(42) == "shard_00042.parquet"
    assert index_to_filename(MAX_SHARD) == f"shard_{MAX_SHARD:05d}.parquet"


def test_list_parquet_files_returns_sorted_full_paths(tmp_path: Path):
    # Build a synthetic data dir with three parquet files plus a .tmp.
    paths = [tmp_path / f for f in ("shard_00002.parquet", "shard_00001.parquet", "shard_00000.parquet")]
    for p in paths:
        p.write_bytes(b"")
    (tmp_path / "shard_00003.parquet.tmp").write_bytes(b"")  # excluded

    out = list_parquet_files(data_dir=str(tmp_path))
    assert out == sorted(str(p) for p in paths)


def test_list_parquet_files_legacy_warning(tmp_path: Path, capsys):
    """When the requested dir does not exist, fall back and (optionally) warn."""
    missing = tmp_path / "nonexistent"
    legacy = tmp_path / "base_data"
    legacy.mkdir()
    (legacy / "shard_00000.parquet").write_bytes(b"")

    # Patch base_dir for the fallback to land in our tmp.
    monkey_base = tmp_path
    saved_base = dataset_module.base_dir
    dataset_module.base_dir = str(monkey_base)
    try:
        out = list_parquet_files(data_dir=str(missing), warn_on_legacy=True)
    finally:
        dataset_module.base_dir = saved_base
    captured = capsys.readouterr()
    assert "DATASET UPGRADE REQUIRED" in captured.out
    assert any("shard_00000.parquet" in p for p in out)


def test_parquets_iter_batched_yields_text_lists(tmp_path: Path, monkeypatch):
    """Build a tiny parquet file with a 'text' column and iterate."""
    table = pa.table({"text": pa.array(["doc one", "doc two", "doc three"])})
    val_path = tmp_path / "shard_00000.parquet"
    pq.write_table(table, str(val_path), row_group_size=2)
    monkeypatch.setattr(dataset_module, "DATA_DIR", str(tmp_path))

    out = list(parquets_iter_batched("val", start=0, step=1))
    flat = [doc for batch in out for doc in batch]
    assert flat == ["doc one", "doc two", "doc three"]


def test_parquets_iter_batched_rejects_unknown_split():
    with pytest.raises(AssertionError, match="split must be"):
        list(parquets_iter_batched("test"))
