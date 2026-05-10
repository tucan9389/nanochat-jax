"""Minimal tests for nanochat_jax.golden_loader.

Builds tiny synthetic ``.npz`` files in a temp dir and exercises the loader's
prefix routing, dtype restoration (including bf16 view), and required-key
validation. Round-trip tests against real generator output live alongside
``scripts/generate_golden.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from nanochat_jax.golden_loader import GoldenSnapshot, load_golden


def _write_npz(path: Path, arrays: dict, dtype_map: dict, meta: dict) -> None:
    """Save a synthetic golden .npz with the reserved metadata keys."""
    payload = dict(arrays)
    payload["_dtype_map"] = np.array(json.dumps(dtype_map))
    payload["_meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **payload)


def test_load_minimal_snapshot(tmp_path: Path):
    """A tiny snapshot with weights + tensors + input_ids loads correctly."""
    arrays = {
        "weight/transformer.wte.weight": np.zeros((4, 8), dtype=np.float32),
        "tensor/x": np.ones((1, 2, 8), dtype=np.float32),
        "input/input_ids": np.array([[0, 1]], dtype=np.int64),
    }
    dtype_map = {k: str(v.dtype) for k, v in arrays.items()}
    meta = {"depth": 1, "seed": 42}
    path = tmp_path / "golden_min.npz"
    _write_npz(path, arrays, dtype_map, meta)

    snap = load_golden(path)
    assert isinstance(snap, GoldenSnapshot)
    assert set(snap.weights.keys()) == {"transformer.wte.weight"}
    assert set(snap.tensors.keys()) == {"x"}
    assert snap.input_ids.shape == (1, 2)
    assert snap.targets is None
    assert snap.meta == meta
    # Optional fields default to empty / None.
    assert snap.grads == {}
    assert snap.train_loss_curve is None
    assert snap.bpb_eval_total_bytes is None


def test_dtype_restoration_bf16(tmp_path: Path):
    """bf16 arrays survive np.savez via the |V2 view + dtype_map restoration."""
    bf16_arr = np.array([[1.0, 2.0]], dtype=ml_dtypes.bfloat16)
    arrays = {
        "weight/transformer.wte.weight": bf16_arr,
        "input/input_ids": np.array([[0]], dtype=np.int64),
    }
    dtype_map = {
        "weight/transformer.wte.weight": "bfloat16",
        "input/input_ids": "int64",
    }
    path = tmp_path / "golden_bf16.npz"
    _write_npz(path, arrays, dtype_map, {})

    snap = load_golden(path)
    restored = snap.weights["transformer.wte.weight"]
    assert restored.dtype == np.dtype(ml_dtypes.bfloat16), restored.dtype
    np.testing.assert_array_equal(restored, bf16_arr)


def test_train_loop_prefixes(tmp_path: Path):
    """Per-step training-loop prefixes route to their fields."""
    n_steps = 3
    arrays = {
        "weight/transformer.wte.weight": np.zeros((4, 8), dtype=np.float32),
        "input/input_ids": np.array([[0]], dtype=np.int64),
        "train/loss_curve": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "train/per_step_grads/transformer.wte.weight": np.zeros(
            (n_steps, 4, 8), dtype=np.float32
        ),
        "train/per_step_weights_after/transformer.wte.weight": np.ones(
            (n_steps, 4, 8), dtype=np.float32
        ),
        "train/per_step_optim_state_after/transformer.wte.weight/exp_avg": np.zeros(
            (n_steps, 4, 8), dtype=np.float32
        ),
        "train/weights_after_N/transformer.wte.weight": np.ones((4, 8), dtype=np.float32),
        "train/optim_state_after_N/transformer.wte.weight/exp_avg": np.zeros(
            (4, 8), dtype=np.float32
        ),
    }
    dtype_map = {k: str(v.dtype) for k, v in arrays.items()}
    path = tmp_path / "golden_train.npz"
    _write_npz(path, arrays, dtype_map, {})

    snap = load_golden(path)
    assert snap.train_loss_curve is not None
    assert snap.train_loss_curve.shape == (n_steps,)
    assert "transformer.wte.weight" in snap.train_per_step_grads
    assert snap.train_per_step_grads["transformer.wte.weight"].shape == (n_steps, 4, 8)
    assert "transformer.wte.weight" in snap.train_per_step_weights_after
    assert "transformer.wte.weight/exp_avg" in snap.train_per_step_optim_state_after
    assert "transformer.wte.weight" in snap.train_weights_after_N
    assert "transformer.wte.weight/exp_avg" in snap.train_optim_state_after_N


def test_bpb_eval_prefixes(tmp_path: Path):
    """BPB-eval prefixes route to typed fields with correct casts."""
    arrays = {
        "weight/transformer.wte.weight": np.zeros((4, 8), dtype=np.float32),
        "input/input_ids": np.array([[0]], dtype=np.int64),
        "bpb_eval/input_ids": np.array([[[0, 1]]], dtype=np.int64),
        "bpb_eval/targets": np.array([[[1, -1]]], dtype=np.int64),
        "bpb_eval/token_bytes": np.array([1, 2, 3, 4], dtype=np.int32),
        "bpb_eval/per_step_loss_2d": np.array([[0.1, 0.2]], dtype=np.float32),
        "bpb_eval/total_nats": np.array(1.5, dtype=np.float32),
        "bpb_eval/total_bytes": np.array(10, dtype=np.int64),
        "bpb_eval/expected_bpb": np.array(0.21, dtype=np.float32),
    }
    dtype_map = {k: str(v.dtype) for k, v in arrays.items()}
    path = tmp_path / "golden_bpb.npz"
    _write_npz(path, arrays, dtype_map, {})

    snap = load_golden(path)
    assert snap.bpb_eval_input_ids.shape == (1, 1, 2)
    assert snap.bpb_eval_targets.shape == (1, 1, 2)
    assert snap.bpb_eval_token_bytes.shape == (4,)
    assert snap.bpb_eval_per_step_loss_2d.shape == (1, 2)
    assert snap.bpb_eval_total_nats == pytest.approx(1.5)
    assert snap.bpb_eval_total_bytes == 10
    assert snap.bpb_eval_expected_bpb == pytest.approx(0.21)


def test_missing_dtype_map_raises(tmp_path: Path):
    """A .npz without _dtype_map is rejected."""
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        **{"weight/transformer.wte.weight": np.zeros((1, 1), dtype=np.float32)},
    )
    with pytest.raises(ValueError, match="missing reserved key '_dtype_map'"):
        load_golden(path)


def test_missing_meta_raises(tmp_path: Path):
    """A .npz without _meta is rejected."""
    path = tmp_path / "no_meta.npz"
    np.savez_compressed(
        path,
        **{
            "weight/transformer.wte.weight": np.zeros((1, 1), dtype=np.float32),
            "_dtype_map": np.array(json.dumps({"weight/transformer.wte.weight": "float32"})),
        },
    )
    with pytest.raises(ValueError, match="missing reserved key '_meta'"):
        load_golden(path)


def test_missing_input_ids_raises(tmp_path: Path):
    """A .npz without input/input_ids is rejected even if other keys are present."""
    arrays = {"weight/transformer.wte.weight": np.zeros((1, 1), dtype=np.float32)}
    dtype_map = {k: str(v.dtype) for k, v in arrays.items()}
    path = tmp_path / "no_input.npz"
    _write_npz(path, arrays, dtype_map, {})
    with pytest.raises(ValueError, match="missing required key 'input/input_ids'"):
        load_golden(path)


def test_unknown_prefix_raises(tmp_path: Path):
    """An unknown key prefix raises ValueError so loader/generator stay in sync."""
    arrays = {
        "input/input_ids": np.array([[0]], dtype=np.int64),
        "unknown_prefix/foo": np.zeros((1,), dtype=np.float32),
    }
    dtype_map = {k: str(v.dtype) for k, v in arrays.items()}
    path = tmp_path / "unknown.npz"
    _write_npz(path, arrays, dtype_map, {})
    with pytest.raises(ValueError, match="unknown key prefix"):
        load_golden(path)


def test_dtype_map_key_missing_raises(tmp_path: Path):
    """An array key without a _dtype_map entry is rejected."""
    arrays = {
        "input/input_ids": np.array([[0]], dtype=np.int64),
        "weight/transformer.wte.weight": np.zeros((1, 1), dtype=np.float32),
    }
    dtype_map = {"input/input_ids": "int64"}  # missing weight entry
    path = tmp_path / "no_dtype_entry.npz"
    _write_npz(path, arrays, dtype_map, {})
    with pytest.raises(ValueError, match="not in _dtype_map"):
        load_golden(path)
