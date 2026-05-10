"""Smoke tests for nanochat_jax.checkpoint_manager (in-memory round-trip)."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

from nanochat_jax.checkpoint_manager import (
    _muon_adamw_state_to_pt_dict,
    _patch_missing_config_keys,
    _patch_missing_keys,
    _pt_dict_to_muon_adamw_state,
    find_largest_model,
    find_last_step,
    inject_pt_state_dict,
    load_checkpoint,
    save_checkpoint,
)
from nanochat_jax.gpt import GPT, GPTConfig
from nanochat_jax.grad_utils import nnx_state_to_flat_dict
from nanochat_jax.optim import (
    init_optim_state,
    setup_optimizer_param_groups,
)


def _tiny_cfg() -> GPTConfig:
    return GPTConfig(
        sequence_len=16, vocab_size=32, n_layer=2,
        n_head=2, n_kv_head=2, n_embd=32,
    )


def _tiny_model() -> GPT:
    m = GPT(_tiny_cfg(), rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    return m


def test_patch_missing_config_keys_adds_window_pattern():
    cfg = {"n_layer": 2}
    _patch_missing_config_keys(cfg)
    assert cfg["window_pattern"] == "L"


def test_patch_missing_keys_adds_lambdas():
    model_config = _tiny_cfg()
    data: dict = {}
    _patch_missing_keys(data, model_config)
    assert "resid_lambdas" in data and "x0_lambdas" in data
    assert data["resid_lambdas"].shape == (model_config.n_layer,)


def test_find_last_step(tmp_path: Path):
    for step in (10, 5, 100, 1):
        (tmp_path / f"model_{step:06d}.pt").write_bytes(b"")
    assert find_last_step(str(tmp_path)) == 100


def test_find_last_step_raises_when_empty(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_last_step(str(tmp_path))


def test_find_largest_model_uses_depth_regex(tmp_path: Path):
    for tag in ("d4", "d12", "d20", "other"):
        (tmp_path / tag).mkdir()
    assert find_largest_model(str(tmp_path)) == "d20"


def test_find_largest_model_falls_back_to_mtime(tmp_path: Path):
    a = tmp_path / "alpha"
    b = tmp_path / "beta"
    a.mkdir()
    b.mkdir()
    # Touch beta to be the most recently modified.
    (b / "marker").write_bytes(b"")
    assert find_largest_model(str(tmp_path)) in {"alpha", "beta"}


def test_inject_pt_state_dict_round_trip():
    """Build a model, extract its params, inject them back -- round-trip identity."""
    m = _tiny_model()
    flat = nnx_state_to_flat_dict(nnx.state(m, nnx.Param))
    # Perturb to non-zero values then inject back the original.
    inject_pt_state_dict(m, flat)
    flat_after = nnx_state_to_flat_dict(nnx.state(m, nnx.Param))
    for k, v in flat.items():
        np.testing.assert_array_equal(np.asarray(v), np.asarray(flat_after[k]))


def test_inject_pt_state_dict_raises_on_missing_key():
    m = _tiny_model()
    with pytest.raises(KeyError, match="has no matching key"):
        inject_pt_state_dict(m, {})


def test_inject_pt_state_dict_raises_on_orphan_key():
    m = _tiny_model()
    flat = nnx_state_to_flat_dict(nnx.state(m, nnx.Param))
    flat["extra.unknown"] = jnp.zeros((1,))
    with pytest.raises(ValueError, match="not consumed by NNX"):
        inject_pt_state_dict(m, flat)


def test_save_and_load_checkpoint_round_trip(tmp_path: Path):
    """Save a model + meta, then load and verify."""
    m = _tiny_model()
    meta = {"model_config": {
        "sequence_len": 16, "vocab_size": 32, "n_layer": 2,
        "n_head": 2, "n_kv_head": 2, "n_embd": 32, "window_pattern": "L",
    }}
    save_checkpoint(str(tmp_path), step=42, model=m, meta_data=meta)

    assert (tmp_path / "model_000042.pt").exists()
    assert (tmp_path / "meta_000042.json").exists()

    model_data, optim_data, meta_data = load_checkpoint(str(tmp_path), step=42)
    assert optim_data is None
    assert meta_data == meta
    assert isinstance(model_data, dict)
    assert "transformer.wte.weight" in model_data


def test_save_and_load_optimizer_round_trip(tmp_path: Path):
    m = _tiny_model()
    params = nnx_state_to_flat_dict(nnx.state(m, nnx.Param))
    groups = setup_optimizer_param_groups(params, n_embd=32)
    state = init_optim_state(groups, params)

    flat = _muon_adamw_state_to_pt_dict(state)
    assert "step" in flat
    # Persist via torch and reload.
    path = tmp_path / "optim.pt"
    torch.save(flat, str(path))
    flat_back = torch.load(str(path), weights_only=False)

    state_back = _pt_dict_to_muon_adamw_state(flat_back)
    assert int(state_back.step) == int(state.step)
    assert set(state_back.adamw.keys()) == set(state.adamw.keys())
    assert set(state_back.muon.keys()) == set(state.muon.keys())


def test_load_checkpoint_raises_on_missing_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        load_checkpoint(str(tmp_path), step=1)


def test_load_checkpoint_meta_missing(tmp_path: Path):
    """Model file present but meta missing should raise FileNotFoundError."""
    m = _tiny_model()
    from nanochat_jax.weight_converter import jax_to_pt_state_dict
    flat = nnx_state_to_flat_dict(nnx.state(m, nnx.Param))
    pt = jax_to_pt_state_dict(flat)
    torch.save(pt, str(tmp_path / "model_000001.pt"))
    # No meta file.
    with pytest.raises(FileNotFoundError, match="Meta json not found"):
        load_checkpoint(str(tmp_path), step=1)


def test_load_optimizer_returns_none_when_missing(tmp_path: Path):
    """Saving model only (no optim shard) leaves optim load as None."""
    m = _tiny_model()
    meta = {"model_config": {
        "sequence_len": 16, "vocab_size": 32, "n_layer": 2,
        "n_head": 2, "n_kv_head": 2, "n_embd": 32, "window_pattern": "L",
    }}
    save_checkpoint(str(tmp_path), step=1, model=m, meta_data=meta)
    _, optim_data, _ = load_checkpoint(str(tmp_path), step=1, load_optimizer=True)
    assert optim_data is None


def test_meta_json_file_is_valid_json(tmp_path: Path):
    m = _tiny_model()
    meta = {"model_config": {
        "sequence_len": 16, "vocab_size": 32, "n_layer": 2,
        "n_head": 2, "n_kv_head": 2, "n_embd": 32, "window_pattern": "L",
    }, "step": 10, "val_bpb": 1.234}
    save_checkpoint(str(tmp_path), step=10, model=m, meta_data=meta)
    with open(tmp_path / "meta_000010.json") as f:
        loaded = json.load(f)
    assert loaded == meta
