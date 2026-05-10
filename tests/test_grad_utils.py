"""Smoke tests for nanochat_jax.grad_utils."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nanochat_jax.gpt import GPT, GPTConfig
from nanochat_jax.grad_utils import (
    cast_grad_dict_to_fp32,
    compute_loss_and_grad,
    grad_dict_dtype_summary,
    nnx_state_to_flat_dict,
)


def _tiny_model(**overrides) -> GPT:
    base = dict(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
    )
    base.update(overrides)
    cfg = GPTConfig(**base)
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    return m


def test_compute_loss_and_grad_returns_scalar_and_flat_dict():
    m = _tiny_model()
    idx = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 6]])

    loss, grads = compute_loss_and_grad(m, idx, targets)
    assert loss.shape == ()
    assert isinstance(grads, dict) and len(grads) > 0
    # Sample expected keys.
    assert "transformer.wte.embedding" in grads
    assert "lm_head.kernel" in grads
    assert "smear_lambda" in grads


def test_grad_keys_align_with_param_keys():
    m = _tiny_model()
    state = nnx.state(m, nnx.Param)
    param_keys = set(nnx_state_to_flat_dict(state).keys())

    idx = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 6]])
    _, grads = compute_loss_and_grad(m, idx, targets)
    assert set(grads.keys()) == param_keys


def test_grad_shapes_match_param_shapes():
    m = _tiny_model()
    state = nnx.state(m, nnx.Param)
    params = nnx_state_to_flat_dict(state)  # values are already raw arrays

    idx = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 6]])
    _, grads = compute_loss_and_grad(m, idx, targets)
    for k, p in params.items():
        assert grads[k].shape == p.shape, (
            f"shape mismatch at {k}: param={p.shape}, grad={grads[k].shape}"
        )


def test_compute_loss_and_grad_is_deterministic():
    idx = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 6]])

    a = _tiny_model()
    b = _tiny_model()
    loss_a, grads_a = compute_loss_and_grad(a, idx, targets)
    loss_b, grads_b = compute_loss_and_grad(b, idx, targets)
    assert float(loss_a) == pytest.approx(float(loss_b))
    for k in grads_a:
        np.testing.assert_array_equal(grads_a[k], grads_b[k])


def test_grad_dtype_summary_all_fp32_in_default_mode():
    m = _tiny_model()
    idx = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 6]])
    _, grads = compute_loss_and_grad(m, idx, targets)
    summary = grad_dict_dtype_summary(grads)
    assert set(summary.keys()) == {"float32"}, summary


def test_grad_dtype_summary_in_bf16_mode_keeps_fp32_params():
    """Param leaves stay fp32 even with compute_dtype=bf16, so grads stay fp32."""
    m = _tiny_model(compute_dtype=jnp.bfloat16)
    idx = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 6]])
    _, grads = compute_loss_and_grad(m, idx, targets)
    summary = grad_dict_dtype_summary(grads)
    assert "float32" in summary


def test_cast_grad_dict_to_fp32_handles_numpy_and_jax():
    mixed = {
        "a": np.zeros((2, 2), dtype=np.float32),
        "b": jnp.zeros((2, 2), dtype=jnp.bfloat16),
    }
    out = cast_grad_dict_to_fp32(mixed)
    assert all(v.dtype == jnp.float32 for v in out.values())
    assert out["a"].shape == (2, 2)
    assert out["b"].shape == (2, 2)


def test_num_scaling_params_works_after_grad_utils_present():
    """num_scaling_params imports nnx_state_to_flat_dict from this module."""
    m = _tiny_model()
    counts = m.num_scaling_params()
    assert counts["total"] > 0
    assert counts["total"] == (
        counts["wte"]
        + counts["value_embeds"]
        + counts["lm_head"]
        + counts["transformer_matrices"]
        + counts["scalars"]
    )
