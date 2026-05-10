"""Tests for nanochat_jax.ops against PyTorch reference (CPU)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nanochat_jax.ops import apply_rotary_emb, relu_squared, rms_norm, softcap


def _np_random(shape, *, seed, scale=1.0, dtype=np.float32):
    return (np.random.RandomState(seed).randn(*shape) * scale).astype(dtype)


def _pt_apply_rotary_emb(x, cos, sin):
    """nanochat reference apply_rotary_emb (PyTorch verbatim)."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def test_rms_norm_fp32_matches_pytorch():
    np_x = _np_random((2, 8, 32), seed=42)
    pt_y = F.rms_norm(torch.from_numpy(np_x), (np_x.shape[-1],)).numpy()
    jx_y = np.asarray(rms_norm(jnp.asarray(np_x)))
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)


def test_rms_norm_bf16_matches_pytorch():
    np_x = _np_random((2, 8, 32), seed=43)
    pt_y = F.rms_norm(
        torch.from_numpy(np_x).to(torch.bfloat16), (np_x.shape[-1],)
    ).to(torch.float32).numpy()
    jx_y = np.asarray(
        rms_norm(jnp.asarray(np_x).astype(jnp.bfloat16)).astype(jnp.float32)
    )
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-2, rtol=1e-2)


def test_rms_norm_eps_default_finfo():
    # Small magnitudes make eps load-bearing; mismatched eps would diverge.
    np_x = _np_random((2, 4, 16), seed=7, scale=1e-3)
    pt_y = F.rms_norm(torch.from_numpy(np_x), (np_x.shape[-1],)).numpy()
    jx_y = np.asarray(rms_norm(jnp.asarray(np_x)))
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("last_dim", [12, 24, 32, 64, 128, 256])
def test_rms_norm_lastdim_variation(last_dim):
    np_x = _np_random((2, 8, last_dim), seed=last_dim)
    pt_y = F.rms_norm(torch.from_numpy(np_x), (last_dim,)).numpy()
    jx_y = np.asarray(rms_norm(jnp.asarray(np_x)))
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)


def test_apply_rotary_emb_fp32():
    B, T, H, D = 2, 8, 4, 16
    np_x = _np_random((B, T, H, D), seed=42)
    np_cos = _np_random((1, T, 1, D // 2), seed=43)
    np_sin = _np_random((1, T, 1, D // 2), seed=44)

    pt_y = _pt_apply_rotary_emb(
        torch.from_numpy(np_x), torch.from_numpy(np_cos), torch.from_numpy(np_sin)
    ).numpy()
    jx_y = np.asarray(
        apply_rotary_emb(jnp.asarray(np_x), jnp.asarray(np_cos), jnp.asarray(np_sin))
    )
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)


def test_apply_rotary_emb_bf16():
    B, T, H, D = 2, 8, 4, 16
    np_x = _np_random((B, T, H, D), seed=42)
    np_cos = _np_random((1, T, 1, D // 2), seed=43)
    np_sin = _np_random((1, T, 1, D // 2), seed=44)

    pt_y = _pt_apply_rotary_emb(
        torch.from_numpy(np_x).to(torch.bfloat16),
        torch.from_numpy(np_cos).to(torch.bfloat16),
        torch.from_numpy(np_sin).to(torch.bfloat16),
    ).to(torch.float32).numpy()

    jx_y = np.asarray(
        apply_rotary_emb(
            jnp.asarray(np_x).astype(jnp.bfloat16),
            jnp.asarray(np_cos).astype(jnp.bfloat16),
            jnp.asarray(np_sin).astype(jnp.bfloat16),
        ).astype(jnp.float32)
    )
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-2, rtol=1e-2)


def test_apply_rotary_emb_special_rotations():
    # Identity rotation (cos=1, sin=0) returns the input.
    B, T, H, D = 1, 1, 1, 4
    np_x = _np_random((B, T, H, D), seed=99)
    cos_id = jnp.ones((1, T, 1, D // 2))
    sin_id = jnp.zeros((1, T, 1, D // 2))
    y_id = apply_rotary_emb(jnp.asarray(np_x), cos_id, sin_id)
    np.testing.assert_array_equal(np.asarray(y_id), np_x)

    # Quarter rotation (cos=0, sin=1) maps (x1, x2) -> (x2, -x1).
    cos_q = jnp.zeros((1, T, 1, D // 2))
    sin_q = jnp.ones((1, T, 1, D // 2))
    y_q = np.asarray(apply_rotary_emb(jnp.asarray(np_x), cos_q, sin_q))
    expected = np.concatenate([np_x[..., D // 2:], -np_x[..., : D // 2]], axis=-1)
    np.testing.assert_allclose(y_q, expected, atol=1e-7)


def test_apply_rotary_emb_rank_guard():
    bad_x = jnp.ones((2, 8, 32))  # rank 3
    cos = jnp.ones((1, 8, 1, 16))
    sin = jnp.zeros((1, 8, 1, 16))
    with pytest.raises(AssertionError, match="rank-4"):
        apply_rotary_emb(bad_x, cos, sin)


def test_relu_squared_fp32():
    np_x = _np_random((2, 8, 64), seed=42)
    pt_y = F.relu(torch.from_numpy(np_x)).square().numpy()
    jx_y = np.asarray(relu_squared(jnp.asarray(np_x)))
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)
    assert (jx_y >= 0).all()


def test_softcap_fp32_saturation():
    np_x = _np_random((2, 8, 32), seed=42, scale=30.0)
    cap = 15.0
    pt_y = (cap * torch.tanh(torch.from_numpy(np_x) / cap)).numpy()
    jx_y = np.asarray(softcap(jnp.asarray(np_x)))
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-5, rtol=1e-5)
    assert (np.abs(jx_y) <= cap + 1e-5).all()


def test_softcap_custom_cap():
    np_x = _np_random((2, 4), seed=11, scale=10.0)
    cap = 5.0
    pt_y = (cap * torch.tanh(torch.from_numpy(np_x) / cap)).numpy()
    jx_y = np.asarray(softcap(jnp.asarray(np_x), cap=cap))
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-5, rtol=1e-5)


def test_jit_compat_rms_norm():
    np_x = _np_random((2, 8, 32), seed=1)
    pt_y = F.rms_norm(torch.from_numpy(np_x), (np_x.shape[-1],)).numpy()
    jx_y = np.asarray(jax.jit(rms_norm)(jnp.asarray(np_x)))
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)


def test_jit_compat_apply_rotary_emb():
    B, T, H, D = 2, 8, 4, 16
    np_x = _np_random((B, T, H, D), seed=2)
    np_cos = _np_random((1, T, 1, D // 2), seed=3)
    np_sin = _np_random((1, T, 1, D // 2), seed=4)
    pt_y = _pt_apply_rotary_emb(
        torch.from_numpy(np_x), torch.from_numpy(np_cos), torch.from_numpy(np_sin)
    ).numpy()
    jx_y = np.asarray(
        jax.jit(apply_rotary_emb)(
            jnp.asarray(np_x), jnp.asarray(np_cos), jnp.asarray(np_sin)
        )
    )
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)


def test_jit_compat_relu_squared():
    np_x = _np_random((2, 8, 64), seed=5)
    pt_y = F.relu(torch.from_numpy(np_x)).square().numpy()
    jx_y = np.asarray(jax.jit(relu_squared)(jnp.asarray(np_x)))
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)


def test_jit_compat_softcap():
    np_x = _np_random((2, 8, 32), seed=6, scale=30.0)
    cap = 15.0
    pt_y = (cap * torch.tanh(torch.from_numpy(np_x) / cap)).numpy()
    jx_y = np.asarray(jax.jit(softcap)(jnp.asarray(np_x)))
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-5, rtol=1e-5)
