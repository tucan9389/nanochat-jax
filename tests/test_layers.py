"""Tests for nanochat_jax.layers.NanochatLinear against PyTorch ``F.linear``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from flax import nnx

from nanochat_jax.layers import NanochatLinear


def _make_linear(in_f, out_f, np_weight, **kwargs):
    """Build a NanochatLinear and inject the PyTorch-style ``(out, in)``
    weight after transposing to JAX ``(in, out)`` convention."""
    seed = kwargs.pop("seed", 0)
    linear = NanochatLinear(in_f, out_f, rngs=nnx.Rngs(seed), **kwargs)
    linear.kernel[...] = jnp.asarray(np_weight.T)
    return linear


def _torch_linear_out(np_x, np_w):
    pt_x = torch.from_numpy(np_x)
    pt_w = torch.from_numpy(np_w)
    return F.linear(pt_x, pt_w).numpy()


def test_fp32_forward_matches_pytorch():
    rng = np.random.RandomState(42)
    in_f, out_f = 32, 64
    np_w = rng.randn(out_f, in_f).astype(np.float32)
    np_x = rng.randn(2, 8, in_f).astype(np.float32)

    linear = _make_linear(in_f, out_f, np_w)
    jx_y = np.asarray(linear(jnp.asarray(np_x)))
    pt_y = _torch_linear_out(np_x, np_w)

    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)


def test_bf16_forward_matches_pytorch_within_tolerance():
    rng = np.random.RandomState(42)
    in_f, out_f = 32, 64
    np_w = rng.randn(out_f, in_f).astype(np.float32)
    np_x = rng.randn(2, 8, in_f).astype(np.float32)

    pt_x = torch.from_numpy(np_x).to(torch.bfloat16)
    pt_w = torch.from_numpy(np_w).to(torch.bfloat16)
    pt_y = F.linear(pt_x, pt_w).to(torch.float32).numpy()

    linear = _make_linear(in_f, out_f, np_w)
    jx_x = jnp.asarray(np_x).astype(jnp.bfloat16)
    jx_y = np.asarray(linear(jx_x).astype(jnp.float32))

    np.testing.assert_allclose(jx_y, pt_y, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "name,in_f,out_f",
    [
        ("c_q", 64, 64),
        ("c_k", 64, 32),
        ("c_v", 64, 32),
        ("c_proj", 64, 64),
        ("ve_gate", 12, 2),
        ("mlp_c_fc", 64, 256),
        ("mlp_c_proj", 256, 64),
        ("lm_head", 64, 128),
        ("smear_gate", 24, 1),
    ],
)
def test_nanochat_callsite_shapes(name, in_f, out_f):
    rng = np.random.RandomState(hash(name) & 0xFFFFFFFF)
    np_w = rng.randn(out_f, in_f).astype(np.float32)
    np_x = rng.randn(2, 8, in_f).astype(np.float32)

    linear = _make_linear(in_f, out_f, np_w)
    jx_y = np.asarray(linear(jnp.asarray(np_x)))
    pt_y = _torch_linear_out(np_x, np_w)

    assert jx_y.shape == (2, 8, out_f), f"{name} shape mismatch"
    np.testing.assert_allclose(
        jx_y, pt_y, atol=1e-6, rtol=1e-5,
        err_msg=f"call site '{name}' (in={in_f}, out={out_f})",
    )


def test_weight_transpose_convention():
    rng = np.random.RandomState(7)
    in_f, out_f = 16, 24
    np_w = rng.randn(out_f, in_f).astype(np.float32)

    linear = _make_linear(in_f, out_f, np_w)
    assert linear.kernel[...].shape == (in_f, out_f)
    np.testing.assert_array_equal(np.asarray(linear.kernel[...]), np_w.T)


def test_dot_general_hook_preserves_output():
    rng = np.random.RandomState(11)
    in_f, out_f = 16, 32
    np_w = rng.randn(out_f, in_f).astype(np.float32)
    np_x = rng.randn(2, 4, in_f).astype(np.float32)

    calls = []

    def identity_dot_general(lhs, rhs, dimension_numbers, precision=None, **kwargs):
        calls.append((lhs.shape, rhs.shape, dimension_numbers))
        return jax.lax.dot_general(
            lhs, rhs, dimension_numbers, precision=precision, **kwargs
        )

    baseline = _make_linear(in_f, out_f, np_w)
    swapped = _make_linear(in_f, out_f, np_w, dot_general=identity_dot_general)

    y_base = np.asarray(baseline(jnp.asarray(np_x)))
    y_swap = np.asarray(swapped(jnp.asarray(np_x)))

    np.testing.assert_array_equal(y_base, y_swap)
    assert len(calls) == 1
    assert calls[0][2] == (((np_x.ndim - 1,), (0,)), ((), ()))


def test_kernel_axes_metadata_attached():
    axes = ("embed", "mlp")
    linear = NanochatLinear(64, 256, kernel_axes=axes, rngs=nnx.Rngs(0))

    assert linear.kernel_axes == axes
    md = linear.kernel.get_metadata()
    assert md.get("out_sharding") == axes or md.get("sharding") == axes, md


def test_kernel_axes_length_validated():
    with pytest.raises(ValueError, match="kernel_axes length"):
        NanochatLinear(64, 256, kernel_axes=("embed",), rngs=nnx.Rngs(0))


def test_kernel_axes_none_default():
    rng = np.random.RandomState(3)
    in_f, out_f = 16, 32
    np_w = rng.randn(out_f, in_f).astype(np.float32)
    np_x = rng.randn(2, 4, in_f).astype(np.float32)

    linear = _make_linear(in_f, out_f, np_w)
    assert linear.kernel_axes is None

    jx_y = np.asarray(linear(jnp.asarray(np_x)))
    pt_y = _torch_linear_out(np_x, np_w)
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)


def test_nnx_jit_compatibility():
    rng = np.random.RandomState(5)
    in_f, out_f = 16, 32
    np_w = rng.randn(out_f, in_f).astype(np.float32)
    np_x = rng.randn(2, 4, in_f).astype(np.float32)

    linear = _make_linear(in_f, out_f, np_w)

    @nnx.jit
    def call(model, x):
        return model(x)

    jx_y = np.asarray(call(linear, jnp.asarray(np_x)))
    pt_y = _torch_linear_out(np_x, np_w)
    np.testing.assert_allclose(jx_y, pt_y, atol=1e-6, rtol=1e-5)
