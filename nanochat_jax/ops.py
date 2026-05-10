"""Pure JAX functional ops: rms_norm, rotary embeddings, ReLU squared, softcap."""

from __future__ import annotations

import jax
import jax.lax
import jax.nn
import jax.numpy as jnp


def rms_norm(x: jax.Array, *, eps: float | None = None) -> jax.Array:
    """Root-Mean-Square Layer Normalization (no learnable scale).

    Mirrors PyTorch ``F.rms_norm(x, (x.size(-1),))`` with ``weight=None``.

    For bf16 inputs PyTorch internally promotes to fp32 for the
    mean-of-squares + rsqrt + multiply, then casts back. Native bf16
    reduction diverges at ~3e-2 from PyTorch. The default eps is
    ``finfo(fp32).eps`` regardless of input dtype.
    """
    x_fp32 = x.astype(jnp.float32)
    if eps is None:
        eps = jnp.finfo(jnp.float32).eps
    rsqrt = jax.lax.rsqrt(
        jnp.mean(jnp.square(x_fp32), axis=-1, keepdims=True) + eps
    )
    return (x_fp32 * rsqrt).astype(x.dtype)


def apply_rotary_emb(
    x: jax.Array, cos: jax.Array, sin: jax.Array
) -> jax.Array:
    """Apply rotary positional embeddings to ``(B, T, n_head, head_dim)``.

    Uses the split-half scheme: the first and second halves of ``head_dim``
    are rotated together via the matrix ``[[cos, sin], [-sin, cos]]``.

    Shapes:
    - ``x``: ``(B, T, n_head, head_dim)``
    - ``cos`` / ``sin``: ``(1, T, 1, head_dim/2)`` (broadcast over B and head).
    """
    assert x.ndim == 4, f"apply_rotary_emb expects rank-4 input, got shape {x.shape}"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)


def relu_squared(x: jax.Array) -> jax.Array:
    """ReLU squared activation: ``relu(x) ** 2`` (not ``relu(x ** 2)``)."""
    return jax.nn.relu(x) ** 2


def softcap(x: jax.Array, *, cap: float = 15.0) -> jax.Array:
    """Smoothly cap values to ``[-cap, cap]`` via ``cap * tanh(x / cap)``."""
    return cap * jnp.tanh(x / cap)
