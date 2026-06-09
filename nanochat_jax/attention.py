"""Attention backend helpers for nanochat_jax.

The model in :mod:`nanochat_jax.gpt` should read like upstream nanochat's
Transformer. TPU/Pallas-specific details live here so the backend divergence is
isolated and easier to test.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


_ACTIVE_SPLASH_MESH: "jax.sharding.Mesh | None" = None


def set_splash_mesh(mesh: "jax.sharding.Mesh | None") -> None:
    """Register the active mesh for Splash multi-chip ``shard_map`` wrapping.

    Call this from the training entry point before tracing the train step so
    the mesh value is captured at trace time. Pass ``None`` to revert to the
    single-chip ``vmap`` fallback.
    """
    global _ACTIVE_SPLASH_MESH
    _ACTIVE_SPLASH_MESH = mesh


def get_splash_mesh() -> "jax.sharding.Mesh | None":
    """Read the currently-registered Splash mesh."""
    return _ACTIVE_SPLASH_MESH


def make_splash_kernel(
    T: int,
    n_head: int,
    window_left: int,
    interpret: bool = False,
    *,
    block_q: int | None = None,
    block_kv: int | None = None,
    block_kv_compute: int | None = None,
):
    """Build a Splash MHA kernel for the given ``(T, n_head, window_left)``.

    ``window_left < 0`` or ``>= T`` means full causal mask. Otherwise the kernel
    is built with ``CausalMask & LocalMask`` for sliding-window attention.

    ``block_q`` / ``block_kv`` tune the Splash/Mosaic tile sizes. ``None`` keeps
    Splash defaults. The setting changes TPU performance, not model math.

    Note: do not cache this builder with ``functools.lru_cache``. Inside an
    ``nnx.jit`` trace the cache leaks intermediate JAX tracers and triggers
    ``UnexpectedTracerError``. JAX's own JIT cache handles trace re-use via
    mask object identity.

    ``interpret=True`` runs Pallas in CPU emulation mode, which is structurally
    testable but numerically incorrect. TPU is the numerical source of truth.
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as sak,
        splash_attention_mask as sam,
    )

    mask_shape = (T, T)
    if window_left < 0 or window_left >= T:
        mask = sam.CausalMask(shape=mask_shape)
    else:
        mask = sam.CausalMask(shape=mask_shape) & sam.LocalMask(
            shape=mask_shape,
            window_size=(window_left, None),
            offset=0,
        )
    multi_head_mask = sam.MultiHeadMask(masks=(mask,) * n_head)
    block_sizes = None
    if block_q is not None or block_kv is not None:
        if block_q is None or block_kv is None:
            raise ValueError("splash block_q and block_kv must be set together")
        if block_q <= 0 or block_kv <= 0:
            raise ValueError("splash block sizes must be positive")
        bkc = block_kv_compute if block_kv_compute is not None else min(block_kv, 128)
        if bkc <= 0 or bkc % 128 != 0:
            raise ValueError(
                "splash block_kv_compute must be a positive multiple of 128"
            )
        block_sizes = dataclasses.replace(
            sak.BlockSizes.get_default(),
            block_q=block_q,
            block_kv=block_kv,
            block_kv_compute=bkc,
            block_q_dkv=block_q,
            block_kv_dkv=block_kv,
            block_kv_dkv_compute=bkc,
            block_q_dq=block_q,
            block_kv_dq=block_kv,
        )
    return sak.make_splash_mha_single_device(
        mask=multi_head_mask,
        block_sizes=block_sizes,
        is_mqa=False,
        head_shards=1,
        q_seq_shards=1,
        interpret=interpret,
    )


def call_splash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    window_size: tuple[int, int],
    interpret: bool = False,
    mesh: "jax.sharding.Mesh | None" = None,
    data_axis: str = "data",
    block_q: int | None = None,
    block_kv: int | None = None,
    block_kv_compute: int | None = None,
) -> jax.Array:
    """Splash attention call. Inputs and outputs are BTND.

    The Splash kernel does not automatically apply the ``1/sqrt(head_dim)``
    attention scale, unlike ``jax.nn.dot_product_attention`` whose default
    ``scale=None`` resolves to that. We pre-scale ``q`` here so the two paths
    agree numerically.

    Pallas Mosaic kernels cannot be auto-partitioned by the XLA SPMD compiler.
    When a multi-chip mesh is provided the splash call is wrapped in
    ``jax.shard_map`` along the ``data_axis``. Single-chip meshes, or
    ``mesh=None``, take a plain ``jax.vmap`` path.
    """
    T = q.shape[1]
    n_head = q.shape[2]
    head_dim = q.shape[3]
    window_left = window_size[0]
    splash_kernel = make_splash_kernel(
        T,
        n_head,
        window_left,
        interpret,
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
    )

    scale = jax.lax.rsqrt(jnp.asarray(head_dim, dtype=jnp.float32))
    q_scaled = (q.astype(jnp.float32) * scale).astype(q.dtype)

    splash_batched = jax.vmap(splash_kernel, in_axes=(0, 0, 0))

    def _splash_fn(q_btnd, k_btnd, v_btnd):
        # BTND -> BHTD -> splash -> BHTD -> BTND
        q_bhtd = jnp.transpose(q_btnd, (0, 2, 1, 3))
        k_bhtd = jnp.transpose(k_btnd, (0, 2, 1, 3))
        v_bhtd = jnp.transpose(v_btnd, (0, 2, 1, 3))
        y_bhtd = splash_batched(q_bhtd, k_bhtd, v_bhtd)
        return jnp.transpose(y_bhtd, (0, 2, 1, 3))

    active_mesh = mesh if mesh is not None else _ACTIVE_SPLASH_MESH

    if active_mesh is not None and active_mesh.devices.size > 1:
        from jax.sharding import PartitionSpec as P

        sharded_fn = jax.shard_map(
            _splash_fn,
            mesh=active_mesh,
            in_specs=(P(data_axis), P(data_axis), P(data_axis)),
            out_specs=P(data_axis),
            check_vma=False,
        )
        return sharded_fn(q_scaled, k, v)

    return _splash_fn(q_scaled, k, v)


def dpa_window_size(window_size: tuple[int, int]) -> tuple[int, int] | None:
    """Map our ``(-1, 0)`` full-causal sentinel to JAX DPA's ``None``.

    JAX DPA does not interpret ``-1`` as unlimited; passing ``(-1, 0)`` directly
    produces a different result. Sliding windows ``(N, 0)`` pass through.
    """
    if window_size[0] == -1:
        return None
    return window_size
