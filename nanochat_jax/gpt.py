"""GPT model -- JAX/Flax NNX port of Karpathy's nanochat.

Notable features (mirroring upstream nanochat):
- rotary positional embeddings, no learnable position embeddings
- QK norm with q, k pre-scale of 1.2 (sharper attention)
- untied token embedding and lm_head
- ReLU squared activation in the MLP
- per-layer learnable scalars (resid_lambdas, x0_lambdas, smear, backout)
- value embeddings on alternating layers (last layer always included)
- two attention paths: ``jax.nn.dot_product_attention`` (default) or
  Pallas Splash Attention on TPU (opt-in via ``GPTConfig.attn_impl``)
- bf16 compute on fp32 master weights, controlled by ``GPTConfig.compute_dtype``
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx

from nanochat_jax.layers import NanochatLinear
from nanochat_jax.ops import apply_rotary_emb, relu_squared, rms_norm, softcap

Granularity = Literal["high", "block", "minimal"]


# -----------------------------------------------------------------------------
# Splash Attention (TPU Pallas kernel)
# -----------------------------------------------------------------------------
#
# Splash Attention is a TPU-native FlashAttention. It cannot be auto-partitioned
# by the XLA SPMD compiler, so multi-chip use requires wrapping the kernel in
# ``jax.shard_map``. We register the active mesh at module level and the kernel
# call picks it up automatically.

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


def _make_splash_kernel(
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

    Note: do NOT cache this builder with ``functools.lru_cache``. Inside an
    ``nnx.jit`` trace the cache leaks intermediate JAX tracers and triggers
    ``UnexpectedTracerError``. JAX's own JIT cache handles trace re-use via
    mask object identity.

    ``interpret=True`` runs Pallas in CPU emulation mode, which is structurally
    testable but numerically incorrect (~1.4 max diff vs xla). TPU is the only
    source of numerical truth.
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


def _call_splash_attention(
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

    The Splash kernel does NOT automatically apply the ``1/sqrt(head_dim)``
    attention scale, unlike ``jax.nn.dot_product_attention`` whose default
    ``scale=None`` resolves to that. We pre-scale ``q`` here so the two paths
    agree numerically.

    Pallas Mosaic kernels cannot be auto-partitioned by the XLA SPMD compiler
    (``NotImplementedError: Mosaic kernels cannot be automatically partitioned.
    Please wrap the call in a shard_map``). When a multi-chip mesh is provided
    the splash call is wrapped in ``jax.shard_map`` along the ``data_axis``.
    Single-chip meshes (or ``mesh=None``) take a plain ``jax.vmap`` path.
    """
    T = q.shape[1]
    n_head = q.shape[2]
    head_dim = q.shape[3]
    window_left = window_size[0]
    splash_kernel = _make_splash_kernel(
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class GPTConfig:
    """Mirror of nanochat ``GPTConfig`` with extra knobs for JAX precision.

    ``compute_dtype`` controls the activation dtype. ``jnp.float32`` keeps the
    full-precision path; ``jnp.bfloat16`` triggers the cascade (cos/sin/wte
    cast) and lets every op keep input dtype as it propagates. Linear master
    weights remain fp32 inside :class:`NanochatLinear`.

    ``attn_impl="splash"`` opts into the TPU Pallas kernel for training/prefill
    (decode keeps xla). Splash is only valid on TPU; CPU emulation is
    numerically incorrect.
    """

    sequence_len: int = 128
    vocab_size: int = 100
    n_layer: int = 1
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 64
    window_pattern: str = "L"
    compute_dtype: jnp.dtype = jnp.float32
    attn_impl: Literal["xla", "splash"] = "xla"
    lm_head_precision: "jax.lax.PrecisionLike | None" = None
    splash_block_q: int | None = None
    splash_block_kv: int | None = None
    splash_block_kv_compute: int | None = None
    ve_grad_impl: Literal["scatter", "onehot", "segsum", "segsum_fp32"] = "scatter"


def has_ve(layer_idx: int, n_layer: int) -> bool:
    """True if layer ``layer_idx`` carries a value embedding (alternating, last layer always)."""
    return layer_idx % 2 == (n_layer - 1) % 2


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def ve_lookup(table: jax.Array, idx: jax.Array, impl: str) -> jax.Array:
    """Value-embedding lookup with an optional custom backward implementation.

    Forward is identical to ``nnx.Embed``. ``impl="scatter"`` keeps the default
    path. ``impl="onehot"`` computes the table gradient as ``one_hot(idx).T @ g``.
    """
    return jnp.take(table, idx, axis=0)


def _ve_lookup_fwd(table: jax.Array, idx: jax.Array, impl: str):
    return jnp.take(table, idx, axis=0), (table.shape[0], idx)


def _ve_lookup_bwd(impl: str, res, g: jax.Array):
    num_embeddings, idx = res
    flat_idx = idx.reshape(-1)
    flat_g = g.reshape(-1, g.shape[-1])
    if impl == "onehot":
        one_hot = jax.nn.one_hot(flat_idx, num_embeddings, dtype=jnp.float32)
        grad_table = jax.lax.dot_general(
            one_hot,
            flat_g.astype(jnp.float32),
            (((0,), (0,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
    elif impl == "segsum_fp32":
        grad_table = jax.ops.segment_sum(
            flat_g.astype(jnp.float32),
            flat_idx,
            num_segments=num_embeddings,
        )
    elif impl == "segsum":
        grad_table = jax.ops.segment_sum(
            flat_g,
            flat_idx,
            num_segments=num_embeddings,
        )
    else:
        raise ValueError(f"unknown ve_grad_impl={impl!r}")
    return (grad_table.astype(g.dtype), None)


ve_lookup.defvjp(_ve_lookup_fwd, _ve_lookup_bwd)


def _ve(embed: "nnx.Embed", idx: jax.Array, impl: str) -> jax.Array:
    if impl == "scatter":
        return embed(idx)
    return ve_lookup(embed.embedding.value, idx, impl)


def _compute_window_sizes(cfg: "GPTConfig") -> list[tuple[int, int]]:
    """Build per-layer ``(left, right)`` window sizes from the pattern string.

    Pattern characters: ``L`` -> long (full context), ``S`` -> short (rounded
    up to the nearest 128-multiple of ``ceil(long/4)``). The pattern tiles
    across layers and the final layer is forced to long.
    """
    pattern = cfg.window_pattern.upper()
    assert all(c in "SL" for c in pattern), (
        f"Invalid window_pattern: {pattern!r}. Use only S and L."
    )
    long_window = cfg.sequence_len
    short_window = -(-long_window // 4 // 128) * 128
    char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
    window_sizes = [
        char_to_window[pattern[i % len(pattern)]] for i in range(cfg.n_layer)
    ]
    window_sizes[-1] = (long_window, 0)
    return window_sizes


def _dpa_window_size(
    window_size: tuple[int, int],
) -> tuple[int, int] | None:
    """Map our ``(-1, 0)`` full-causal sentinel to the value
    ``jax.nn.dot_product_attention`` expects (``None``).

    JAX's dpa does NOT interpret ``-1`` as unlimited; passing ``(-1, 0)``
    directly produces a wildly different result. Sliding windows ``(N, 0)``
    pass through unchanged.
    """
    if window_size[0] == -1:
        return None
    return window_size


def _precompute_rotary_embeddings(
    seq_len: int, head_dim: int, *, base: float = 100000.0
) -> tuple[jax.Array, jax.Array]:
    """Return ``(cos, sin)`` of shape ``(1, seq_len, 1, head_dim/2)``."""
    channel = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return cos, sin


def cross_entropy_with_ignore(
    logits: jax.Array,
    targets: jax.Array,
    *,
    ignore_index: int = -1,
    reduction: str = "mean",
) -> jax.Array:
    """JAX equivalent of ``F.cross_entropy(..., ignore_index, reduction)``.

    ``reduction="mean"`` returns the scalar loss (training path). ``"none"``
    returns a per-token loss with ignored positions set to zero (eval path).
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    mask = (targets != ignore_index).astype(nll.dtype)
    if reduction == "none":
        return nll * mask
    if reduction == "mean":
        n_valid = jnp.maximum(mask.sum(), 1.0)
        return (nll * mask).sum() / n_valid
    raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------


class MLP(nnx.Module):
    """Two-Linear MLP with ReLU squared activation and 4x expansion."""

    def __init__(self, cfg: GPTConfig, *, rngs: nnx.Rngs):
        self.c_fc = NanochatLinear(cfg.n_embd, 4 * cfg.n_embd, rngs=rngs)
        self.c_proj = NanochatLinear(4 * cfg.n_embd, cfg.n_embd, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.c_fc(x)
        x = relu_squared(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nnx.Module):
    """Causal self-attention with QK norm, RoPE, and a 1.2 q/k pre-scale.

    GQA is supported through ``n_kv_head <= n_head AND n_head % n_kv_head == 0``.
    ``jax.nn.dot_product_attention`` natively broadcasts ``k``/``v`` when their
    head count is smaller than the query's. Sliding window is forwarded via
    ``local_window_size``. The Splash kernel is opt-in via
    ``GPTConfig.attn_impl="splash"`` and currently requires MHA.
    """

    def __init__(self, cfg: GPTConfig, layer_idx: int, *, rngs: nnx.Rngs):
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_kv_head <= cfg.n_head and cfg.n_head % cfg.n_kv_head == 0, (
            "GQA: n_kv_head must divide n_head (n_kv_head <= n_head). "
            f"Got n_head={cfg.n_head}, n_kv_head={cfg.n_kv_head}."
        )
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.ve_gate_channels = 12
        self.attn_impl: Literal["xla", "splash"] = cfg.attn_impl
        self.splash_block_q = cfg.splash_block_q
        self.splash_block_kv = cfg.splash_block_kv
        self.splash_block_kv_compute = cfg.splash_block_kv_compute

        self.c_q = NanochatLinear(cfg.n_embd, cfg.n_head * self.head_dim, rngs=rngs)
        self.c_k = NanochatLinear(cfg.n_embd, cfg.n_kv_head * self.head_dim, rngs=rngs)
        self.c_v = NanochatLinear(cfg.n_embd, cfg.n_kv_head * self.head_dim, rngs=rngs)
        self.c_proj = NanochatLinear(cfg.n_embd, cfg.n_embd, rngs=rngs)
        self.ve_gate: NanochatLinear | None = (
            NanochatLinear(self.ve_gate_channels, cfg.n_kv_head, rngs=rngs)
            if has_ve(layer_idx, cfg.n_layer)
            else None
        )

    def __call__(
        self,
        x: jax.Array,
        ve: jax.Array | None,
        cos_sin: tuple[jax.Array, jax.Array],
        window_size: tuple[int, int] = (-1, 0),
        kv_cache=None,
    ) -> jax.Array:
        B, T, _ = x.shape
        assert window_size[1] == 0, (
            "right window must be 0 (causal); future-attending masks are not supported"
        )

        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            assert self.ve_gate is not None
            ve_view = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 3.0 * jax.nn.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate[..., None] * ve_view

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # The 1.2 multipliers compose with dpa's default 1/sqrt(D) scale to
        # an effective 1.44/sqrt(D) -- a sharper attention scale that the
        # upstream nanochat introduced empirically.
        q = rms_norm(q)
        k = rms_norm(k)
        q = q * 1.2
        k = k * 1.2

        if kv_cache is None:
            if self.attn_impl == "splash":
                assert self.n_kv_head == self.n_head, (
                    "attn_impl='splash' currently requires MHA; got "
                    f"n_head={self.n_head}, n_kv_head={self.n_kv_head}."
                )
                y = _call_splash_attention(
                    q,
                    k,
                    v,
                    window_size=window_size,
                    block_q=self.splash_block_q,
                    block_kv=self.splash_block_kv,
                    block_kv_compute=self.splash_block_kv_compute,
                )
            else:
                y = jax.nn.dot_product_attention(
                    q, k, v,
                    is_causal=True,
                    local_window_size=_dpa_window_size(window_size),
                    implementation="xla",
                )
        else:
            # Decode path: write K/V at the current cache position, read full
            # cache, and build an explicit mask. ``jax.nn.dpa(is_causal=True)``
            # with T_q != T_kv aligns the query at q-position 0 instead of the
            # last kv position, which is the wrong semantic for incremental
            # decoding -- hence the manual mask below.
            kv_cache.write_kv(self.layer_idx, k, v)
            k_cache_layer, v_cache_layer = kv_cache.get_layer_cache(self.layer_idx)

            T_kv = k_cache_layer.shape[1]
            cache_pos = kv_cache.cache_seqlen[...]
            q_idx = jnp.arange(T, dtype=jnp.int32)
            k_idx = jnp.arange(T_kv, dtype=jnp.int32)
            q_global = cache_pos + q_idx
            total_valid = cache_pos + jnp.asarray(T, dtype=jnp.int32)
            causal = k_idx[None, :] <= q_global[:, None]
            bounded = k_idx[None, :] < total_valid
            mask_2d = causal & bounded
            left_window = window_size[0]
            if left_window > 0:
                window_mask = k_idx[None, :] >= (q_global[:, None] - left_window + 1)
                mask_2d = mask_2d & window_mask
            mask = jnp.broadcast_to(mask_2d[None, None, :, :], (B, 1, T, T_kv))

            y = jax.nn.dot_product_attention(
                q, k_cache_layer, v_cache_layer,
                mask=mask,
                implementation="xla",
            )

            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        y = y.reshape(B, T, self.n_embd)
        y = self.c_proj(y)
        return y


class Block(nnx.Module):
    """Pre-norm transformer block: ``x + attn(rms_norm(x)) ; x + mlp(rms_norm(x))``."""

    def __init__(self, cfg: GPTConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.attn = CausalSelfAttention(cfg, layer_idx, rngs=rngs)
        self.mlp = MLP(cfg, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        ve: jax.Array | None,
        cos_sin: tuple[jax.Array, jax.Array],
        window_size: tuple[int, int] = (-1, 0),
        kv_cache=None,
    ) -> jax.Array:
        x = x + self.attn(rms_norm(x), ve, cos_sin, window_size, kv_cache=kv_cache)
        x = x + self.mlp(rms_norm(x))
        return x


class Transformer(nnx.Module):
    """Sub-module grouping ``wte`` and the block list.

    The grouping keeps NNX leaf paths aligned with PyTorch ``state_dict``
    keys (``transformer.wte.weight``, ``transformer.h.{i}.attn.c_q.weight``,
    ...), which lets the weight converter map keys 1:1 with no extra rewrite.
    """

    def __init__(self, cfg: GPTConfig, padded_vocab_size: int, *, rngs: nnx.Rngs):
        self.wte = nnx.Embed(padded_vocab_size, cfg.n_embd, rngs=rngs)
        self.h = nnx.List([Block(cfg, i, rngs=rngs) for i in range(cfg.n_layer)])


class GPT(nnx.Module):
    """nanochat GPT in JAX/Flax NNX.

    The production ``__call__`` returns logits, or a scalar loss if ``targets``
    is provided. ``forward_with_intermediates`` is a testing helper that
    captures named intermediate tensors at one of three granularities; it is
    paired with the PyTorch-side reference in ``scripts/generate_golden.py``.
    """

    def __init__(
        self,
        cfg: GPTConfig,
        *,
        pad_vocab_size_to: int = 64,
        rngs: nnx.Rngs,
    ):
        self.config = cfg
        padded_vocab_size = (
            (cfg.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        self.padded_vocab_size = padded_vocab_size

        head_dim = cfg.n_embd // cfg.n_head
        self.head_dim = head_dim
        kv_dim = cfg.n_kv_head * head_dim

        self.transformer = Transformer(cfg, padded_vocab_size, rngs=rngs)
        self.lm_head = NanochatLinear(
            cfg.n_embd,
            padded_vocab_size,
            precision=cfg.lm_head_precision,
            rngs=rngs,
        )

        self.resid_lambdas = nnx.Param(jnp.ones((cfg.n_layer,), dtype=jnp.float32))
        self.x0_lambdas = nnx.Param(jnp.zeros((cfg.n_layer,), dtype=jnp.float32))
        self.smear_gate = NanochatLinear(24, 1, rngs=rngs)
        self.smear_lambda = nnx.Param(jnp.zeros((1,), dtype=jnp.float32))
        self.backout_lambda = nnx.Param(0.2 * jnp.ones((1,), dtype=jnp.float32))

        self.value_embeds = nnx.Dict(
            {
                str(i): nnx.Embed(padded_vocab_size, kv_dim, rngs=rngs)
                for i in range(cfg.n_layer)
                if has_ve(i, cfg.n_layer)
            }
        )

        # Over-allocate the rotary cache by 10x (matches upstream nanochat).
        self.rotary_seq_len = cfg.sequence_len * 10
        cos, sin = _precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos = nnx.data(cos)
        self.sin = nnx.data(sin)

        # Plain Python list -- not jax.Array, no NNX wrapping needed.
        self.window_sizes: list[tuple[int, int]] = _compute_window_sizes(cfg)

    def estimate_flops_per_token(self) -> int:
        """Estimated FLOPs per token (forward + backward).

        Mirrors ``karpathy/nanochat::GPT.estimate_flops`` — excludes
        embedding-class params (wte, value_embeds) and per-layer scalar
        lambdas, since those don't contribute matmul FLOPs. Each remaining
        matmul weight contributes 6 (2 fwd + 4 bwd). Attention adds
        ``12 * h * q * effective_seq_len`` per layer (sliding-window aware).

        Used by ``scripts/base_train.py`` to log MFU each step.
        """
        cfg = self.config
        state = nnx.state(self, nnx.Param)
        total = sum(int(p.size) for p in jax.tree.leaves(state))

        excluded = int(self.transformer.wte.embedding.value.size)
        for ve in self.value_embeds.values():
            excluded += int(ve.embedding.value.size)
        excluded += int(self.resid_lambdas.value.size)
        excluded += int(self.x0_lambdas.value.size)
        excluded += int(self.smear_gate.kernel.value.size)
        excluded += int(self.smear_lambda.value.size)
        excluded += int(self.backout_lambda.value.size)
        matmul_params = total - excluded

        h = cfg.n_head
        q = cfg.n_embd // cfg.n_head
        t = cfg.sequence_len
        attn_flops = 0
        for window_left, _window_right in self.window_sizes:
            effective_seq = t if window_left < 0 else min(window_left, t)
            attn_flops += 12 * h * q * effective_seq

        return 6 * matmul_params + attn_flops

    def init_weights(
        self,
        *,
        seed: int = 42,
        cast_embeddings_to_compute_dtype: bool = False,
    ) -> None:
        """Re-initialize all parameters using nanochat's custom scheme.

        Call after ``__init__`` to overwrite the NNX defaults.

        With ``cast_embeddings_to_compute_dtype=True`` and ``cfg.compute_dtype
        == jnp.bfloat16`` the wte and value_embeds Params are physically cast
        to bf16 (matching upstream's storage, plus the AdamW state then
        inherits bf16 via ``zeros_like``). The default is fp32 storage with an
        on-the-fly cast at forward.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        cfg = self.config
        n_embd = cfg.n_embd
        n_layer = cfg.n_layer

        wte_shape = self.transformer.wte.embedding.value.shape
        self.transformer.wte.embedding = nnx.Param(
            jnp.array(rng.normal(0.0, 0.8, wte_shape), dtype=jnp.float32)
        )

        lm_shape = self.lm_head.kernel.value.shape
        self.lm_head.kernel = nnx.Param(
            jnp.array(rng.normal(0.0, 0.001, lm_shape), dtype=jnp.float32)
        )

        # s = sqrt(3) / sqrt(n_embd): Uniform(-s, s) achieves the same std as Normal(0, 1/sqrt(n_embd))
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            for linear in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
                shape = linear.kernel.value.shape
                linear.kernel = nnx.Param(
                    jnp.array(rng.uniform(-s, s, shape), dtype=jnp.float32)
                )
            block.attn.c_proj.kernel = nnx.Param(
                jnp.zeros_like(block.attn.c_proj.kernel.value)
            )
            fc_shape = block.mlp.c_fc.kernel.value.shape
            block.mlp.c_fc.kernel = nnx.Param(
                jnp.array(rng.uniform(-s * 0.4, s * 0.4, fc_shape), dtype=jnp.float32)
            )
            block.mlp.c_proj.kernel = nnx.Param(
                jnp.zeros_like(block.mlp.c_proj.kernel.value)
            )

        resid = jnp.array(
            [1.15 - (0.10 * i / max(n_layer - 1, 1)) for i in range(n_layer)],
            dtype=jnp.float32,
        )
        self.resid_lambdas = nnx.Param(resid)

        x0 = jnp.array(
            [0.20 - (0.15 * i / max(n_layer - 1, 1)) for i in range(n_layer)],
            dtype=jnp.float32,
        )
        self.x0_lambdas = nnx.Param(x0)

        for ve in self.value_embeds.values():
            ve_shape = ve.embedding.value.shape
            ve.embedding = nnx.Param(
                jnp.array(rng.uniform(-s, s, ve_shape), dtype=jnp.float32)
            )

        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                gate_shape = block.attn.ve_gate.kernel.value.shape
                block.attn.ve_gate.kernel = nnx.Param(
                    jnp.array(rng.uniform(0.0, 0.02, gate_shape), dtype=jnp.float32)
                )

        if cast_embeddings_to_compute_dtype and cfg.compute_dtype != jnp.float32:
            wte_val = self.transformer.wte.embedding.value.astype(cfg.compute_dtype)
            self.transformer.wte.embedding = nnx.Param(wte_val)
            for ve in self.value_embeds.values():
                ve_val = ve.embedding.value.astype(cfg.compute_dtype)
                ve.embedding = nnx.Param(ve_val)

    def num_scaling_params(self) -> dict:
        """Counts of parameters split by category.

        ``num_scaling_params = transformer_matrices + lm_head`` follows the
        Kaplan et al. scaling-laws convention (embeddings excluded). The
        training script uses this together with ``--target-param-data-ratio``
        to auto-compute the iteration count.
        """
        from nanochat_jax.grad_utils import nnx_state_to_flat_dict
        import numpy as np

        flat = nnx_state_to_flat_dict(nnx.state(self, nnx.Param))
        counts = {"wte": 0, "value_embeds": 0, "lm_head": 0,
                  "transformer_matrices": 0, "scalars": 0, "total": 0}
        for k, v in flat.items():
            n = int(np.prod(v.shape))
            if k.startswith("transformer.wte"):
                counts["wte"] += n
            elif k.startswith("value_embeds"):
                counts["value_embeds"] += n
            elif k.startswith("lm_head"):
                counts["lm_head"] += n
            elif k.startswith("transformer.h"):
                counts["transformer_matrices"] += n
            else:
                counts["scalars"] += n
            counts["total"] += n
        return counts

    def __call__(
        self,
        idx: jax.Array,
        *,
        targets: jax.Array | None = None,
        kv_cache=None,
    ) -> jax.Array:
        """Forward pass returning logits, or a scalar loss if ``targets`` is set.

        ``kv_cache=None`` runs the standard training/prefill path (T > 1).
        ``kv_cache!=None`` runs the decode path: T can be 1 (decode) or > 1
        (prefill). The cos/sin slice is offset by the cache position.
        """
        cfg = self.config
        B, T = idx.shape

        if kv_cache is None:
            assert T > 1, "training requires T > 1"
        else:
            assert T >= 1, "decode/prefill requires T >= 1"

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        assert T0 + T <= self.cos.shape[1], (
            f"position {T0 + T} exceeds rotary cache {self.cos.shape[1]}"
        )

        cos_sin = (
            self.cos[:, T0:T0 + T].astype(cfg.compute_dtype),
            self.sin[:, T0:T0 + T].astype(cfg.compute_dtype),
        )

        x = self.transformer.wte(idx).astype(cfg.compute_dtype)
        x = rms_norm(x)

        if kv_cache is None:
            gate = self.smear_lambda[...].astype(x.dtype) * jax.nn.sigmoid(
                self.smear_gate(x[:, 1:, :24])
            )
            x = jnp.concatenate([x[:, :1], x[:, 1:] + gate * x[:, :-1]], axis=1)
        else:
            x_pre_smear_present = bool(kv_cache.has_prev[...])
            x_pre_smear = (
                kv_cache.prev_embedding[...] if x_pre_smear_present else None
            )
            kv_cache.prev_embedding[...] = x[:, -1:, :]
            kv_cache.has_prev[...] = jnp.array(True)

            if T > 1:
                gate = self.smear_lambda[...].astype(x.dtype) * jax.nn.sigmoid(
                    self.smear_gate(x[:, 1:, :24])
                )
                x = jnp.concatenate(
                    [x[:, :1], x[:, 1:] + gate * x[:, :-1]], axis=1
                )
            elif x_pre_smear is not None:
                gate = self.smear_lambda[...].astype(x.dtype) * jax.nn.sigmoid(
                    self.smear_gate(x[:, :, :24])
                )
                x = x + gate * x_pre_smear
            # else: T=1 first step after empty cache (no prev) -- skip smear.

        x0 = x

        # Cast scalar params to x.dtype explicitly: PyTorch promotes a 0-D
        # bf16 scalar against a bf16 tensor to bf16, but JAX promotes to fp32,
        # which would silently break the bf16 cascade.
        n_layer = cfg.n_layer
        backout_layer = n_layer // 2
        x_backout: jax.Array | None = None
        for i, block in enumerate(self.transformer.h):
            x_block_in = (
                self.resid_lambdas[i].astype(x.dtype) * x
                + self.x0_lambdas[i].astype(x.dtype) * x0
            )
            ve = (
                _ve(self.value_embeds[str(i)], idx, cfg.ve_grad_impl).astype(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            x = block(x_block_in, ve, cos_sin, self.window_sizes[i], kv_cache=kv_cache)
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda[...].astype(x.dtype) * x_backout

        x = rms_norm(x)
        logits = self.lm_head(x)
        logits = logits[..., : cfg.vocab_size]
        logits = logits.astype(jnp.float32)
        logits = softcap(logits, cap=15.0)

        if targets is not None:
            return cross_entropy_with_ignore(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits

    def forward_with_intermediates(
        self,
        idx: jax.Array,
        *,
        targets: jax.Array | None = None,
        granularity: Granularity = "high",
    ) -> dict[str, jax.Array]:
        """Capture named intermediate tensors during a training-style forward.

        Granularities:
        - ``minimal``: ``x_after_wte``, ``logits``, ``loss``
        - ``block``: minimal + ``x_after_smear``, ``x0``, ``block_{i}_out``,
          ``x_after_backout``, ``x_final_norm``
        - ``high``: block + ``x_after_norm0``, ``block_{i}_input``,
          ``logits_pre_softcap``

        Paired with the PyTorch-side reference in ``scripts/generate_golden.py``.
        """
        cfg = self.config
        ints: dict[str, jax.Array] = {}
        B, T = idx.shape
        assert T > 1, "forward_with_intermediates expects T > 1 (training path)"

        cos_sin = (
            self.cos[:, :T].astype(cfg.compute_dtype),
            self.sin[:, :T].astype(cfg.compute_dtype),
        )

        x = self.transformer.wte(idx).astype(cfg.compute_dtype)
        ints["x_after_wte"] = x

        x = rms_norm(x)
        if granularity == "high":
            ints["x_after_norm0"] = x

        gate = self.smear_lambda[...].astype(x.dtype) * jax.nn.sigmoid(
            self.smear_gate(x[:, 1:, :24])
        )
        x = jnp.concatenate([x[:, :1], x[:, 1:] + gate * x[:, :-1]], axis=1)
        if granularity in ("high", "block"):
            ints["x_after_smear"] = x

        x0 = x
        if granularity in ("high", "block"):
            ints["x0"] = x0

        n_layer = cfg.n_layer
        backout_layer = n_layer // 2
        x_backout: jax.Array | None = None
        for i, block in enumerate(self.transformer.h):
            x_block_in = (
                self.resid_lambdas[i].astype(x.dtype) * x
                + self.x0_lambdas[i].astype(x.dtype) * x0
            )
            if granularity == "high":
                ints[f"block_{i}_input"] = x_block_in
            ve = (
                _ve(self.value_embeds[str(i)], idx, cfg.ve_grad_impl).astype(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            x = block(x_block_in, ve, cos_sin, self.window_sizes[i])
            if granularity in ("high", "block"):
                ints[f"block_{i}_out"] = x
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda[...].astype(x.dtype) * x_backout
        if granularity in ("high", "block"):
            ints["x_after_backout"] = x

        x = rms_norm(x)
        if granularity in ("high", "block"):
            ints["x_final_norm"] = x

        logits = self.lm_head(x)
        logits = logits[..., : cfg.vocab_size]
        logits = logits.astype(jnp.float32)
        if granularity == "high":
            ints["logits_pre_softcap"] = logits
        logits = softcap(logits, cap=15.0)
        ints["logits"] = logits

        if targets is not None:
            ints["loss"] = cross_entropy_with_ignore(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-1,
            )

        return ints
