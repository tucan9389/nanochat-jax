"""Config and sizing helpers for ``scripts.base_train``.

These helpers keep deterministic nanochat parity math outside the long training
entry point while leaving the script itself as the readable top-level recipe.
"""

from __future__ import annotations

import jax.numpy as jnp

from nanochat_jax.gpt import GPTConfig


B_REF_TOKENS = 2**19
# muP D_ref: the d12 reference scaling-param count (transformer_matrices + lm_head of
# ``make_config(depth=12)``, n_embd=768) used in the T_epoch weight-decay scaling. Kept
# as a constant rather than computed live because JAX has no free meta device, so
# instantiating a d12 model on every run would waste time/HBM (risky next to the d24
# model). A regression test in the maintainer's local
# verification net asserts this equals a live ``make_config(depth=12)`` count, so
# it cannot silently drift.
D12_SCALING_PARAMS = 110_100_912


def make_config(
    *,
    depth: int,
    aspect_ratio: int = 64,
    head_dim: int = 128,
    sequence_len: int = 2048,
    vocab_size: int = 32768,
    window_pattern: str = "SSSL",
    bf16: bool = True,
    attn_impl: str = "xla",
    lm_head_precision: str | None = None,
    splash_block_q: int | None = None,
    splash_block_kv: int | None = None,
    splash_block_kv_compute: int | None = None,
    ve_grad_impl: str = "scatter",
) -> GPTConfig:
    """Mirror upstream ``build_model_meta`` shape math.

    ``n_embd = ceil(depth * aspect_ratio / head_dim) * head_dim`` and MHA is
    the default, so ``n_kv_head = n_head``.
    """
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=n_embd,
        window_pattern=window_pattern,
        compute_dtype=jnp.bfloat16 if bf16 else jnp.float32,
        attn_impl=attn_impl,
        lm_head_precision=lm_head_precision,
        splash_block_q=splash_block_q,
        splash_block_kv=splash_block_kv,
        splash_block_kv_compute=splash_block_kv_compute,
        ve_grad_impl=ve_grad_impl,
    )


def compute_total_batch_size_tokens(
    *,
    total_batch_size: int,
    device_batch_size: int,
    seq_len: int,
    device_count: int,
    grad_accum_steps: int,
) -> int:
    """Resolve global token batch size from explicit or device-local settings."""
    if total_batch_size > 0:
        return int(total_batch_size)
    return (
        int(device_batch_size)
        * int(seq_len)
        * max(int(device_count), 1)
        * max(int(grad_accum_steps), 1)
    )


def compute_batch_lr_scale(
    total_batch_size_tokens: int,
    *,
    b_ref_tokens: int = B_REF_TOKENS,
) -> float:
    """Karpathy-style LR scale: ``sqrt(B / B_ref)``."""
    return (int(total_batch_size_tokens) / int(b_ref_tokens)) ** 0.5


def compute_weight_decay_scaled(
    *,
    weight_decay: float,
    total_batch_size_tokens: int,
    d_x_scaling_params: int,
    d12_scaling_params: int = D12_SCALING_PARAMS,
    b_ref_tokens: int = B_REF_TOKENS,
) -> float:
    """T_epoch-style WD scaling: ``wd * sqrt(B/B_ref) * (D_ref / D_target)``.

    ``d12_scaling_params`` is the muP D_ref (defaults to the d12 reference
    ``D12_SCALING_PARAMS``); ``d_x_scaling_params`` is the current model's D_target.
    """
    batch_lr_scale = compute_batch_lr_scale(
        total_batch_size_tokens,
        b_ref_tokens=b_ref_tokens,
    )
    return float(weight_decay) * batch_lr_scale * (
        int(d12_scaling_params) / int(d_x_scaling_params)
    )


def resolve_num_iterations(
    *,
    num_iterations: int,
    target_param_data_ratio: float,
    scaling_params: int,
    total_batch_size_tokens: int,
    default_num_iterations: int = 100,
) -> tuple[int, str]:
    """Resolve the optimizer iteration horizon and its human-readable source."""
    if int(num_iterations) > 0:
        return int(num_iterations), "explicit --num-iterations"
    if target_param_data_ratio > 0:
        target_tokens = int(float(target_param_data_ratio) * int(scaling_params))
        resolved = target_tokens // int(total_batch_size_tokens)
        source = (
            f"--target-param-data-ratio={target_param_data_ratio} "
            f"(target_tokens={target_tokens:,}, scaling_params={int(scaling_params):,})"
        )
        return resolved, source
    return int(default_num_iterations), "default 100"
