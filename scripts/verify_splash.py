"""Splash Attention TPU verification: build, forward, gradient, timing.

Run on a TPU VM after activating the venv::

    python -m scripts.verify_splash --batch-size=2 --seq-len=512

CPU interpret mode is structurally testable but numerically incorrect (~1.4
max diff vs xla); TPU is the only source of numerical truth.

Verification matrix:
1. Splash kernel TPU build OK
2. forward shape (B,T,N,D) -> (B,T,N,D) preserved
3. fp32 xla vs Splash atol
4. bf16 xla vs Splash atol
5. 1-step gradient atol (xla vs Splash)
6. memory_stats per-call (xla baseline vs Splash scratch reduction)
7. sec/op timing (xla vs Splash latency)
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as sak,
    splash_attention_mask as sam,
)

from nanochat_jax.gpt import _call_splash_attention


def manual_attn_btnd(q, k, v, *, causal=True, window_left=-1):
    """Reference manual attention (BTND layout). Used as a third-party check."""
    scale = 1.0 / np.sqrt(q.shape[-1])
    logits = jnp.einsum("btnd,bsnd->bnts", q, k) * scale
    Tq = q.shape[1]
    if causal:
        causal_mask = jnp.tril(jnp.ones((Tq, Tq), dtype=bool))
        logits = jnp.where(causal_mask[None, None, :, :], logits, -1e30)
    if window_left >= 0:
        q_idx = jnp.arange(Tq)[None, None, :, None]
        k_idx = jnp.arange(Tq)[None, None, None, :]
        window_mask = k_idx >= (q_idx - window_left)
        logits = jnp.where(window_mask, logits, -1e30)
    attn = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bnts,bsnd->btnd", attn, v)


def build_splash_kernel(T: int, n_head: int, *, window_left: int = -1, interpret: bool = False):
    """Build a Splash kernel partial. Caller wraps with vmap."""
    mask_shape = (T, T)
    if window_left < 0 or window_left >= T:
        mask = sam.CausalMask(shape=mask_shape)
    else:
        mask = sam.CausalMask(shape=mask_shape) & sam.LocalMask(
            shape=mask_shape, window_size=(window_left, None), offset=0
        )
    multi_head_mask = sam.MultiHeadMask(masks=(mask,) * n_head)
    return sak.make_splash_mha_single_device(
        mask=multi_head_mask,
        is_mqa=False,
        head_shards=1,
        q_seq_shards=1,
        interpret=interpret,
    )


def call_splash_batched(splash_kernel, q_btnd, k_btnd, v_btnd, *, window_size=(-1, 0)):
    """Forward via the production helper so the verify path is bit-identical
    to what the model actually runs."""
    del splash_kernel
    return _call_splash_attention(q_btnd, k_btnd, v_btnd, window_size=window_size)


def stats(label: str) -> dict:
    """Print memory_stats with label. Returns dict for caller to compare."""
    try:
        s = jax.devices()[0].memory_stats()
    except Exception as e:
        print(f"[{label}] memory_stats unavailable: {e}")
        return {}
    if s is None:
        print(f"[{label}] memory_stats=None (CPU; TPU/GPU only)")
        return {}
    in_use_gb = s.get("bytes_in_use", 0) / 1e9
    peak_gb = s.get("peak_bytes_in_use", 0) / 1e9
    free_gb = s.get("largest_free_block_bytes", 0) / 1e9
    print(f"[{label}] in_use={in_use_gb:.3f}GB peak={peak_gb:.3f}GB free={free_gb:.3f}GB")
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Splash Attention TPU verify")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-interpret", action="store_true",
                        help="CPU interpret mode (structural sanity only; numerically wrong vs xla)")
    parser.add_argument("--window-left", type=int, default=-1,
                        help="Local sliding window left (negative = full causal)")
    args = parser.parse_args()

    print("=== Splash Attention TPU Verify ===")
    print(f"jax.devices() = {jax.devices()}")
    print(f"jax.process_count() = {jax.process_count()}, process_index = {jax.process_index()}")

    B, T, H, D = args.batch_size, args.seq_len, args.n_head, args.head_dim
    print(f"B={B} T={T} H={H} D={D}, window_left={args.window_left}")

    stats("00_fresh")

    # Inputs
    key = jax.random.PRNGKey(args.seed)
    k1, k2, k3 = jax.random.split(key, 3)
    q_fp32 = jax.random.normal(k1, (B, T, H, D), dtype=jnp.float32) * 0.5
    k_fp32 = jax.random.normal(k2, (B, T, H, D), dtype=jnp.float32) * 0.5
    v_fp32 = jax.random.normal(k3, (B, T, H, D), dtype=jnp.float32) * 0.5

    # xla baseline (reference)
    if args.window_left < 0:
        y_xla_fp32 = jax.nn.dot_product_attention(q_fp32, k_fp32, v_fp32, is_causal=True, implementation="xla")
    else:
        y_xla_fp32 = jax.nn.dot_product_attention(
            q_fp32, k_fp32, v_fp32, is_causal=True,
            local_window_size=(args.window_left, 0), implementation="xla",
        )
    y_xla_fp32 = jax.block_until_ready(y_xla_fp32)
    stats("10_xla_fp32")

    # Manual softmax sanity vs xla
    y_manual = manual_attn_btnd(q_fp32, k_fp32, v_fp32, causal=True, window_left=args.window_left)
    diff_xla_manual = float(jnp.abs(y_xla_fp32 - y_manual).max())
    print(f"[CHECK] xla vs manual softmax max diff = {diff_xla_manual:.2e} (expected near bit-exact)")

    # Splash kernel build + call
    print("\n=== Splash kernel build + call (fp32) ===")
    splash_kernel = build_splash_kernel(T, H, window_left=args.window_left, interpret=args.cpu_interpret)
    print(f"Splash kernel built: {type(splash_kernel).__name__}")
    stats("20_splash_built")

    t0 = time.time()
    y_splash_fp32 = call_splash_batched(splash_kernel, q_fp32, k_fp32, v_fp32)
    y_splash_fp32 = jax.block_until_ready(y_splash_fp32)
    print(f"Splash fp32 first call: {time.time() - t0:.3f}s, shape={y_splash_fp32.shape}, dtype={y_splash_fp32.dtype}")
    stats("30_splash_fp32_done")

    diff_fp32 = float(jnp.abs(y_xla_fp32 - y_splash_fp32).max())
    diff_fp32_mean = float(jnp.abs(y_xla_fp32 - y_splash_fp32).mean())
    print(f"[CHECK fp32] xla vs Splash max diff = {diff_fp32:.2e} (expected ULP-ish on TPU)")
    print(f"[CHECK fp32] mean diff = {diff_fp32_mean:.2e}")

    # bf16
    print("\n=== bf16 ===")
    q_bf16 = q_fp32.astype(jnp.bfloat16)
    k_bf16 = k_fp32.astype(jnp.bfloat16)
    v_bf16 = v_fp32.astype(jnp.bfloat16)
    if args.window_left < 0:
        y_xla_bf16 = jax.nn.dot_product_attention(q_bf16, k_bf16, v_bf16, is_causal=True, implementation="xla")
    else:
        y_xla_bf16 = jax.nn.dot_product_attention(
            q_bf16, k_bf16, v_bf16, is_causal=True,
            local_window_size=(args.window_left, 0), implementation="xla",
        )
    y_xla_bf16 = jax.block_until_ready(y_xla_bf16)
    stats("40_xla_bf16")

    t0 = time.time()
    y_splash_bf16 = call_splash_batched(splash_kernel, q_bf16, k_bf16, v_bf16)
    y_splash_bf16 = jax.block_until_ready(y_splash_bf16)
    print(f"Splash bf16 first call: {time.time() - t0:.3f}s")
    stats("50_splash_bf16")

    diff_bf16 = float(jnp.abs(y_xla_bf16.astype(jnp.float32) - y_splash_bf16.astype(jnp.float32)).max())
    diff_bf16_mean = float(jnp.abs(y_xla_bf16.astype(jnp.float32) - y_splash_bf16.astype(jnp.float32)).mean())
    print(f"[CHECK bf16] xla vs Splash max diff = {diff_bf16:.2e}")
    print(f"[CHECK bf16] mean diff = {diff_bf16_mean:.2e}")

    # Backward (1-step gradient)
    print("\n=== Backward (1-step gradient atol verify) ===")

    def loss_xla(q, k, v):
        if args.window_left < 0:
            y = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="xla")
        else:
            y = jax.nn.dot_product_attention(q, k, v, is_causal=True,
                                             local_window_size=(args.window_left, 0), implementation="xla")
        return y.sum()

    def loss_splash(q, k, v):
        return call_splash_batched(splash_kernel, q, k, v).sum()

    grad_xla = jax.grad(loss_xla, argnums=(0, 1, 2))(q_fp32, k_fp32, v_fp32)
    grad_splash = jax.grad(loss_splash, argnums=(0, 1, 2))(q_fp32, k_fp32, v_fp32)
    for i, name in enumerate(["q", "k", "v"]):
        d = float(jnp.abs(grad_xla[i] - grad_splash[i]).max())
        print(f"[CHECK grad fp32 {name}] max diff = {d:.2e}")

    # Timing (5-call avg, after warmup)
    print("\n=== Timing (5-call avg, after 2 warmup) ===")
    for _ in range(2):
        jax.block_until_ready(call_splash_batched(splash_kernel, q_bf16, k_bf16, v_bf16))
        jax.block_until_ready(jax.nn.dot_product_attention(q_bf16, k_bf16, v_bf16, is_causal=True, implementation="xla"))
    t0 = time.time()
    for _ in range(5):
        y = call_splash_batched(splash_kernel, q_bf16, k_bf16, v_bf16)
    jax.block_until_ready(y)
    splash_avg = (time.time() - t0) / 5
    t0 = time.time()
    for _ in range(5):
        y = jax.nn.dot_product_attention(q_bf16, k_bf16, v_bf16, is_causal=True, implementation="xla")
    jax.block_until_ready(y)
    xla_avg = (time.time() - t0) / 5
    print(f"Splash bf16 avg: {splash_avg * 1000:.2f} ms / call")
    print(f"xla    bf16 avg: {xla_avg * 1000:.2f} ms / call")
    print(f"Splash speedup: {xla_avg / splash_avg:.2f}x")

    # Summary
    print("\n=== Summary ===")
    print(f"  xla vs manual (fp32, sanity)  = {diff_xla_manual:.2e}")
    print(f"  xla vs Splash (fp32 forward)  = {diff_fp32:.2e}")
    print(f"  xla vs Splash (bf16 forward)  = {diff_bf16:.2e}")
    print(f"  Splash latency speedup (bf16) = {xla_avg / splash_avg:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
