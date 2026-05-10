"""Smoke tests for nanochat_jax.loss_eval."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from flax import nnx

from nanochat_jax.gpt import GPT, GPTConfig
from nanochat_jax.loss_eval import evaluate_bpb


def _tiny_model() -> GPT:
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
    )
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    return m


def _make_batches(n_batches, B=2, T=8, vocab_size=32, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_batches):
        x = jnp.asarray(rng.randint(0, vocab_size, size=(B, T)).astype(np.int32))
        y = jnp.asarray(rng.randint(0, vocab_size, size=(B, T)).astype(np.int32))
        out.append((x, y))
    return out


def test_evaluate_bpb_returns_finite_positive_value():
    m = _tiny_model()
    batches = _make_batches(3)
    token_bytes = jnp.asarray(np.full(32, 4, dtype=np.int32))
    bpb = evaluate_bpb(m, iter(batches), steps=3, token_bytes=token_bytes)
    assert math.isfinite(bpb)
    assert bpb > 0


def test_evaluate_bpb_returns_inf_when_all_bytes_zero():
    m = _tiny_model()
    batches = _make_batches(2)
    token_bytes = jnp.asarray(np.zeros(32, dtype=np.int32))
    bpb = evaluate_bpb(m, iter(batches), steps=2, token_bytes=token_bytes)
    assert bpb == float("inf")


def test_evaluate_bpb_handles_ignore_index():
    m = _tiny_model()
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.randint(0, 32, size=(2, 8)).astype(np.int32))
    y_np = rng.randint(0, 32, size=(2, 8)).astype(np.int32)
    y_np[0, 0] = -1  # ignore index
    y = jnp.asarray(y_np)
    token_bytes = jnp.asarray(np.full(32, 3, dtype=np.int32))

    bpb = evaluate_bpb(m, iter([(x, y)]), steps=1, token_bytes=token_bytes)
    assert math.isfinite(bpb)
    assert bpb > 0


def test_evaluate_bpb_uniform_initialization_is_near_log2_vocab():
    """For an effectively-random model, BPB approximates log2(vocab) / bytes_per_token."""
    m = _tiny_model()
    batches = _make_batches(8, seed=123)
    token_bytes = jnp.asarray(np.full(32, 4, dtype=np.int32))
    bpb = evaluate_bpb(m, iter(batches), steps=8, token_bytes=token_bytes)
    # log2(32) ~ 5; per byte ~ 5 / 4 = 1.25. Loose bounds for an untrained model.
    assert 0.5 < bpb < 5.0
