"""Smoke tests for nanochat_jax.engine: KV cache, sampling, calculator tool."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from nanochat_jax.engine import (
    KVCache,
    eval_with_timeout,
    sample_next_token,
    use_calculator,
)
from nanochat_jax.gpt import GPT, GPTConfig


def test_use_calculator_simple_math():
    assert use_calculator("1 + 2 * 3") == 7
    assert use_calculator("100/4") == 25


def test_use_calculator_rejects_power_operator():
    assert use_calculator("2 ** 10") is None


def test_use_calculator_rejects_dangerous_strings():
    assert use_calculator("__import__('os')") is None
    assert use_calculator("open('/etc/passwd')") is None


def test_use_calculator_handles_string_count():
    assert use_calculator("'foo'.count('o')") == 2


def test_eval_with_timeout_returns_value():
    assert eval_with_timeout("1 + 1") == 2


def test_eval_with_timeout_returns_none_on_error():
    assert eval_with_timeout("not valid python") is None


def test_kv_cache_init_state_zero():
    cache = KVCache(
        batch_size=2, num_heads=2, seq_len=4, head_dim=8,
        num_layers=2, n_embd=16,
    )
    assert cache.get_pos() == 0
    assert bool(cache.has_prev[...]) is False
    assert cache.k_cache[...].shape == (2, 2, 4, 2, 8)


def test_kv_cache_advance():
    cache = KVCache(
        batch_size=1, num_heads=2, seq_len=4, head_dim=8,
        num_layers=1, n_embd=16,
    )
    cache.advance(2)
    assert cache.get_pos() == 2
    cache.reset()
    assert cache.get_pos() == 0


def test_kv_cache_write_then_read():
    cache = KVCache(
        batch_size=1, num_heads=2, seq_len=4, head_dim=8,
        num_layers=1, n_embd=16,
    )
    k_new = jnp.ones((1, 2, 2, 8))  # write 2 timesteps
    v_new = 2 * jnp.ones((1, 2, 2, 8))
    cache.write_kv(0, k_new, v_new)
    k_layer, v_layer = cache.get_layer_cache(0)
    assert jnp.all(k_layer[:, :2, :, :] == 1.0)
    assert jnp.all(v_layer[:, :2, :, :] == 2.0)


def test_sample_next_token_greedy():
    logits = jnp.array([[1.0, 5.0, 2.0, 0.0]])
    key = jax.random.PRNGKey(0)
    token = sample_next_token(logits, key, temperature=0.0)
    assert token.shape == (1, 1)
    assert int(token[0, 0]) == 1


def test_sample_next_token_top_k():
    logits = jnp.array([[1.0, 5.0, 2.0, 0.0, -5.0]])
    key = jax.random.PRNGKey(0)
    token = sample_next_token(logits, key, temperature=1.0, top_k=2)
    assert token.shape == (1, 1)
    # Top-2 indices are 1 (5.0) and 2 (2.0); the sample must be one of these.
    assert int(token[0, 0]) in {1, 2}


def test_sample_next_token_full_vocab():
    logits = jnp.zeros((2, 5))
    key = jax.random.PRNGKey(0)
    token = sample_next_token(logits, key, temperature=1.0)
    assert token.shape == (2, 1)


def test_kv_cache_prefill_broadcasts_b1_to_bn():
    src = KVCache(
        batch_size=1, num_heads=2, seq_len=4, head_dim=8,
        num_layers=1, n_embd=16,
    )
    src.write_kv(0, jnp.ones((1, 3, 2, 8)), jnp.ones((1, 3, 2, 8)))
    src.advance(3)

    dst = KVCache(
        batch_size=4, num_heads=2, seq_len=4, head_dim=8,
        num_layers=1, n_embd=16,
    )
    dst.prefill(src)

    assert dst.get_pos() == 3
    k_layer, _ = dst.get_layer_cache(0)
    assert jnp.all(k_layer[:, :3, :, :] == 1.0)


def test_engine_decode_path_does_not_crash():
    """Build a tiny model and run a 1-token decode through the cached path."""
    cfg = GPTConfig(
        sequence_len=16, vocab_size=32, n_layer=2,
        n_head=2, n_kv_head=2, n_embd=32,
    )
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)

    cache = KVCache(
        batch_size=1, num_heads=2, seq_len=8, head_dim=cfg.n_embd // cfg.n_head,
        num_layers=cfg.n_layer, n_embd=cfg.n_embd, dtype=cfg.compute_dtype,
    )
    # Prefill (T=4) followed by a single-token decode (T=1).
    prefill = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
    _ = m(prefill, kv_cache=cache)
    step = jnp.array([[4]], dtype=jnp.int32)
    out = m(step, kv_cache=cache)
    assert out.shape == (1, 1, cfg.vocab_size)
