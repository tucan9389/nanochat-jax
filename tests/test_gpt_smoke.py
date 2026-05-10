"""Smoke tests for nanochat_jax.gpt: shapes, basic forward, init_weights.

Deeper integration tests that depend on golden ``.npz`` snapshots live alongside
``scripts/generate_golden.py`` once that script is migrated.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from nanochat_jax.gpt import (
    GPT,
    GPTConfig,
    Block,
    CausalSelfAttention,
    MLP,
    cross_entropy_with_ignore,
    has_ve,
)


def _tiny_cfg(**overrides):
    """Smallest config that still exercises smear_gate (needs >= 24 channels)."""
    base = dict(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    base.update(overrides)
    return GPTConfig(**base)


def test_forward_returns_logits():
    cfg = _tiny_cfg()
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    idx = jnp.array([[1, 2, 3, 4, 5]])
    logits = m(idx)
    assert logits.shape == (1, 5, cfg.vocab_size)
    assert logits.dtype == jnp.float32


def test_forward_with_targets_returns_scalar_loss():
    cfg = _tiny_cfg()
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    idx = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 6]])
    loss = m(idx, targets=targets)
    assert loss.shape == ()
    # An untrained model should be near uniform: loss ~ log(vocab_size).
    assert 0.0 < float(loss) < 2 * math.log(cfg.vocab_size)


def test_init_weights_changes_parameters():
    cfg = _tiny_cfg()
    m = GPT(cfg, rngs=nnx.Rngs(0))
    before = jnp.asarray(m.transformer.wte.embedding.value)
    m.init_weights(seed=42)
    after = jnp.asarray(m.transformer.wte.embedding.value)
    assert not jnp.allclose(before, after)


def test_forward_with_intermediates_returns_expected_keys():
    cfg = _tiny_cfg(n_layer=2)
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    idx = jnp.array([[1, 2, 3, 4, 5]])

    minimal = m.forward_with_intermediates(idx, granularity="minimal")
    assert set(minimal.keys()) == {"x_after_wte", "logits"}

    block = m.forward_with_intermediates(idx, granularity="block")
    assert "x_after_smear" in block and "x0" in block and "x_after_backout" in block
    assert "block_0_out" in block and "block_1_out" in block

    high = m.forward_with_intermediates(idx, granularity="high")
    assert "x_after_norm0" in high and "block_0_input" in high
    assert "logits_pre_softcap" in high


def test_forward_with_intermediates_loss_when_targets_provided():
    cfg = _tiny_cfg()
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    ints = m.forward_with_intermediates(
        jnp.array([[1, 2, 3, 4, 5]]),
        targets=jnp.array([[2, 3, 4, 5, 6]]),
    )
    assert "loss" in ints and ints["loss"].shape == ()


def test_bf16_compute_propagates_through_forward():
    cfg = _tiny_cfg(compute_dtype=jnp.bfloat16)
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    idx = jnp.array([[1, 2, 3, 4, 5]])
    logits = m(idx)
    # Logits cast back to fp32 by design (loss / softcap stability).
    assert logits.dtype == jnp.float32
    ints = m.forward_with_intermediates(idx, granularity="block")
    assert ints["x_after_wte"].dtype == jnp.bfloat16


def test_sliding_window_pattern_runs():
    cfg = _tiny_cfg(n_layer=4, window_pattern="SSSL")
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    idx = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    logits = m(idx)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_gqa_runs():
    cfg = _tiny_cfg(n_head=4, n_kv_head=2)
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    idx = jnp.array([[1, 2, 3, 4, 5]])
    logits = m(idx)
    assert logits.shape == (1, 5, cfg.vocab_size)


def test_gqa_assertion_rejects_non_dividing_n_kv_head():
    with pytest.raises(AssertionError, match="n_kv_head must divide n_head"):
        cfg = _tiny_cfg(n_head=4, n_kv_head=3)
        GPT(cfg, rngs=nnx.Rngs(0))


def test_invalid_window_pattern_rejected():
    cfg = GPTConfig(
        sequence_len=16, vocab_size=32, n_layer=2, n_head=2, n_kv_head=2,
        n_embd=32, window_pattern="X",
    )
    with pytest.raises(AssertionError, match="Invalid window_pattern"):
        GPT(cfg, rngs=nnx.Rngs(0))


def test_has_ve_alternates_with_last_layer():
    # n_layer=4 -> last layer index is 3 (odd); has_ve odd layers.
    assert [has_ve(i, 4) for i in range(4)] == [False, True, False, True]
    # n_layer=3 -> last layer index is 2 (even); has_ve even layers.
    assert [has_ve(i, 3) for i in range(3)] == [True, False, True]


def test_cross_entropy_with_ignore_mean_skips_ignore_index():
    logits = jnp.array(
        [[[2.0, 1.0, 0.5], [0.1, 0.2, 0.3]]],
        dtype=jnp.float32,
    )
    targets = jnp.array([[0, -1]])
    loss = cross_entropy_with_ignore(
        logits.reshape(-1, 3), targets.reshape(-1)
    )
    # With one ignored target, mean reduces over only the valid entry.
    log_probs = jax.nn.log_softmax(logits.reshape(-1, 3), axis=-1)
    expected = -log_probs[0, 0]
    assert jnp.isclose(loss, expected)


def test_cross_entropy_with_ignore_none_returns_per_token():
    logits = jnp.zeros((4, 3), dtype=jnp.float32)
    targets = jnp.array([0, 1, -1, 2])
    per_token = cross_entropy_with_ignore(logits, targets, reduction="none")
    assert per_token.shape == (4,)
    assert per_token[2] == 0.0  # ignored
    # Uniform logits: nll for valid tokens equals log(3).
    assert jnp.allclose(per_token[0], math.log(3.0))


def test_mlp_module_runs():
    cfg = _tiny_cfg()
    mlp = MLP(cfg, rngs=nnx.Rngs(0))
    x = jnp.zeros((1, 4, cfg.n_embd))
    y = mlp(x)
    assert y.shape == x.shape


def test_block_module_runs_without_ve():
    cfg = _tiny_cfg()  # n_layer=2 -> layer 0 has no value embedding
    block = Block(cfg, layer_idx=0, rngs=nnx.Rngs(0))
    cos, sin = jnp.zeros((1, 4, 1, cfg.n_embd // cfg.n_head // 2)), jnp.zeros(
        (1, 4, 1, cfg.n_embd // cfg.n_head // 2)
    )
    x = jnp.zeros((1, 4, cfg.n_embd))
    y = block(x, None, (cos, sin))
    assert y.shape == x.shape


def test_block_module_runs_with_ve():
    cfg = _tiny_cfg()  # n_layer=2 -> layer 1 has value embedding
    block = Block(cfg, layer_idx=1, rngs=nnx.Rngs(0))
    cos, sin = jnp.zeros((1, 4, 1, cfg.n_embd // cfg.n_head // 2)), jnp.zeros(
        (1, 4, 1, cfg.n_embd // cfg.n_head // 2)
    )
    x = jnp.zeros((1, 4, cfg.n_embd))
    ve = jnp.zeros((1, 4, cfg.n_kv_head * (cfg.n_embd // cfg.n_head)))
    y = block(x, ve, (cos, sin))
    assert y.shape == x.shape


def test_attention_rejects_non_zero_right_window():
    cfg = _tiny_cfg()
    attn = CausalSelfAttention(cfg, layer_idx=0, rngs=nnx.Rngs(0))
    cos, sin = jnp.zeros((1, 4, 1, cfg.n_embd // cfg.n_head // 2)), jnp.zeros(
        (1, 4, 1, cfg.n_embd // cfg.n_head // 2)
    )
    x = jnp.zeros((1, 4, cfg.n_embd))
    ve = jnp.zeros((1, 4, cfg.n_kv_head * (cfg.n_embd // cfg.n_head)))
    with pytest.raises(AssertionError, match="right window must be 0"):
        attn(x, ve, (cos, sin), window_size=(8, 1))
