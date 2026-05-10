"""Smoke tests for nanochat_jax.core_eval helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nanochat_jax.core_eval import (
    batch_sequences_lm,
    batch_sequences_mc,
    batch_sequences_schema,
    find_common_length,
    forward_model,
    render_prompts_lm,
    render_prompts_mc,
    render_prompts_schema,
    stack_sequences,
)


class _FakeTokenizer:
    """Char-level tokenizer used as a stand-in for unit tests."""

    BOS = 0

    def get_bos_token_id(self):
        return self.BOS

    def encode(self, texts, prepend=None):
        out = []
        for t in texts:
            ids = [self.BOS] if prepend is not None else []
            ids.extend([1 + (ord(c) % 100) for c in t])
            out.append(ids)
        return out

    def __call__(self, texts, prepend=None):
        return self.encode(texts, prepend=prepend)


def test_find_common_length_left():
    assert find_common_length([[1, 2, 3, 4], [1, 2, 5, 6]], "left") == 2


def test_find_common_length_right():
    assert find_common_length([[1, 2, 3, 4], [9, 8, 3, 4]], "right") == 2


def test_find_common_length_full():
    assert find_common_length([[1, 2, 3], [1, 2, 3]], "left") == 3


def test_stack_sequences_pads_to_max():
    tokens = [[1, 2, 3], [4, 5], [6]]
    out = stack_sequences(tokens, pad_token_id=0)
    assert out.shape == (3, 3)
    assert out[1, 2] == 0  # pad


def test_stack_sequences_pow2_padding():
    tokens = [[1] * 70]  # > 64
    out = stack_sequences(tokens, pad_token_id=0, pad_to_pow2=True)
    assert out.shape == (1, 128)


def test_stack_sequences_pow2_min_64():
    tokens = [[1, 2, 3]]
    out = stack_sequences(tokens, pad_token_id=0, pad_to_pow2=True)
    assert out.shape == (1, 64)


def test_render_prompts_mc_no_fewshot():
    item = {"query": "Q?", "choices": ["A", "B", "C"], "gold": 0}
    prompts = render_prompts_mc(item, continuation_delimiter=" ", fewshot_examples=None)
    assert len(prompts) == 3
    assert prompts[0].endswith("Q? A")


def test_render_prompts_schema_runs():
    item = {"context_options": ["ctx1", "ctx2"], "continuation": "x", "gold": 0}
    prompts = render_prompts_schema(item, continuation_delimiter=" ", fewshot_examples=[])
    assert len(prompts) == 2


def test_render_prompts_lm_returns_two():
    item = {"context": "ctx", "continuation": "cont"}
    prompts = render_prompts_lm(item, continuation_delimiter=" ", fewshot_examples=[])
    assert len(prompts) == 2
    assert prompts[1].endswith("cont")


def test_batch_sequences_mc_finds_common_prefix():
    tok = _FakeTokenizer()
    prompts = ["Q? A", "Q? B"]
    tokens, start, end = batch_sequences_mc(tok, prompts)
    assert len(tokens) == 2
    assert start[0] == start[1]
    # Last token differs ('A' vs 'B').
    assert end[0] == len(tokens[0])


def test_batch_sequences_schema_finds_common_suffix():
    tok = _FakeTokenizer()
    prompts = ["AAA xx", "BBBB xx"]
    tokens, start, end = batch_sequences_schema(tok, prompts)
    assert end[0] - start[0] == end[1] - start[1]


def test_batch_sequences_lm_returns_singleton():
    tok = _FakeTokenizer()
    prompts = ["abc", "abcdef"]
    tokens, start, end = batch_sequences_lm(tok, prompts)
    assert len(tokens) == 1
    assert end[0] == len(tokens[0])


def test_forward_model_returns_loss_and_predictions():
    """Build a tiny model, run forward, sanity-check shapes."""
    from flax import nnx
    from nanochat_jax.gpt import GPT, GPTConfig

    cfg = GPTConfig(
        sequence_len=16, vocab_size=32, n_layer=2,
        n_head=2, n_kv_head=2, n_embd=32,
    )
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)

    input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    losses, predictions = forward_model(m, input_ids)
    assert losses.shape == (1, 5)
    assert predictions.shape == (1, 5)
    # Last column is set to NaN by design.
    assert jnp.isnan(losses[0, -1])
