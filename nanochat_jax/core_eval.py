"""CORE metric evaluation -- DCLM benchmark (22 ICL tasks).

Reference: https://arxiv.org/abs/2406.11794. CORE is upstream nanochat's
primary metric ("time to GPT-2"), as recorded in ``dev/LEADERBOARD.md``.

The forward pass is jit-cached per model instance and uses power-of-2 padding
so the JIT cache amortizes across the ~22 ICL tasks: only a handful of
unique ``(B, T)`` shapes are compiled instead of one per example.
"""

from __future__ import annotations

import random
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jinja2 import Template

# -----------------------------------------------------------------------------
# Prompt rendering
# -----------------------------------------------------------------------------


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompts = [template.render(choice=choice, **context) for choice in item["choices"]]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question."""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompts = [
        template.render(context=context_option, **context)
        for context_option in item["context_options"]
    ]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a language modeling task.

    Returns 2 prompts: without and with the continuation. The context is
    explicitly trimmed of trailing whitespace because some datasets ship
    contexts with stray trailing whitespace.
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    """Find the length of the common prefix or suffix across token sequences.

    ``direction="left"`` for prefix, ``"right"`` for suffix.
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        "left": range(min_len),
        "right": range(-1, -min_len - 1, -1),
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id, *, pad_to_pow2=False):
    """Stack a list of token sequences and pad to the longest on the right.

    With ``pad_to_pow2=True`` the sequence length is rounded up to the
    nearest power of two (minimum 64). This caps the number of unique
    JIT-compiled shapes to roughly seven (64/128/256/512/1024/2048/4096),
    which avoids per-example recompilation across the 22 ICL tasks. Padded
    positions are never read by the downstream slicing logic, so the algorithm
    is unchanged.
    """
    bsz, max_len = len(tokens), max(len(x) for x in tokens)
    if pad_to_pow2:
        seq_len = max(64, 1 << (max_len - 1).bit_length()) if max_len > 0 else 64
    else:
        seq_len = max_len
    input_ids = np.full((bsz, seq_len), pad_token_id, dtype=np.int64)
    for i, x in enumerate(tokens):
        input_ids[i, : len(x)] = np.asarray(x, dtype=np.int64)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    """Tokenize multiple-choice prompts. Contexts are common (shared prefix);
    the answer starts where the prompts diverge."""
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    """Tokenize schema prompts. Continuations are common (shared suffix);
    contexts vary."""
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    """Tokenize LM-task prompts (without/with continuation). Batch size is 1."""
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], (
        "prompt without is supposed to be a prefix of prompt with"
    )
    return [tokens_with], [start_idx], [end_idx]


# -----------------------------------------------------------------------------
# Forward + per-example evaluation
# -----------------------------------------------------------------------------


# Module-level JIT cache. Closure-captured model object means the model is not
# a JIT input (no abstract array tracer requirement). The cache is keyed by
# input_ids shape internally, so frozen-state eval models reuse compiled code.
_jit_forward_cache: dict[int, Any] = {}


def _get_jitted_forward(model):
    """Return a JIT-compiled forward bound to ``model`` via closure.

    JIT amortizes TPU dispatch overhead across many examples. With power-of-2
    padding the cache collapses thousands of unique shapes (22 tasks x N
    examples) to a handful, giving order-of-magnitude speedups on TPU.
    """
    key = id(model)
    if key not in _jit_forward_cache:
        @jax.jit
        def _forward(input_ids):
            outputs = model(input_ids)
            target_ids = jnp.roll(input_ids, shift=-1, axis=1)
            log_probs = jax.nn.log_softmax(outputs, axis=-1)
            losses = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)
            losses = losses.at[:, -1].set(jnp.nan)
            predictions = jnp.argmax(outputs, axis=-1)
            return losses, predictions
        _jit_forward_cache[key] = _forward
    return _jit_forward_cache[key]


def forward_model(model, input_ids):
    """Forward pass returning ``(losses, predictions)`` per token.

    The last column of ``losses`` is set to ``nan`` because there is no
    autoregressive target there.
    """
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    jitted = _get_jitted_forward(model)
    return jitted(input_ids)


def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate one example. Returns ``True`` if correct, else ``False``.

    Few-shot determinism: ``random.Random(1234 + idx)`` seeds the few-shot
    sampler so the same ``idx`` produces the same few-shot batch across runs.
    The ``device`` argument is kept for signature compatibility with the
    PyTorch reference and is not used here.
    """
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]

    fewshot_examples: list[Any] = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # NOTE: intentional divergence from upstream, kept to preserve the
    # published-d24 CORE score (see dev/REPRODUCTION-GUARDS.md). Our native
    # model always has config.sequence_len, so prompts longer than it are
    # ALWAYS cropped here; upstream only crops models that expose max_seq_len,
    # which its native GPT does not — upstream never truncates.
    max_seq_len = getattr(getattr(model, "config", None), "sequence_len", None)
    if max_seq_len is not None:
        max_tokens = max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id, pad_to_pow2=True)

    losses, predictions = forward_model(model, input_ids)

    if task_type == "language_modeling":
        si = start_idxs[0]
        ei = end_idxs[0]
        predicted_tokens = predictions[0, si - 1 : ei - 1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = bool(jnp.all(predicted_tokens == jnp.asarray(actual_tokens)))
    elif task_type in ("multiple_choice", "schema"):
        mean_losses = [
            float(losses[i, si - 1 : ei - 1].mean())
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))
        ]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item["gold"]
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """Evaluate one task across many examples; dispatch via rank-stride.

    Single-process runs go through the natural path (CPU / single TPU).
    Multi-host runs auto-activate when ``jax.process_count() > 1``: each rank
    handles a stride and the partial result vectors are combined via
    ``process_allgather + sum``.
    """
    rank = jax.process_index()
    world_size = jax.process_count()
    correct = np.zeros(len(data), dtype=np.float32)
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    if world_size > 1:
        from jax.experimental.multihost_utils import process_allgather

        all_correct = process_allgather(jnp.asarray(correct))
        correct = np.asarray(all_correct).sum(axis=0)
    mean_correct = float(np.mean(correct))
    return mean_correct
