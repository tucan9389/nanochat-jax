"""Inference engine for nanochat-jax models with a KV cache.

The :class:`KVCache` is an ``nnx.Module`` whose buffers are ``nnx.Variable``
arrays so the cache mutates cleanly inside ``nnx.jit``. Cache writes use
``jax.lax.dynamic_update_slice_in_dim``; the attention path reads the full
allocated cache and uses an explicit mask (built in ``gpt.py``) to align the
query at the latest cache position. The :class:`Engine` runs an outer
Python loop for tool dispatch and a per-step model forward (which carries
all the JIT-friendly compute).

Calculator tool (``use_calculator``) and the timeout helper are
framework-agnostic Python; they are Unix-only because they rely on
``signal.SIGALRM``.
"""

from __future__ import annotations

import signal
import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import jax
import jax.numpy as jnp
from flax import nnx


# -----------------------------------------------------------------------------
# Calculator tool
# -----------------------------------------------------------------------------


@contextmanager
def timeout(duration: int, formula: str):
    """Signal-based timeout context manager. Unix-only (uses ``SIGALRM``)."""

    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula: str, max_time: int = 3):
    """Sandboxed ``eval`` with a hard timeout.

    Returns the eval result, or ``None`` on timeout / exception. Uses an
    empty ``__builtins__`` so most dangerous functions are unreachable.
    """
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        return None


def use_calculator(expr: str):
    """Calculator tool that evaluates a math expression or a restricted string op.

    Allowed: pure math expressions over digits and ``*+-/.()``, plus
    ``str.count``-style operations. Disallowed: ``**``, dunder names,
    ``import``/``exec``/``eval``/``compile``/``open``/``file``/``input``,
    ``getattr``/``setattr``/``delattr``/``hasattr``, ``globals``/``locals``/
    ``vars``/``dir``.
    """
    expr = expr.replace(",", "")

    if all(x in "0123456789*+-/.() " for x in expr):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)

    allowed_chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789'\"()._ "
    )
    if not all(x in allowed_chars for x in expr):
        return None

    dangerous_patterns = [
        "__", "import", "exec", "eval", "compile", "open", "file",
        "input", "raw_input", "globals", "locals", "vars", "dir",
        "getattr", "setattr", "delattr", "hasattr",
    ]
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    if ".count(" not in expr:
        return None

    return eval_with_timeout(expr)


# -----------------------------------------------------------------------------
# KV cache
# -----------------------------------------------------------------------------


class KVCache(nnx.Module):
    """KV cache for autoregressive decoding.

    Storage layout (BTHD, matching the reshape in :class:`gpt.CausalSelfAttention`):

    - ``k_cache`` / ``v_cache``: ``(n_layer, B, T_max, n_kv_head, head_dim)``
    - ``cache_seqlen``: scalar ``int32`` -- all rows share the same write position
    - ``prev_embedding``: ``(B, 1, n_embd)`` for the smear cache
    - ``has_prev``: scalar ``bool`` sentinel that replaces the ``None`` check
      from the PyTorch reference (so it stays jit-traceable)
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        n_embd: int,
        dtype: jnp.dtype = jnp.float32,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        del rngs  # zero-init only; kept for nnx.Module signature compatibility
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.n_embd = n_embd
        self.dtype = dtype

        kv_shape = (num_layers, batch_size, seq_len, num_heads, head_dim)
        self.k_cache = nnx.Variable(jnp.zeros(kv_shape, dtype=dtype))
        self.v_cache = nnx.Variable(jnp.zeros(kv_shape, dtype=dtype))
        self.cache_seqlen = nnx.Variable(jnp.array(0, dtype=jnp.int32))

        prev_shape = (batch_size, 1, n_embd)
        self.prev_embedding = nnx.Variable(jnp.zeros(prev_shape, dtype=dtype))
        self.has_prev = nnx.Variable(jnp.array(False))

    def reset(self) -> None:
        """Reset the cache to the empty state."""
        self.cache_seqlen[...] = jnp.array(0, dtype=jnp.int32)
        self.has_prev[...] = jnp.array(False)
        # No need to zero the buffers; cache_seqlen=0 means writes start at 0.

    def get_pos(self) -> int:
        """Current cache position as a Python int (host-side, not jit-traceable)."""
        return int(self.cache_seqlen[...])

    def get_layer_cache(self, layer_idx: int) -> tuple[jax.Array, jax.Array]:
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def write_kv(self, layer_idx: int, k_new: jax.Array, v_new: jax.Array) -> None:
        """Write ``k_new`` / ``v_new`` at the current cache position for a layer."""
        k_layer = self.k_cache[layer_idx]
        v_layer = self.v_cache[layer_idx]

        k_layer_new = jax.lax.dynamic_update_slice_in_dim(
            k_layer, k_new, self.cache_seqlen[...], axis=1
        )
        v_layer_new = jax.lax.dynamic_update_slice_in_dim(
            v_layer, v_new, self.cache_seqlen[...], axis=1
        )

        self.k_cache[...] = self.k_cache[...].at[layer_idx].set(k_layer_new)
        self.v_cache[...] = self.v_cache[...].at[layer_idx].set(v_layer_new)

    def advance(self, num_tokens: int | jax.Array) -> None:
        """Advance the cache position. Called once per forward, after all layers."""
        self.cache_seqlen[...] = self.cache_seqlen[...] + jnp.asarray(
            num_tokens, dtype=jnp.int32
        )

    def prefill(self, other: "KVCache") -> None:
        """Copy cached KV from another (B=1) cache into this (B=N) one.

        Used for batch=1 prefill followed by batch=N decode: the prompt is run
        once with B=1 and the resulting cache is broadcast across N samples.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert (
            self.n_layers == other.n_layers
            and self.n_heads == other.n_heads
            and self.head_dim == other.head_dim
        ), "KV cache shape mismatch"
        assert self.max_seq_len >= other.max_seq_len, (
            f"prefill target max_seq_len {self.max_seq_len} < source "
            f"{other.max_seq_len}"
        )

        other_pos = other.get_pos()

        k_src = other.k_cache[...][:, :, :other_pos, :, :]
        v_src = other.v_cache[...][:, :, :other_pos, :, :]
        k_src_b = jnp.broadcast_to(
            k_src,
            (other.n_layers, self.batch_size, other_pos, other.n_heads, other.head_dim),
        )
        v_src_b = jnp.broadcast_to(
            v_src,
            (other.n_layers, self.batch_size, other_pos, other.n_heads, other.head_dim),
        )
        self.k_cache[...] = jax.lax.dynamic_update_slice_in_dim(
            self.k_cache[...], k_src_b, 0, axis=2
        )
        self.v_cache[...] = jax.lax.dynamic_update_slice_in_dim(
            self.v_cache[...], v_src_b, 0, axis=2
        )
        self.cache_seqlen[...] = jnp.array(other_pos, dtype=jnp.int32)

        if bool(other.has_prev[...]):
            prev_src = other.prev_embedding[...]
            prev_b = jnp.broadcast_to(
                prev_src, (self.batch_size, 1, prev_src.shape[-1])
            )
            self.prev_embedding[...] = prev_b
            self.has_prev[...] = jnp.array(True)


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------


def sample_next_token(
    logits: jax.Array,
    key: jax.Array,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> jax.Array:
    """Sample a single next token from logits ``(B, vocab)`` and return ``(B, 1)``.

    ``temperature == 0`` is greedy (``argmax``). For ``top_k > 0`` the top-k
    logits are kept and ``jax.random.categorical`` samples from them; otherwise
    sampling runs over the full vocabulary. ``categorical`` accepts logits
    directly so an explicit ``softmax`` is unnecessary.
    """
    assert temperature >= 0.0, "temperature must be non-negative"

    if temperature == 0.0:
        return jnp.argmax(logits, axis=-1, keepdims=True)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        vals, idx = jax.lax.top_k(logits, k)
        vals = vals / temperature
        choice = jax.random.categorical(key, vals, axis=-1)
        return jnp.take_along_axis(idx, choice[..., None], axis=-1)

    logits_scaled = logits / temperature
    choice = jax.random.categorical(key, logits_scaled, axis=-1)
    return choice[..., None]


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------


@dataclass
class RowState:
    """Per-row generation state. Tool-use is a Python-only state machine."""

    current_tokens: list[int] = field(default_factory=list)
    forced_tokens: deque = field(default_factory=deque)
    in_python_block: bool = False
    python_expr_tokens: list[int] = field(default_factory=list)
    completed: bool = False


class Engine:
    """KV-cached inference engine for nanochat-jax GPT models.

    The model forward (jit-friendly) is invoked from an outer Python loop
    that tracks per-row state, dispatches tool calls (the calculator), and
    yields token columns for streaming consumers.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        tokens: list[int],
        num_samples: int = 1,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int = 42,
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Streaming generation.

        Yields ``(token_column, token_masks)`` per step:
        - ``token_column``: list of next-token ids (length ``num_samples``)
        - ``token_masks``: 1 for sampled tokens, 0 for forced tokens (tool output)
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), (
            "expecting list of ints"
        )

        cfg = self.model.config
        dtype = cfg.compute_dtype

        key = jax.random.PRNGKey(seed)

        get_special = lambda s: self.tokenizer.encode_special(s)  # noqa: E731
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # 1) Batch=1 prefill of prompt tokens.
        kv_model_kwargs = {
            "num_heads": cfg.n_kv_head,
            "head_dim": cfg.n_embd // cfg.n_head,
            "num_layers": cfg.n_layer,
            "n_embd": cfg.n_embd,
        }
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids_prefill = jnp.array([tokens], dtype=jnp.int32)
        logits_prefill = self.model(
            ids_prefill, kv_cache=kv_cache_prefill
        )
        logits = jnp.broadcast_to(
            logits_prefill[:, -1, :], (num_samples, logits_prefill.shape[-1])
        )

        # 2) Replicate KV cache for each sample.
        kv_length_hint = (
            (len(tokens) + max_tokens) if max_tokens is not None
            else cfg.sequence_len
        )
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # 3) Per-row generation states.
        row_states = [
            RowState(current_tokens=tokens.copy()) for _ in range(num_samples)
        ]

        # 4) Main generation loop.
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            key, subkey = jax.random.split(key)

            next_ids = sample_next_token(logits, subkey, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = (
                    state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                )
                token_column.append(next_token)
                state.current_tokens.append(next_token)

                if next_token == assistant_end or next_token == bos:
                    state.completed = True

                # Tool-use state machine (calculator).
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            ids_step = jnp.array(token_column, dtype=jnp.int32)[:, None]
            logits = self.model(
                ids_step, kv_cache=kv_cache_decode
            )[:, -1, :]

    def generate_batch(
        self,
        tokens: list[int],
        num_samples: int = 1,
        **kwargs,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Non-streaming batch generation.

        Returns ``(results, masks)`` -- each is a list of length ``num_samples``
        whose entries are token sequences and corresponding masks. Terminal
        tokens (``<|assistant_end|>``, ``<|bos|>``) are not included in the
        returned sequences.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks
