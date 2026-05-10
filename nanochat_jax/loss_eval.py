"""Bits-per-byte (BPB) evaluation.

BPB is a tokenization-vocab-size-independent loss metric. Per-token nat
losses are weighted by the byte length of the target token (special tokens
have ``token_bytes[i] == 0`` and are masked out), summed across batches, and
divided by ``log(2) * total_bytes``::

    bpb = sum(loss_per_token * (token_bytes > 0 mask)) / (log(2) * sum(token_bytes))

Single-process by default; if ``jax.process_count() > 1`` the local sums are
all-gathered across processes via ``jax.experimental.multihost_utils`` so
multi-host evaluation produces a global metric.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import jax
import jax.numpy as jnp

from nanochat_jax.gpt import GPT, cross_entropy_with_ignore


def _forward_loss_per_token(
    model: GPT, idx: jax.Array, targets: jax.Array
) -> jax.Array:
    """Forward pass returning per-token loss ``(B*T,)``.

    Mirrors PyTorch ``model(x, y, loss_reduction='none').view(-1)`` by
    composing ``GPT.__call__`` (logits-only) with
    ``cross_entropy_with_ignore(reduction='none')`` so the model signature
    stays unchanged. Ignored positions (target == ``-1``) get a per-token
    loss of zero.
    """
    logits = model(idx)
    return cross_entropy_with_ignore(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-1,
        reduction="none",
    )


def evaluate_bpb(
    model: GPT,
    batches: Iterable[tuple[jax.Array, jax.Array]],
    steps: int,
    token_bytes: jax.Array,
) -> float:
    """Compute bits-per-byte (BPB) over ``steps`` evaluation batches.

    Parameters
    ----------
    model
        Initialized GPT instance. No train-mode toggle is needed because
        nanochat does not use dropout or other training-only ops.
    batches
        Iterable yielding ``(input_ids, targets)`` pairs. Both arrays are
        ``(B, T)`` integers; ``targets == -1`` marks ignored positions.
    steps
        Number of batches to draw from ``batches``.
    token_bytes
        ``(vocab_size,)`` integer array. ``token_bytes[i] == 0`` marks a
        special token whose loss is excluded.

    Returns
    -------
    float
        BPB scalar, or ``float('inf')`` if every byte was masked out.
    """
    total_nats = jnp.float32(0.0)
    total_bytes = jnp.int32(0)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)

        loss2d = _forward_loss_per_token(model, x, y)
        y_flat = y.reshape(-1)

        # Slow path handles negative target ids (the ignore sentinel).
        # The branch decision uses host-side Python control flow, mirroring
        # the PyTorch reference -- the heavy compute lives inside the model's
        # JIT-traced ``__call__``.
        has_negative = bool((y_flat < 0).any())
        if has_negative:
            valid = y_flat >= 0
            y_safe = jnp.where(valid, y_flat, jnp.zeros_like(y_flat))
            num_bytes2d = jnp.where(
                valid,
                token_bytes[y_safe],
                jnp.zeros_like(y_flat, dtype=token_bytes.dtype),
            )
            total_nats = total_nats + (
                loss2d * (num_bytes2d > 0).astype(loss2d.dtype)
            ).sum()
            total_bytes = total_bytes + num_bytes2d.sum().astype(jnp.int32)
        else:
            num_bytes2d = token_bytes[y_flat]
            total_nats = total_nats + (
                loss2d * (num_bytes2d > 0).astype(loss2d.dtype)
            ).sum()
            total_bytes = total_bytes + num_bytes2d.sum().astype(jnp.int32)

    total_nats_host = float(total_nats)
    total_bytes_host = int(total_bytes)

    # Multi-host all-reduce SUM. process_allgather is a process-level
    # collective (not device-level), so it runs after the local accumulation
    # completes. No-op when running on a single process.
    if jax.process_count() > 1:
        nats_local = jnp.array(total_nats_host, dtype=jnp.float32)
        bytes_local = jnp.array(total_bytes_host, dtype=jnp.int64)
        nats_all = jax.experimental.multihost_utils.process_allgather(nats_local)
        bytes_all = jax.experimental.multihost_utils.process_allgather(bytes_local)
        total_nats_host = float(jnp.sum(nats_all))
        total_bytes_host = int(jnp.sum(bytes_all))

    if total_bytes_host == 0:
        return float("inf")
    return total_nats_host / (math.log(2) * total_bytes_host)
