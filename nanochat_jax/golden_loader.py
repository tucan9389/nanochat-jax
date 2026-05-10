"""Loader for ``.npz`` golden snapshots produced by ``scripts/generate_golden.py``.

A golden snapshot bundles a PyTorch reference run (forward intermediates,
gradients, optimizer state, optional training trajectory, optional BPB eval)
into a single ``.npz`` for downstream JAX/Flax NNX comparison.

Per-array dtype labels are restored from a ``_dtype_map`` JSON entry because
``np.savez_compressed`` does not preserve the ``ml_dtypes.bfloat16`` label
(the raw bytes survive but the dtype is reported as ``|V2`` void). On load
this loader calls ``arr.view(dtype)`` for any key whose stored dtype label
disagrees with the on-disk numpy dtype. fp32 / int keys round-trip natively.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np


_DTYPE_RESTORE_VIEW: dict[str, np.dtype] = {
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
}


@dataclass
class GoldenSnapshot:
    """One loaded golden ``.npz`` file.

    Most fields are optional: they are populated only when the generator was
    invoked with the corresponding ``with_*`` flag. Empty containers / ``None``
    indicate the snapshot did not capture that stage.
    """

    weights: dict[str, np.ndarray]
    """Initial state_dict (PT-format keys), dtype-restored."""

    tensors: dict[str, np.ndarray]
    """Forward-pass intermediates (named per generator granularity)."""

    grads: dict[str, np.ndarray]
    """Per-parameter gradients (PT-format keys), dtype matches PT autograd
    storage. Empty unless the generator captured grads."""

    weights_after: dict[str, np.ndarray]
    """state_dict after a single optimizer step. Empty unless the generator
    ran one optimizer step."""

    optim_state: dict[str, np.ndarray]
    """Optimizer state buffers after a single step, flat-keyed
    ``<state_dict_key>/<attr>``. AdamW: ``exp_avg``, ``exp_avg_sq``, ``step``.
    Muon: ``momentum_buffer``, ``second_momentum_buffer`` per group, stored
    under the first param key."""

    train_loss_curve: np.ndarray | None
    """Per-step pre-backward loss values from an N-step training loop,
    shape ``(num_iterations,) float32``. ``None`` unless the generator ran a
    training loop."""

    train_weights_after_N: dict[str, np.ndarray]
    """state_dict after the N-th training step. Empty unless the generator
    ran a training loop."""

    train_optim_state_after_N: dict[str, np.ndarray]
    """Optimizer state after the N-th training step, same flat-key convention
    as ``optim_state``."""

    train_per_step_grads: dict[str, np.ndarray]
    """Per-step gradients for every step of the training loop, each value
    shaped ``(N, *grad_shape)``. ``train_per_step_grads[k][i]`` is the
    gradient for parameter ``k`` computed at the start of step ``i`` (before
    the optimizer step). Used by per-step strict tests that feed the PT
    gradient into JAX optim, bypassing chaotic divergence."""

    train_per_step_weights_after: dict[str, np.ndarray]
    """Per-step weights after each optimizer step, shaped ``(N, *weight_shape)``.
    The initial weights (input for step 0) are not included; they are in
    ``weights``."""

    train_per_step_optim_state_after: dict[str, np.ndarray]
    """Per-step optimizer state after each optimizer step, same flat-key
    convention as ``optim_state``, shaped ``(N, *attr_shape)``."""

    bpb_eval_input_ids: np.ndarray | None
    """Eval batches' ``input_ids``, shape ``(eval_steps, B, T) int64``."""

    bpb_eval_targets: np.ndarray | None
    """Eval batches' ``targets``, shape ``(eval_steps, B, T) int64`` (with
    ``-1`` ignore_index at a configured masking step)."""

    bpb_eval_token_bytes: np.ndarray | None
    """Per-token byte length array, shape ``(vocab_size,) int32``."""

    bpb_eval_per_step_loss_2d: np.ndarray | None
    """Per-step PyTorch ``model(x, y, loss_reduction='none').view(-1)``,
    shape ``(eval_steps, B*T) fp32``. Used for element-wise verification of
    ``cross_entropy_with_ignore(reduction='none')``."""

    bpb_eval_total_nats: float | None
    """Accumulated nats (fp32 scalar)."""

    bpb_eval_total_bytes: int | None
    """Accumulated byte count (int64 scalar)."""

    bpb_eval_expected_bpb: float | None
    """Final BPB scalar = ``total_nats / (log(2) * total_bytes)``. May be
    ``float('inf')`` if ``total_bytes == 0``."""

    input_ids: np.ndarray
    """``(B, T) int64`` token ids."""

    targets: np.ndarray | None
    """``(B, T) int64`` targets, or ``None`` for logits-only goldens."""

    meta: dict[str, Any]
    """Generator metadata (depth, config, granularity, dtype, seed, ...)."""


def _restore_dtype(arr: np.ndarray, dtype_label: str) -> np.ndarray:
    """Restore the dtype label of ``arr``.

    For dtypes that ``np.savez`` cannot persist (bf16: stored as ``|V2`` void)
    the raw bytes are ``.view``-ed back to the target dtype. fp32 / int round-
    trip natively.
    """
    if dtype_label in _DTYPE_RESTORE_VIEW:
        target = _DTYPE_RESTORE_VIEW[dtype_label]
        if arr.dtype != target:
            return arr.view(target)
        return arr
    if str(arr.dtype) != dtype_label:
        raise ValueError(
            f"_dtype_map mismatch: label={dtype_label!r}, on-disk={arr.dtype!r}; "
            f"add this dtype to _DTYPE_RESTORE_VIEW if it requires .view restoration"
        )
    return arr


def load_golden(path: str | Path) -> GoldenSnapshot:
    """Load a ``golden_*.npz`` file into a :class:`GoldenSnapshot`.

    Reserved keys ``_dtype_map`` and ``_meta`` are required. All array keys
    are routed by prefix; unknown prefixes raise ``ValueError`` so the loader
    stays in sync with the generator.
    """
    path = Path(path)
    data = np.load(path)

    if "_dtype_map" not in data.files:
        raise ValueError(f"{path}: missing reserved key '_dtype_map' (legacy or malformed .npz)")
    if "_meta" not in data.files:
        raise ValueError(f"{path}: missing reserved key '_meta'")

    dtype_map: dict[str, str] = json.loads(str(data["_dtype_map"]))
    meta: dict[str, Any] = json.loads(str(data["_meta"]))

    weights: dict[str, np.ndarray] = {}
    tensors: dict[str, np.ndarray] = {}
    grads: dict[str, np.ndarray] = {}
    weights_after: dict[str, np.ndarray] = {}
    optim_state: dict[str, np.ndarray] = {}
    train_loss_curve: np.ndarray | None = None
    train_weights_after_N: dict[str, np.ndarray] = {}
    train_optim_state_after_N: dict[str, np.ndarray] = {}
    train_per_step_grads: dict[str, np.ndarray] = {}
    train_per_step_weights_after: dict[str, np.ndarray] = {}
    train_per_step_optim_state_after: dict[str, np.ndarray] = {}
    bpb_eval_input_ids: np.ndarray | None = None
    bpb_eval_targets: np.ndarray | None = None
    bpb_eval_token_bytes: np.ndarray | None = None
    bpb_eval_per_step_loss_2d: np.ndarray | None = None
    bpb_eval_total_nats: float | None = None
    bpb_eval_total_bytes: int | None = None
    bpb_eval_expected_bpb: float | None = None
    input_ids: np.ndarray | None = None
    targets: np.ndarray | None = None

    for key in data.files:
        if key in ("_dtype_map", "_meta"):
            continue
        if key not in dtype_map:
            raise ValueError(
                f"{path}: key {key!r} not in _dtype_map "
                f"(generator must populate dtype_map for every array)"
            )
        arr = _restore_dtype(data[key], dtype_map[key])
        # Order matters: longer prefixes must be tested before their shorter
        # siblings to avoid mis-routing.
        if key == "train/loss_curve":
            train_loss_curve = arr
        elif key.startswith("train/per_step_grads/"):
            train_per_step_grads[key[len("train/per_step_grads/") :]] = arr
        elif key.startswith("train/per_step_weights_after/"):
            train_per_step_weights_after[key[len("train/per_step_weights_after/") :]] = arr
        elif key.startswith("train/per_step_optim_state_after/"):
            train_per_step_optim_state_after[
                key[len("train/per_step_optim_state_after/") :]
            ] = arr
        elif key.startswith("train/weights_after_N/"):
            train_weights_after_N[key[len("train/weights_after_N/") :]] = arr
        elif key.startswith("train/optim_state_after_N/"):
            train_optim_state_after_N[key[len("train/optim_state_after_N/") :]] = arr
        elif key == "bpb_eval/input_ids":
            bpb_eval_input_ids = arr
        elif key == "bpb_eval/targets":
            bpb_eval_targets = arr
        elif key == "bpb_eval/token_bytes":
            bpb_eval_token_bytes = arr
        elif key == "bpb_eval/per_step_loss_2d":
            bpb_eval_per_step_loss_2d = arr
        elif key == "bpb_eval/total_nats":
            bpb_eval_total_nats = float(arr)
        elif key == "bpb_eval/total_bytes":
            bpb_eval_total_bytes = int(arr)
        elif key == "bpb_eval/expected_bpb":
            bpb_eval_expected_bpb = float(arr)
        elif key.startswith("weights_after/"):
            weights_after[key[len("weights_after/") :]] = arr
        elif key.startswith("optim_state/"):
            optim_state[key[len("optim_state/") :]] = arr
        elif key.startswith("weight/"):
            weights[key[len("weight/") :]] = arr
        elif key.startswith("tensor/"):
            tensors[key[len("tensor/") :]] = arr
        elif key.startswith("grad/"):
            grads[key[len("grad/") :]] = arr
        elif key == "input/input_ids":
            input_ids = arr
        elif key == "input/targets":
            targets = arr
        else:
            raise ValueError(f"{path}: unknown key prefix {key!r}")

    if input_ids is None:
        raise ValueError(f"{path}: missing required key 'input/input_ids'")

    return GoldenSnapshot(
        weights=weights,
        tensors=tensors,
        grads=grads,
        weights_after=weights_after,
        optim_state=optim_state,
        train_loss_curve=train_loss_curve,
        train_weights_after_N=train_weights_after_N,
        train_optim_state_after_N=train_optim_state_after_N,
        train_per_step_grads=train_per_step_grads,
        train_per_step_weights_after=train_per_step_weights_after,
        train_per_step_optim_state_after=train_per_step_optim_state_after,
        bpb_eval_input_ids=bpb_eval_input_ids,
        bpb_eval_targets=bpb_eval_targets,
        bpb_eval_token_bytes=bpb_eval_token_bytes,
        bpb_eval_per_step_loss_2d=bpb_eval_per_step_loss_2d,
        bpb_eval_total_nats=bpb_eval_total_nats,
        bpb_eval_total_bytes=bpb_eval_total_bytes,
        bpb_eval_expected_bpb=bpb_eval_expected_bpb,
        input_ids=input_ids,
        targets=targets,
        meta=meta,
    )
