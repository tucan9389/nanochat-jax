"""Checkpoint save / load for nanochat-jax models.

Mirrors upstream nanochat/checkpoint_manager.py with framework substitutions:

- ``torch.load(map_location='cpu')`` is still used (torch is a dev dep, used as
  the ``.pt`` reader); JAX device placement happens later inside
  :func:`nanochat_jax.weight_converter.pt_state_dict_to_jax`.
- ``model.load_state_dict(...)`` becomes :func:`inject_pt_state_dict` --
  the converted flat dict is written into the NNX :class:`GPT` in-place.
- ``model.eval()`` is a no-op (the GPT has no dropout / batchnorm).
- The ``device`` argument from upstream is replaced by ``compute_dtype``.

Optimizer state is serialized in a JAX-port-specific flat format (mirroring
``MuonAdamWState`` exactly), not the PyTorch param-groups state_dict format.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import dataclasses

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax import nnx

from nanochat_jax.common import get_base_dir
from nanochat_jax.gpt import GPT, GPTConfig
from nanochat_jax.tokenizer import get_tokenizer
from nanochat_jax.weight_converter import pt_state_dict_to_jax


logger = logging.getLogger(__name__)


def _log0(message: str) -> None:
    """Log only on the master process."""
    if jax.process_index() == 0:
        logger.info(message)


# -----------------------------------------------------------------------------
# Patch helpers for old checkpoints with missing keys
# -----------------------------------------------------------------------------


def _patch_missing_config_keys(model_config_kwargs: dict) -> None:
    """Add default values for new config keys missing in old checkpoints."""
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["window_pattern"] = "L"
        _log0("Patching missing window_pattern in model config to 'L'")


def _drop_unknown_config_keys(model_config_kwargs: dict) -> None:
    """Drop meta keys that this checkout's ``GPTConfig`` does not implement."""
    valid = {field.name for field in dataclasses.fields(GPTConfig)}
    unknown = sorted(set(model_config_kwargs) - valid)
    for key in unknown:
        model_config_kwargs.pop(key)
    if unknown:
        _log0(f"Ignoring unsupported model_config keys: {unknown}")


def _patch_missing_keys(model_data: dict, model_config: GPTConfig) -> None:
    """Add default scalar params that may be missing in old checkpoints."""
    n_layer = model_config.n_layer
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = np.ones(n_layer, dtype=np.float32)
        _log0("Patching missing resid_lambdas in model data to 1.0")
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = np.zeros(n_layer, dtype=np.float32)
        _log0("Patching missing x0_lambdas in model data to 0.0")


# -----------------------------------------------------------------------------
# State dict injection
# -----------------------------------------------------------------------------


def inject_pt_state_dict(
    model: GPT, jax_pytree: dict[str, jax.Array]
) -> None:
    """Inject a flat dict of jax arrays into an NNX GPT in-place.

    Walks the NNX :class:`nnx.Param` graph, matches each leaf path to its
    corresponding flat key, and updates the value via NNX in-place mutation.
    The flat-key conversion mirrors :func:`nanochat_jax.grad_utils.nnx_state_to_flat_dict`
    so weight-converter output keys align 1:1 with NNX Param leaves.

    Raises:
        KeyError: if any NNX Param leaf has no matching key in ``jax_pytree``
            (likely a meta_<step>.json model_config mismatch).
        ValueError: if any ``jax_pytree`` key is not consumed by an NNX Param
            leaf (orphaned PT state_dict key).
    """
    state = nnx.state(model, nnx.Param)
    flat_with_path, treedef = jtu.tree_flatten_with_path(state)

    consumed: set[str] = set()
    new_leaves: list[jax.Array] = []
    for path, leaf in flat_with_path:
        parts: list[str] = []
        for elt in path:
            if isinstance(elt, jtu.DictKey):
                parts.append(str(elt.key))
            elif isinstance(elt, jtu.SequenceKey):
                parts.append(str(elt.idx))
            elif isinstance(elt, jtu.GetAttrKey):
                parts.append(elt.name)
            else:  # pragma: no cover -- defensive
                parts.append(str(elt))
        if parts and parts[-1] == "value":
            parts = parts[:-1]
        flat_key = ".".join(parts)

        if flat_key not in jax_pytree:
            raise KeyError(
                f"NNX Param path {flat_key!r} has no matching key in "
                f"jax_pytree (likely meta_<step>.json model_config mismatch). "
                f"Available pytree keys (first 5): "
                f"{sorted(jax_pytree.keys())[:5]}..."
            )
        new_leaves.append(jax_pytree[flat_key].astype(leaf.dtype))
        consumed.add(flat_key)

    unconsumed = set(jax_pytree.keys()) - consumed
    if unconsumed:
        raise ValueError(
            f"jax_pytree has {len(unconsumed)} keys not consumed by NNX "
            f"Params (orphaned PT state_dict keys): "
            f"{sorted(unconsumed)[:5]}..."
        )

    new_state = jtu.tree_unflatten(treedef, new_leaves)
    nnx.update(model, new_state)


# -----------------------------------------------------------------------------
# Checkpoint I/O
# -----------------------------------------------------------------------------


def load_checkpoint(
    checkpoint_dir: str,
    step: int,
    *,
    load_optimizer: bool = False,
    rank: int = 0,
) -> tuple[dict, dict | None, dict]:
    """Load ``(model_data, optimizer_data, meta_data)`` from a checkpoint directory.

    Reads ``model_<step:06d>.pt`` and ``meta_<step:06d>.json``; optionally also
    reads ``optim_<step:06d>_rank<rank>.pt``. Tensors stay on CPU
    until JAX device placement happens downstream.
    """
    import torch

    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model_data = torch.load(
        model_path, map_location="cpu", weights_only=False
    )

    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(
            checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt"
        )
        if os.path.exists(optimizer_path):
            optimizer_data = torch.load(
                optimizer_path, map_location="cpu", weights_only=False
            )

    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta json not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    return model_data, optimizer_data, meta_data


def build_model(
    checkpoint_dir: str,
    step: int,
    *,
    compute_dtype: jnp.dtype = jnp.float32,
    seed: int = 0,
):
    """Build an NNX :class:`GPT` from a checkpoint, plus tokenizer + meta.

    The default ``compute_dtype=jnp.float32`` is the safe path; pass
    ``jnp.bfloat16`` to opt into the bf16 forward cascade.
    """
    model_data, _, meta_data = load_checkpoint(
        checkpoint_dir, step, load_optimizer=False
    )

    # torch.compile prepends "_orig_mod." to all keys; strip it.
    model_data = {
        k.removeprefix("_orig_mod."): v for k, v in model_data.items()
    }

    model_config_kwargs = dict(meta_data["model_config"])
    _patch_missing_config_keys(model_config_kwargs)
    _drop_unknown_config_keys(model_config_kwargs)
    _log0(f"Building model with config: {model_config_kwargs}")

    model_config = GPTConfig(
        **model_config_kwargs, compute_dtype=compute_dtype
    )
    model = GPT(model_config, rngs=nnx.Rngs(seed))

    _patch_missing_keys(model_data, model_config)

    jax_pytree = pt_state_dict_to_jax(model_data, target_dtype=None)

    inject_pt_state_dict(model, jax_pytree)

    tokenizer = get_tokenizer()

    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} != model config "
        f"vocab_size {model_config_kwargs['vocab_size']}"
    )

    return model, tokenizer, meta_data


def find_largest_model(checkpoints_dir: str) -> str:
    """Find the largest model_tag in a checkpoints dir.

    Tries ``r"d(\\d+)"`` and picks the highest depth, otherwise falls back to
    the most-recently-updated subdir.
    """
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(
            f"Checkpoints dir not found: {checkpoints_dir}"
        )

    model_tags = [
        f
        for f in os.listdir(checkpoints_dir)
        if os.path.isdir(os.path.join(checkpoints_dir, f))
    ]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    model_tags.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)),
        reverse=True,
    )
    return model_tags[0]


def find_last_step(checkpoint_dir: str) -> int:
    """Find the largest step number from ``model_<step>.pt`` files."""
    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, "model_*.pt")
    )
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(
        max(
            os.path.basename(f).split("_")[-1].split(".")[0]
            for f in checkpoint_files
        )
    )
    return last_step


# -----------------------------------------------------------------------------
# Convenience load APIs
# -----------------------------------------------------------------------------


def load_model_from_dir(
    checkpoints_dir: str,
    *,
    compute_dtype: jnp.dtype = jnp.float32,
    model_tag: str | None = None,
    step: int | None = None,
    seed: int = 0,
):
    """:func:`build_model` with auto-detected ``model_tag`` + ``step``."""
    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
        _log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)
    _log0(f"Loading model from {checkpoint_dir} with step {step}")
    return build_model(
        checkpoint_dir, step, compute_dtype=compute_dtype, seed=seed
    )


def load_model(
    source: str,
    *,
    compute_dtype: jnp.dtype = jnp.float32,
    model_tag: str | None = None,
    step: int | None = None,
    seed: int = 0,
    base_dir: str | None = None,
):
    """Top-level: load a model by source (``"base"`` / ``"sft"`` / ``"rl"``).

    Resolves the checkpoints directory via :func:`get_base_dir` (override with
    ``base_dir``). ``"rl"`` raises :exc:`NotImplementedError` until the RL
    training path is ported.
    """
    model_dir_map = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }
    if source not in model_dir_map:
        raise ValueError(
            f"Invalid source {source!r}. Expected one of: "
            f"{list(model_dir_map.keys())}"
        )
    if source == "rl":
        raise NotImplementedError(
            "RL training path is not yet ported; use --source base or --source sft."
        )

    if base_dir is None:
        base_dir = get_base_dir()

    checkpoints_dir = os.path.join(base_dir, model_dir_map[source])
    return load_model_from_dir(
        checkpoints_dir,
        compute_dtype=compute_dtype,
        model_tag=model_tag,
        step=step,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# Save APIs
# -----------------------------------------------------------------------------


def _muon_adamw_state_to_pt_dict(optim_state) -> dict:
    """Serialize :class:`MuonAdamWState` to a flat dict for ``torch.save``.

    The dict structure is JAX-port specific (NOT the PyTorch optim state_dict
    format) and mirrors the dataclass exactly:

    - ``"step"`` -> numpy int32 scalar
    - ``"adamw.{param_key}.{exp_avg|exp_avg_sq}"`` -> numpy array (JAX layout)
    - ``"muon.{first_param_key}.{momentum_buffer|second_momentum_buffer}"``
      -> numpy array (PT layout, stacked first axis = num_params_in_group)
    """
    flat: dict[str, jax.Array | np.ndarray] = {}
    flat["step"] = np.asarray(optim_state.step)
    for param_key, bufs in optim_state.adamw.items():
        for buf_name, buf in bufs.items():
            flat[f"adamw.{param_key}.{buf_name}"] = np.asarray(buf)
    for first_key, bufs in optim_state.muon.items():
        for buf_name, buf in bufs.items():
            flat[f"muon.{first_key}.{buf_name}"] = np.asarray(buf)
    return flat


def _pt_dict_to_muon_adamw_state(flat: dict):
    """Reconstruct :class:`MuonAdamWState` from a flat dict."""
    from nanochat_jax.optim import MuonAdamWState

    step = jnp.asarray(np.asarray(flat["step"]))
    adamw: dict[str, dict[str, jax.Array]] = {}
    muon: dict[str, dict[str, jax.Array]] = {}
    for k, v in flat.items():
        if k == "step":
            continue
        try:
            import torch
            if isinstance(v, torch.Tensor):
                v_np = v.detach().cpu().numpy()
            else:
                v_np = np.asarray(v)
        except ImportError:
            v_np = np.asarray(v)
        v_jax = jnp.asarray(v_np)
        parts = k.split(".")
        if parts[0] == "adamw":
            buf_name = parts[-1]
            param_key = ".".join(parts[1:-1])
            adamw.setdefault(param_key, {})[buf_name] = v_jax
        elif parts[0] == "muon":
            buf_name = parts[-1]
            first_key = ".".join(parts[1:-1])
            muon.setdefault(first_key, {})[buf_name] = v_jax
        else:
            raise ValueError(f"Unrecognized optim flat key prefix: {parts[0]!r}")
    return MuonAdamWState(step=step, adamw=adamw, muon=muon)


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    model: GPT,
    optim_state=None,
    meta_data: dict | None = None,
    *,
    rank: int = 0,
) -> None:
    """Save model + (optional) optimizer + meta.

    Files written:
    - ``model_<step:06d>.pt`` (rank 0 only) -- PT state_dict format.
    - ``meta_<step:06d>.json`` (rank 0 only).
    - ``optim_<step:06d>_rank<r>.pt`` (per-rank, if ``optim_state`` is given) --
      JAX-port specific flat format.
    """
    import torch

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        from nanochat_jax.grad_utils import nnx_state_to_flat_dict
        from nanochat_jax.weight_converter import jax_to_pt_state_dict

        params_state = nnx.state(model, nnx.Param)
        flat_jax = nnx_state_to_flat_dict(params_state)
        pt_state_dict = jax_to_pt_state_dict(flat_jax)

        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(pt_state_dict, model_path)
        logger.info(f"Saved model parameters to: {model_path}")

        if meta_data is not None:
            meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)
            logger.info(f"Saved metadata to: {meta_path}")

    if optim_state is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optim_dict = _muon_adamw_state_to_pt_dict(optim_state)
        optim_path = os.path.join(
            checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt"
        )
        torch.save(optim_dict, optim_path)
        logger.info(f"Saved optimizer state to: {optim_path}")


def load_optimizer_state(
    source: str,
    *,
    rank: int = 0,
    model_tag: str | None = None,
    step: int | None = None,
    base_dir: str | None = None,
):
    """Load just the optimizer shard for a given rank.

    Returns ``None`` if the optim shard file is missing (cold-start fresh
    optimizer state).
    """
    import torch

    model_dir_map = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }
    if source not in model_dir_map:
        raise ValueError(
            f"Invalid source {source!r}. Expected one of: "
            f"{list(model_dir_map.keys())}"
        )

    if base_dir is None:
        base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir_map[source])

    if model_tag is None:
        model_tag = find_largest_model(checkpoints_dir)
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)

    optim_path = os.path.join(
        checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt"
    )
    if not os.path.exists(optim_path):
        _log0(f"Optimizer checkpoint not found: {optim_path}")
        return None

    _log0(f"Loading optimizer state from {optim_path}")
    optim_dict = torch.load(optim_path, map_location="cpu", weights_only=False)
    return _pt_dict_to_muon_adamw_state(optim_dict)
