"""PyTorch nanochat ``state_dict`` <-> JAX flat pytree converter.

Handles three categories of nanochat parameters (exhaustive whitelist):

================ =========================================================== ============== =========
Category         PT key pattern                                              JAX key suffix Transpose
================ =========================================================== ============== =========
Embedding        ``transformer.wte.weight``, ``value_embeds.<i>.weight``     ``.embedding`` No
Linear           ``transformer.h.<i>.{attn|mlp}.<name>.weight``,             ``.kernel``    Yes
                 ``lm_head.weight``, ``smear_gate.weight``
Scalar Parameter ``resid_lambdas``, ``x0_lambdas``, ``smear_lambda``,        (same)         No
                 ``backout_lambda``
================ =========================================================== ============== =========

Linear weights transpose between PyTorch ``(out, in)`` and JAX/Flax
``(in, out)`` to match :class:`nanochat_jax.layers.NanochatLinear`. The
key suffix changes (``.weight`` -> ``.kernel`` / ``.embedding``) follow
NNX convention.

bf16 round-trips through fp32 numpy via ``ml_dtypes.bfloat16`` (bit-exact
both directions); ``np.savez`` does not preserve the bf16 dtype label so
.npz callers must save a separate dtype map.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

if TYPE_CHECKING:
    import torch

    from nanochat_jax.layers import NanochatLinear


_KEY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^transformer\.wte\.weight$"), "embedding"),
    (re.compile(r"^value_embeds\.\d+\.weight$"), "embedding"),
    (
        re.compile(
            r"^transformer\.h\.\d+\.attn\.(c_q|c_k|c_v|c_proj|ve_gate)\.weight$"
        ),
        "linear",
    ),
    (re.compile(r"^transformer\.h\.\d+\.mlp\.(c_fc|c_proj)\.weight$"), "linear"),
    (re.compile(r"^lm_head\.weight$"), "linear"),
    (re.compile(r"^smear_gate\.weight$"), "linear"),
    (re.compile(r"^resid_lambdas$"), "scalar"),
    (re.compile(r"^x0_lambdas$"), "scalar"),
    (re.compile(r"^smear_lambda$"), "scalar"),
    (re.compile(r"^backout_lambda$"), "scalar"),
]


def _classify(key: str) -> str:
    """Classify a PyTorch state_dict key. Raises ValueError on unknown keys.

    Catches typos, model variants with new params, or accidental cos/sin
    buffer leakage (those should not be in state_dict due to
    ``persistent=False``).
    """
    for pattern, kind in _KEY_PATTERNS:
        if pattern.match(key):
            return kind
    raise ValueError(
        f"Unknown nanochat state_dict key: {key!r}. Expected one of: "
        f"transformer.wte.weight, value_embeds.<i>.weight, "
        f"transformer.h.<i>.{{attn,mlp}}.<name>.weight, lm_head.weight, "
        f"smear_gate.weight, or scalar Parameter "
        f"(resid_lambdas/x0_lambdas/smear_lambda/backout_lambda)."
    )


def _pt_key_to_jax_key(pt_key: str, kind: str) -> str:
    if kind == "embedding":
        return pt_key[: -len(".weight")] + ".embedding"
    if kind == "linear":
        return pt_key[: -len(".weight")] + ".kernel"
    return pt_key


def _jax_key_to_pt_key(jax_key: str) -> tuple[str, str]:
    """Return ``(pt_key, kind)`` for a JAX-side key, validating against the whitelist."""
    if jax_key.endswith(".embedding"):
        pt_key = jax_key[: -len(".embedding")] + ".weight"
        suffix_kind = "embedding"
    elif jax_key.endswith(".kernel"):
        pt_key = jax_key[: -len(".kernel")] + ".weight"
        suffix_kind = "linear"
    else:
        pt_key = jax_key
        suffix_kind = "scalar"
    expected = _classify(pt_key)
    if expected != suffix_kind:
        raise ValueError(
            f"JAX key {jax_key!r} suffix={suffix_kind!r} does not match "
            f"expected category {expected!r} for PT key {pt_key!r}."
        )
    return pt_key, suffix_kind


def _torch_to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    """Convert torch tensor to numpy. bf16 routes through fp32 (bit-exact)."""
    import torch

    if tensor.dtype == torch.bfloat16:
        return tensor.detach().float().numpy().astype(ml_dtypes.bfloat16)
    return tensor.detach().numpy()


def _numpy_to_torch(arr: np.ndarray) -> "torch.Tensor":
    """Convert numpy to torch tensor. bf16 routes through fp32 (bit-exact)."""
    import torch

    if arr.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(arr.astype(np.float32).copy()).to(torch.bfloat16)
    return torch.from_numpy(arr.copy())


def pt_state_dict_to_jax(
    state_dict: "dict[str, torch.Tensor]",
    *,
    target_dtype: jnp.dtype | None = None,
) -> dict[str, jax.Array]:
    """Convert a nanochat PyTorch state_dict to a flat JAX pytree.

    With ``target_dtype=None`` (default) each tensor's dtype is preserved
    per-key, which handles mixed precision (e.g. bf16 embeddings + fp32
    Linear master weights).
    """
    out: dict[str, jax.Array] = {}
    for pt_key, tensor in state_dict.items():
        kind = _classify(pt_key)
        jax_key = _pt_key_to_jax_key(pt_key, kind)
        np_arr = _torch_to_numpy(tensor)
        if kind == "linear":
            np_arr = np_arr.T
        jx = jnp.asarray(np_arr)
        if target_dtype is not None:
            jx = jx.astype(target_dtype)
        out[jax_key] = jx
    return out


def jax_to_pt_state_dict(
    jax_pytree: dict[str, jax.Array],
    *,
    target_dtype: "torch.dtype | None" = None,
) -> "dict[str, torch.Tensor]":
    """Reverse of :func:`pt_state_dict_to_jax`.

    Linear ``kernel`` ``(in, out)`` -> PT ``weight`` ``(out, in)``.
    Embedding stays ``(vocab, features)``. Scalar keys unchanged.
    """
    out: dict[str, "torch.Tensor"] = {}
    for jax_key, arr in jax_pytree.items():
        pt_key, kind = _jax_key_to_pt_key(jax_key)
        np_arr = np.asarray(arr)
        if kind == "linear":
            np_arr = np_arr.T
        tensor = _numpy_to_torch(np_arr)
        if target_dtype is not None:
            tensor = tensor.to(target_dtype)
        out[pt_key] = tensor
    return out


def inject_pt_linear_weight(
    nnx_linear: "NanochatLinear",
    pt_weight: "torch.Tensor",
) -> None:
    """Copy a PyTorch Linear weight ``(out, in)`` into a NanochatLinear in place.

    Useful for golden fixture loaders that want to install reference weights
    directly into an NNX module.
    """
    np_w = _torch_to_numpy(pt_weight).T
    expected = (nnx_linear.in_features, nnx_linear.out_features)
    if np_w.shape != expected:
        raise ValueError(
            f"Shape mismatch: pt_weight transpose has shape {np_w.shape}, "
            f"but NanochatLinear expects (in_features, out_features) = {expected}."
        )
    nnx_linear.kernel[...] = jnp.asarray(np_w)
