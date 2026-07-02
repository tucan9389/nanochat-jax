"""Gradient helpers for the NNX GPT model.

The flat-dict keys produced here align with the JAX-side keys returned by
:func:`nanochat_jax.weight_converter.pt_state_dict_to_jax`, which lets tests
compare PyTorch-side gradients (after weight conversion) against JAX-side
gradients key-for-key without any extra rewrite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import nnx

from nanochat_jax.gpt import GPT


def compute_loss_and_grad(
    model: GPT,
    idx: jax.Array,
    targets: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Compute the scalar loss and per-parameter gradient.

    Equivalent to PyTorch's ``loss = model(x, y); loss.backward()``, but the
    gradient is returned as a flat dict keyed for direct comparison.

    Forward uses no PRNG, so the result is deterministic given the model
    state, ``idx``, and ``targets``. With the default init, ``Param`` leaves
    stay fp32 and only the forward activations follow ``cfg.compute_dtype``
    (the opt-in ``init_weights(cast_embeddings_to_compute_dtype=True)`` path
    stores wte/value_embeds — and hence their grads — in bf16).
    """
    def loss_fn(m: GPT) -> jax.Array:
        return m(idx, targets=targets)

    loss, grad_state = nnx.value_and_grad(loss_fn)(model)
    grad_dict = nnx_state_to_flat_dict(grad_state)
    return loss, grad_dict


def nnx_state_to_flat_dict(state: nnx.statelib.State) -> dict[str, jax.Array]:
    """Flatten an :class:`nnx.statelib.State` to a flat dot-separated dict.

    Path keys are converted as follows:
    - :class:`jax.tree_util.DictKey` (NNX ``Dict`` / dataclass attrs) -> ``str(key)``
    - :class:`jax.tree_util.SequenceKey` (NNX ``List`` index) -> ``str(idx)``
    - :class:`jax.tree_util.GetAttrKey` (Python attribute) -> ``name``

    The trailing ``GetAttrKey('value')`` from NNX's ``Param.value`` wrapper is
    stripped so flat keys end at the logical parameter name (e.g.
    ``c_q.kernel``, ``smear_lambda``).
    """
    flat: dict[str, jax.Array] = {}
    for path, leaf in jtu.tree_flatten_with_path(state)[0]:
        parts: list[str] = []
        for elt in path:
            if isinstance(elt, jtu.DictKey):
                parts.append(str(elt.key))
            elif isinstance(elt, jtu.SequenceKey):
                parts.append(str(elt.idx))
            elif isinstance(elt, jtu.GetAttrKey):
                parts.append(elt.name)
            else:  # pragma: no cover -- defensive, unknown jax tree key type
                parts.append(str(elt))
        if parts and parts[-1] == "value":
            parts = parts[:-1]
        flat[".".join(parts)] = leaf
    return flat


def grad_dict_dtype_summary(grad_dict: dict[str, jax.Array]) -> dict[str, int]:
    """Return a count of keys per dtype string (e.g. ``{"float32": 32}``).

    With the default init every JAX-side gradient is fp32 because ``Param``
    leaves stay fp32 even with ``compute_dtype=bf16`` (the opt-in bf16-emb
    cast is the exception). PyTorch-side references may be mixed (bf16 for
    embeddings cast to compute dtype, fp32 elsewhere); use
    :func:`cast_grad_dict_to_fp32` to align before comparing.
    """
    counts: dict[str, int] = {}
    for v in grad_dict.values():
        d = str(v.dtype)
        counts[d] = counts.get(d, 0) + 1
    return counts


def cast_grad_dict_to_fp32(
    grad_dict: dict[str, jnp.ndarray],
) -> dict[str, jax.Array]:
    """Cast every gradient to ``jnp.float32``.

    bf16 -> fp32 promotion is exact (every bf16 value is representable in
    fp32), so this introduces no comparison noise. Accepts numpy or JAX
    arrays as input.
    """
    return {k: jnp.asarray(v).astype(jnp.float32) for k, v in grad_dict.items()}
