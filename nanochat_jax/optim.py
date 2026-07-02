"""Pure-functional MuonAdamW optimizer for JAX/Flax NNX.

Mirrors the PyTorch fused kernels in nanochat (``adamw_step_fused`` and
``muon_step_fused``) so numerical behavior aligns with the reference
implementation across frameworks.

Key entry points:

- :class:`MuonAdamWState` -- pytree-friendly dataclass holding the optimizer
  step counter plus per-key AdamW buffers and per-group Muon buffers.
- :func:`setup_optimizer_param_groups` -- builds the 6 AdamW groups +
  N Muon groups (one per unique matrix shape) that mirror upstream nanochat.
- :func:`init_optim_state` -- zero-initializes all buffers.
- :func:`adamw_step` -- fused AdamW (lerp expansion + decoupled weight decay
  + bias correction).
- :func:`polar_express_iters` -- 5-step Newton-Schulz with the Polar Express
  quintic coefficients.
- :func:`norm_muon_scale` -- NorMuon variance reduction.
- :func:`muon_step` -- fused Muon: Nesterov momentum, Polar Express,
  NorMuon, cautious weight decay.
- :func:`step_optim` -- top-level 1-step that dispatches each group.

Layout convention:

- ``params`` and ``grads`` are flat dicts in JAX/Flax kernel layout
  (``(in, out)`` for Linear). Returned ``new_params`` matches.
- ``MuonAdamWState.adamw[key]`` buffers share the JAX param layout.
- ``MuonAdamWState.muon[first_key]`` buffers are stacked along axis 0 across
  group params and stored in **PyTorch layout** ``(num_params, out, in)``.
  This matches PyTorch's ``optimizer.state[p]['momentum_buffer']`` exactly so
  cross-framework ``.npz`` comparisons need no transpose. ``muon_step``
  performs the JAX-to-PT swap at the function boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


# Polar Express quintic coefficients tuned for ``num_iters=5``,
# ``safety_factor=2e-2``, ``cushion=2`` (Amsel et al. 2025,
# https://arxiv.org/pdf/2505.16932). Framework-agnostic.
POLAR_EXPRESS_COEFFS: tuple[tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


@dataclass
class MuonAdamWState:
    """Pytree state for :func:`step_optim`.

    Fields:
    - ``step`` -- scalar ``int32`` 0-D array, incremented per call.
    - ``adamw`` -- ``{param_key: {"exp_avg": ..., "exp_avg_sq": ...}}`` (JAX layout).
    - ``muon`` -- ``{first_param_key: {"momentum_buffer": ..., "second_momentum_buffer": ...}}``
      stacked across group params, stored in PyTorch layout for cross-framework parity.
    """

    step: jax.Array
    adamw: dict[str, dict[str, jax.Array]] = field(default_factory=dict)
    muon: dict[str, dict[str, jax.Array]] = field(default_factory=dict)


def _is_matrix_key(key: str) -> bool:
    """Matrix params live under ``transformer.h.<i>.{attn|mlp}...``."""
    return key.startswith("transformer.h.")


def _is_value_embed_key(key: str) -> bool:
    return key.startswith("value_embeds.")


def _pt_shape_from_jax(jax_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Map the JAX kernel shape ``(in, out)`` to the PyTorch ``(out, in)``.

    Embeddings ``(vocab, dim)`` and 1-D scalars are unchanged.
    """
    if len(jax_shape) == 2:
        return (jax_shape[1], jax_shape[0])
    return jax_shape


# Order in which PyTorch yields ``transformer.h.parameters()``: the Muon
# stacked state is keyed by index, so we must mirror this order exactly.
_PT_ATTN_NAME_ORDER = ("c_q", "c_k", "c_v", "c_proj", "ve_gate")
_PT_MLP_NAME_ORDER = ("c_fc", "c_proj")


def _pt_matrix_sort_key(jax_key: str) -> tuple[int, int, int]:
    """Reproduce the PyTorch ``list(transformer.h.parameters())`` ordering."""
    parts = jax_key.split(".")
    layer = int(parts[2])
    submod = parts[3]
    name = parts[4]
    if submod == "attn":
        return (layer, 0, _PT_ATTN_NAME_ORDER.index(name))
    if submod == "mlp":
        return (layer, 1, _PT_MLP_NAME_ORDER.index(name))
    raise ValueError(f"Unknown submodule {submod!r} in {jax_key!r}")


def setup_optimizer_param_groups(
    params: dict[str, jax.Array],
    n_embd: int,
    *,
    unembedding_lr: float = 0.004,
    embedding_lr: float = 0.2,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.0,
    scalar_lr: float = 0.5,
    polar_express_dtype: jax.numpy.dtype | None = None,
) -> list[dict]:
    """Build the optimizer param groups.

    Six AdamW groups (lm_head, wte, value_embeds, resid_lambdas, x0_lambdas,
    smear) plus one Muon group per unique matrix shape. The d-model LR scale
    ``(n_embd / 768) ** -0.5`` is applied to AdamW embedding/unembedding/
    value_embeds; the Muon LR receives a per-group
    ``max(1, pt_shape[-2] / pt_shape[-1]) ** 0.5`` factor inside
    :func:`step_optim`.
    """
    dmodel_lr_scale = (n_embd / 768) ** -0.5

    # Matrix keys MUST follow PyTorch module-definition order so that the
    # stacked Muon state buffers align across frameworks.
    matrix_keys = sorted(
        (k for k in params if _is_matrix_key(k)), key=_pt_matrix_sort_key,
    )
    value_embeds_keys = sorted(k for k in params if _is_value_embed_key(k))
    embedding_keys = ["transformer.wte.embedding"]
    lm_head_keys = ["lm_head.kernel"]
    resid_keys = ["resid_lambdas"]
    x0_keys = ["x0_lambdas"]
    smear_keys = ["smear_gate.kernel", "smear_lambda", "backout_lambda"]

    expected = (
        embedding_keys + lm_head_keys + resid_keys + x0_keys + smear_keys
        + value_embeds_keys + matrix_keys
    )
    missing = [k for k in expected if k not in params]
    if missing:
        raise KeyError(f"setup_optimizer_param_groups: missing param keys {missing}")
    extras = [k for k in params if k not in expected]
    if extras:
        raise KeyError(f"setup_optimizer_param_groups: unknown param keys {extras}")

    groups: list[dict] = [
        dict(kind="adamw", param_keys=lm_head_keys,
             lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96),
             eps=1e-10, weight_decay=0.01),
        dict(kind="adamw", param_keys=embedding_keys,
             lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995),
             eps=1e-10, weight_decay=0.001),
        dict(kind="adamw", param_keys=value_embeds_keys,
             lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995),
             eps=1e-10, weight_decay=0.01),
        dict(kind="adamw", param_keys=resid_keys,
             lr=scalar_lr * 0.01, betas=(0.8, 0.95),
             eps=1e-10, weight_decay=0.05),
        dict(kind="adamw", param_keys=x0_keys,
             lr=scalar_lr, betas=(0.96, 0.95),  # higher beta1 for x0
             eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", param_keys=smear_keys,
             lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
    ]

    pt_shapes = sorted({_pt_shape_from_jax(params[k].shape) for k in matrix_keys})
    for pt_shape in pt_shapes:
        group_keys = [
            k for k in matrix_keys
            if _pt_shape_from_jax(params[k].shape) == pt_shape
        ]
        groups.append(dict(
            kind="muon", param_keys=group_keys, lr=matrix_lr,
            momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            polar_express_dtype=polar_express_dtype,
        ))

    return groups


def init_optim_state(
    param_groups: list[dict],
    params: dict[str, jax.Array],
) -> MuonAdamWState:
    """Initialize zero state for all groups.

    AdamW: ``exp_avg``/``exp_avg_sq`` per param (shape and dtype match the
    param). Muon: group-level ``momentum_buffer`` shape
    ``(num_params, *pt_shape)`` and a factored ``second_momentum_buffer``
    shape ``(num_params, pt_shape[-2], 1)`` (or ``(..., 1, pt_shape[-1])``
    if the matrix is wide), stored under the first param key in PyTorch layout.
    """
    state = MuonAdamWState(step=jnp.asarray(0, dtype=jnp.int32))
    for g in param_groups:
        if g["kind"] == "adamw":
            for k in g["param_keys"]:
                p = params[k]
                state.adamw[k] = {
                    "exp_avg": jnp.zeros_like(p),
                    "exp_avg_sq": jnp.zeros_like(p),
                }
        elif g["kind"] == "muon":
            keys = g["param_keys"]
            if not keys:
                continue
            first_key = keys[0]
            num_params = len(keys)
            jax_shape = params[first_key].shape
            pt_shape = _pt_shape_from_jax(jax_shape)
            mom_shape = (num_params, *pt_shape)
            if pt_shape[-2] >= pt_shape[-1]:
                second_shape = (num_params, pt_shape[-2], 1)
            else:
                second_shape = (num_params, 1, pt_shape[-1])
            dtype = params[first_key].dtype
            state.muon[first_key] = {
                "momentum_buffer": jnp.zeros(mom_shape, dtype=dtype),
                "second_momentum_buffer": jnp.zeros(second_shape, dtype=dtype),
            }
        else:
            raise ValueError(f"Unknown optimizer kind: {g['kind']!r}")
    return state


def adamw_step(
    p: jax.Array, g: jax.Array,
    exp_avg: jax.Array, exp_avg_sq: jax.Array,
    step: jax.Array, lr: jax.Array, b1: jax.Array, b2: jax.Array,
    eps: jax.Array, wd: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused AdamW step. Returns ``(p_new, exp_avg_new, exp_avg_sq_new)``.

    Order of operations:
    1. Decoupled weight decay: ``p *= 1 - lr*wd``.
    2. Lerp updates of running averages.
    3. Bias correction: ``bias{1,2} = 1 - {b1,b2}**step``.
    4. Param update: ``p -= (lr/bias1) * exp_avg / (sqrt(exp_avg_sq/bias2) + eps)``.

    Scalar hyperparameters are passed as 0-D ``jnp.float32`` arrays so JIT
    does not recompile when their numeric value changes.
    """
    # PyTorch's in-place ops preserve p / exp_avg dtype (e.g. bf16 wte
    # storage when init_weights cast embeddings). Mirror via explicit
    # ``.astype`` after the fp32 promoted compute.
    p_dtype = p.dtype
    exp_avg_dtype = exp_avg.dtype
    exp_avg_sq_dtype = exp_avg_sq.dtype
    p = p * (1 - lr * wd)
    exp_avg_new = exp_avg * b1 + g * (1 - b1)
    exp_avg_sq_new = exp_avg_sq * b2 + (g * g) * (1 - b2)
    bias1 = 1 - b1 ** step
    bias2 = 1 - b2 ** step
    denom = jnp.sqrt(exp_avg_sq_new / bias2) + eps
    step_size = lr / bias1
    p_new = p - step_size * (exp_avg_new / denom)
    return (
        p_new.astype(p_dtype),
        exp_avg_new.astype(exp_avg_dtype),
        exp_avg_sq_new.astype(exp_avg_sq_dtype),
    )


def polar_express_iters(g_pt: jax.Array, ns_steps: int = 5) -> jax.Array:
    """5-step Polar Express orthogonalization in PyTorch layout ``(..., out, in)``.

    Muon replaces each matrix gradient with the nearest (semi-)orthogonal matrix,
    which equalizes the update's singular values so no single direction dominates
    the step. The Polar Express is a quintic Newton-Schulz iteration that
    approximates that orthogonalization cheaply (no SVD).

    Computes ``X = orthogonalize(g)`` via Newton-Schulz with the Polar Express
    quintic coefficients. The branches handle tall vs wide matrices (the
    quadratic form ``X.T @ X`` or ``X @ X.T`` is whichever is smaller).
    Inputs are normalized by ``||g|| * 1.01 + 1e-6`` to keep eigenvalues
    bounded for the iteration.
    """
    norm = jnp.linalg.norm(g_pt, axis=(-2, -1), keepdims=True) * 1.01 + 1e-6
    X = g_pt / norm

    coeffs = POLAR_EXPRESS_COEFFS[:ns_steps]

    if g_pt.shape[-2] > g_pt.shape[-1]:  # tall
        for a, b, c in coeffs:
            A = X.swapaxes(-2, -1) @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # wide
        for a, b, c in coeffs:
            A = X @ X.swapaxes(-2, -1)
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    return X


def norm_muon_scale(
    g_orth: jax.Array, second_buf: jax.Array,
    beta2: jax.Array, red_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """NorMuon variance reduction (Lu et al. 2025, https://arxiv.org/pdf/2510.05491).

    Per-neuron / per-column adaptive scaling that normalizes update magnitudes
    after orthogonalization (Muon's output has non-uniform per-row scales).
    Inputs are in PyTorch layout. ``red_dim`` is ``-1`` if the matrix is tall
    (``shape[-2] >= shape[-1]``) and ``-2`` otherwise.
    """
    v_mean = jnp.mean(g_orth.astype(jnp.float32) ** 2, axis=red_dim, keepdims=True)
    red_dim_size = g_orth.shape[red_dim]
    v_norm = jnp.sqrt(jnp.sum(v_mean, axis=(-2, -1), keepdims=True) * red_dim_size)
    second_buf_new = (
        second_buf * beta2
        + v_mean.astype(second_buf.dtype) * (1 - beta2)
    )
    step_size = jnp.maximum(second_buf_new, 1e-10) ** (-0.5)
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.astype(jnp.float32) ** 2
    v_norm_new = jnp.sqrt(jnp.sum(scaled_sq_sum, axis=(-2, -1), keepdims=True))
    final_scale = step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))
    g_scaled = g_orth * final_scale.astype(g_orth.dtype)
    return g_scaled, second_buf_new


def muon_step(
    stacked_g_jax: jax.Array, stacked_p_jax: jax.Array,
    momentum_buffer: jax.Array, second_momentum_buffer: jax.Array,
    lr: jax.Array, momentum: jax.Array, wd: jax.Array, beta2: jax.Array,
    ns_steps: int,
    *,
    polar_express_dtype: jax.numpy.dtype | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fused Muon step.

    Sequence:
    1. Nesterov momentum (lerp expansion).
    2. Optional cast of the momentum-updated gradient to ``polar_express_dtype``
       before orthogonalization. Defaults to fp32; setting to bf16 mirrors
       upstream's Newton-Schulz input cast (``X = g.bfloat16()``) — the
       published recipe keeps fp32 here.
    3. Polar Express orthogonalization (5-step Newton-Schulz).
    4. NorMuon variance reduction.
    5. Cautious weight decay + parameter update:
       ``mask = (g * p) >= 0; p -= lr*g + lr*wd*p*mask``.

    Inputs/outputs use JAX layout ``(num_params, in, out)``; momentum and
    second-moment buffers stay in PyTorch layout
    ``(num_params, out, in)``.
    """
    # JAX (in, out) -> PT (out, in)
    g_pt = stacked_g_jax.swapaxes(-2, -1)
    p_pt = stacked_p_jax.swapaxes(-2, -1)

    momentum_buffer_new = (
        momentum_buffer + (1 - momentum) * (g_pt - momentum_buffer)
    )
    g_step = g_pt + momentum * (momentum_buffer_new - g_pt)

    if polar_express_dtype is not None and polar_express_dtype != jax.numpy.float32:
        g_step = g_step.astype(polar_express_dtype)

    g_orth = polar_express_iters(g_step, ns_steps=ns_steps)

    red_dim = -1 if g_orth.shape[-2] >= g_orth.shape[-1] else -2
    g_scaled, second_momentum_buffer_new = norm_muon_scale(
        g_orth, second_momentum_buffer, beta2, red_dim
    )

    # Cautious weight decay: gate the weight decay by sign agreement.
    # Note: bf16 * fp32 -> fp32 in JAX, so this is fp32 even when g_scaled is bf16.
    mask = (g_scaled * p_pt) >= 0
    p_pt_new = p_pt - lr * g_scaled - lr * wd * p_pt * mask.astype(p_pt.dtype)

    return p_pt_new.swapaxes(-2, -1), momentum_buffer_new, second_momentum_buffer_new


def step_optim(
    state: MuonAdamWState,
    grads: dict[str, jax.Array],
    params: dict[str, jax.Array],
    param_groups: list[dict],
) -> tuple[dict[str, jax.Array], MuonAdamWState]:
    """One MuonAdamW step.

    Iterates ``param_groups`` and dispatches each to :func:`adamw_step` or
    :func:`muon_step`. Inputs are not mutated; returns the new params and
    new state. AdamW groups are processed first, then Muon by shape.
    """
    new_step = state.step + 1
    new_adamw: dict[str, dict[str, jax.Array]] = {}
    new_muon: dict[str, dict[str, jax.Array]] = {}
    new_params: dict[str, jax.Array] = dict(params)

    step_f = new_step.astype(jnp.float32)

    for g in param_groups:
        if g["kind"] == "adamw":
            lr = jnp.asarray(g["lr"], dtype=jnp.float32)
            b1 = jnp.asarray(g["betas"][0], dtype=jnp.float32)
            b2 = jnp.asarray(g["betas"][1], dtype=jnp.float32)
            eps = jnp.asarray(g["eps"], dtype=jnp.float32)
            wd = jnp.asarray(g["weight_decay"], dtype=jnp.float32)
            for k in g["param_keys"]:
                buf = state.adamw[k]
                p_new, exp_avg_new, exp_avg_sq_new = adamw_step(
                    params[k], grads[k],
                    buf["exp_avg"], buf["exp_avg_sq"],
                    step_f, lr, b1, b2, eps, wd,
                )
                new_params[k] = p_new
                new_adamw[k] = {
                    "exp_avg": exp_avg_new, "exp_avg_sq": exp_avg_sq_new,
                }
        elif g["kind"] == "muon":
            keys = g["param_keys"]
            if not keys:
                continue
            first_key = keys[0]
            buf = state.muon[first_key]
            pt_shape = _pt_shape_from_jax(params[first_key].shape)
            lr_group = g["lr"] * max(1.0, pt_shape[-2] / pt_shape[-1]) ** 0.5
            lr = jnp.asarray(lr_group, dtype=jnp.float32)
            momentum = jnp.asarray(g["momentum"], dtype=jnp.float32)
            wd = jnp.asarray(g["weight_decay"], dtype=jnp.float32)
            beta2 = jnp.asarray(g["beta2"], dtype=jnp.float32)
            ns_steps = int(g["ns_steps"])
            polar_express_dtype = g.get("polar_express_dtype", None)

            stacked_g = jnp.stack([grads[k] for k in keys], axis=0)
            stacked_p = jnp.stack([params[k] for k in keys], axis=0)

            stacked_p_new, mom_new, second_new = muon_step(
                stacked_g, stacked_p,
                buf["momentum_buffer"], buf["second_momentum_buffer"],
                lr, momentum, wd, beta2, ns_steps,
                polar_express_dtype=polar_express_dtype,
            )
            for i, k in enumerate(keys):
                new_params[k] = stacked_p_new[i]
            new_muon[first_key] = {
                "momentum_buffer": mom_new,
                "second_momentum_buffer": second_new,
            }
        else:
            raise ValueError(f"Unknown optimizer kind: {g['kind']!r}")

    new_state = MuonAdamWState(step=new_step, adamw=new_adamw, muon=new_muon)
    return new_params, new_state
