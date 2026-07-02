"""Pure-functional training-loop helpers for the JAX/Flax port.

Composes :func:`compute_loss_and_grad` (``nanochat_jax/grad_utils.py``) with
:func:`step_optim` (``nanochat_jax/optim.py``) into jitted training-step
builders, mirroring the inner training loop of upstream
``nanochat/scripts/base_train.py`` (``loss = model(x, y); loss.backward();
optimizer.step(); model.zero_grad()``) together with its learning-rate, Muon
momentum, and weight-decay schedules.

Public API:

- :func:`get_lr_multiplier` / :func:`get_muon_momentum` / :func:`get_weight_decay`
  — the LR / momentum / weight-decay schedules as framework-agnostic Python-float
  helpers (no JAX/NumPy/Torch). Being plain float arithmetic, they match the
  PyTorch reference bit-for-bit.
- :func:`init_train_state` — builds the optimizer param groups and zero-initialised
  optimizer state for a freshly constructed :class:`GPT`.
- :func:`make_train_step_sharded` / :func:`make_grad_accum_fns` /
  :func:`make_fused_train_step` — the sharded single-step, gradient-accumulating,
  and fused-accumulation step builders used by ``scripts/base_train.py`` and
  ``scripts/chat_sft.py``. Schedule values are computed on the host per step and
  passed in as JAX scalars, so steps 2..N hit the JIT cache without recompiling.

Determinism: given the same initial weights, batch sequence, and hyperparameters,
the trajectory is bit-exactly reproducible — the forward/backward, the optimizer
step, and the schedules are all free of PRNG.

bf16: setting ``GPTConfig.compute_dtype = jnp.bfloat16`` casts activations (and the
RoPE cos/sin cache and embeddings) to bf16 inside :meth:`GPT.__call__` while the
optimizer master state stays fp32.

Paired with ``nanochat_jax/grad_utils.py`` (``compute_loss_and_grad``) and
``nanochat_jax/optim.py`` (``step_optim`` / ``setup_optimizer_param_groups`` /
``init_optim_state`` / :class:`MuonAdamWState`).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import nnx

from nanochat_jax.gpt import GPT
from nanochat_jax.grad_utils import compute_loss_and_grad, nnx_state_to_flat_dict
from nanochat_jax.optim import (
    MuonAdamWState,
    init_optim_state,
    setup_optimizer_param_groups,
    step_optim,
)
from nanochat_jax.sharding import (
    data_parallel_sharding,
    muon_state_sharding,
    replicated_sharding,
)



# -----------------------------------------------------------------------------
# Schedule helpers (1:1 PyTorch mirrors — Python float ops, framework-agnostic)
# -----------------------------------------------------------------------------


def get_lr_multiplier(
    it: int,
    num_iterations: int,
    warmup_steps: int = 40,
    warmdown_ratio: float = 0.65,
    final_lr_frac: float = 0.05,
) -> float:
    """Linear warmup + constant + linear warmdown (1:1 mirror of
    ``base_train.py:360-369``).

    Note on an edge case: for a short run (e.g. 100 steps) with
    ``warmup=40`` and ``warmdown_ratio=0.65`` the ``warmdown_iters = 65`` so
    ``num_iter - warmdown_iters = 35 < 40 = warmup``. The warmdown branch wins
    (last ``elif``); steps 0..39 are warmup, steps 40..99 are warmdown — the
    constant region has zero length. PT semantics are preserved exactly because
    the same ``elif`` chain is followed.
    """
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_steps:
        return (it + 1) / warmup_steps
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


def get_muon_momentum(
    it: int,
    num_iterations: int,
    warmdown_ratio: float = 0.65,
) -> float:
    """Muon momentum schedule (1:1 mirror of ``base_train.py:372-382``).

    Three phases: 0..399 linear interp 0.85→0.97; 400..warmdown_start constant
    0.97; warmdown linear interp 0.97→0.90. For 100-step setup the iteration
    counter is always < 400 so the first branch dominates: momentum advances
    from ``0.85`` to ``0.85 + 99/400 * 0.12 ≈ 0.880``.
    """
    warmdown_iters = round(warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


def get_weight_decay(
    it: int,
    num_iterations: int,
    weight_decay_init: float,
) -> float:
    """Cosine decay to zero (1:1 mirror of ``base_train.py:385-386``)."""
    return weight_decay_init * 0.5 * (1.0 + math.cos(math.pi * it / num_iterations))


# -----------------------------------------------------------------------------
# Schedule-application helper: scale base param_groups by schedule values
# -----------------------------------------------------------------------------


def _scale_param_groups(
    base_groups: list[dict],
    lr_mult: jax.Array | float,
    muon_momentum: jax.Array | float,
    muon_wd: jax.Array | float,
) -> list[dict]:
    """Apply schedule values to base ``param_groups`` (the ``setup_optimizer``
    output). 1:1 mirror of ``base_train.py:523-527``::

        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd

    Returns a fresh list of group dicts (does not mutate ``base_groups``).
    Matrix params (Muon) get all three updates; non-Muon (AdamW) groups only
    get ``lr`` rescaled (PT mirror — AdamW ``betas`` / ``eps`` / ``weight_decay``
    stay at the ``setup_optimizer`` defaults).
    """
    new_groups = []
    for g in base_groups:
        new_g = {**g, "lr": g["initial_lr"] * lr_mult}
        if g["kind"] == "muon":
            new_g["momentum"] = muon_momentum
            new_g["weight_decay"] = muon_wd
        new_groups.append(new_g)
    return new_groups


# -----------------------------------------------------------------------------
# Inject helper (write a flat param dict back into the NNX model)
# -----------------------------------------------------------------------------


def _inject_jax_pytree(model: GPT, pytree: dict[str, jax.Array]) -> None:
    """Inject flat dict of JAX arrays into NNX :class:`GPT` Param leaves
    in-place.

    Walks each ``"a.b.c"`` flat key, descending into:

    - :class:`flax.nnx.List` via ``int`` index (e.g., ``transformer.h.0``)
    - :class:`flax.nnx.Dict` via ``str`` key (e.g., ``value_embeds.1``)
    - regular :class:`flax.nnx.Module` via ``getattr``

    Then assigns to the leaf :class:`flax.nnx.Param` via ``param[...] = value``
    (NNX 0.12 mutation pattern).

    This is the canonical implementation; the maintainer's golden-parity
    helpers import it from here so production code never depends on test
    code.
    """
    for key, value in pytree.items():
        parts = key.split(".")
        target = model
        for p in parts[:-1]:
            if isinstance(target, nnx.List):
                target = target[int(p)]
            elif isinstance(target, nnx.Dict):
                target = target[p]
            else:
                target = getattr(target, p)
        leaf = getattr(target, parts[-1])
        leaf[...] = value


# -----------------------------------------------------------------------------
# Init helper (model + param_groups + zero optim state)
# -----------------------------------------------------------------------------


def init_train_state(
    model: GPT,
    *,
    unembedding_lr: float = 0.004,
    embedding_lr: float = 0.2,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.0,
    scalar_lr: float = 0.5,
    polar_express_dtype: jnp.dtype | None = None,
) -> tuple[list[dict], MuonAdamWState]:
    """Build ``(param_groups, optim_state)`` for a freshly-built :class:`GPT`.

    Convenience wrapper composing :func:`setup_optimizer_param_groups` +
    :func:`init_optim_state` (zero-init AdamW + Muon buffers). Default LRs
    match ``base_train.py:308`` (``setup_optimizer`` defaults).

    Each returned group dict has an additional ``initial_lr`` field (= ``lr``
    at construction time) — mirror of PyTorch's ``torch.optim.Optimizer``
    standard convention (PT records ``initial_lr`` in ``param_groups[i]`` for
    schedule scaling; ``base_train.py:521`` reads ``group["initial_lr"]``).
    ``setup_optimizer_param_groups`` doesn't add this field (it is
    schedule-agnostic), so we add it here.

    Returns ``(param_groups, optim_state)`` ready to feed into the step
    builders below.
    """
    # Extract params dict from NNX model
    params_state = nnx.state(model, nnx.Param)
    params = nnx_state_to_flat_dict(params_state)
    # setup_optimizer_param_groups requires params dict + n_embd
    n_embd = int(model.config.n_embd)
    param_groups = setup_optimizer_param_groups(
        params, n_embd,
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
        scalar_lr=scalar_lr,
        polar_express_dtype=polar_express_dtype,
    )
    # Record initial_lr for schedule scaling (PT torch.optim standard mirror)
    for g in param_groups:
        g["initial_lr"] = g["lr"]
    optim_state = init_optim_state(param_groups, params)
    return param_groups, optim_state


# -----------------------------------------------------------------------------
# Multi-host sharding-aware train_step factory.
# -----------------------------------------------------------------------------
#
# A separate :func:`make_train_step_sharded` factory exists alongside the
# basic :func:`train_step` so that adding multi-host sharding does not change
# the single-device numeric baseline. The factory closure-captures ``mesh``
# and ``param_groups`` (so they are not JIT inputs and the non-array leaves
# like ``"kind": "adamw"`` are not traced) and returns an ``nnx.jit``-compiled
# callable with explicit ``in_shardings`` / ``out_shardings``.
#
# pytree registration: :class:`MuonAdamWState` is a plain ``@dataclass`` type,
# so we register it as a jax pytree lazily here (``_register_dataclass_pytree``)
# with an idempotent guard so re-import does not raise.

_REGISTERED_PYTREES: set[type] = set()


def _register_dataclass_pytree(cls: type, fields: tuple[str, ...]) -> None:
    """Register a dataclass as a JAX pytree (idempotent).

    Args:
        cls: dataclass type.
        fields: ordered field names whose values are children. Non-array
            metadata fields should NOT be in this list.
    """
    if cls in _REGISTERED_PYTREES:
        return

    def _flatten(obj):
        return tuple(getattr(obj, f) for f in fields), ()

    def _unflatten(_, children):
        return cls(**dict(zip(fields, children, strict=True)))

    jtu.register_pytree_node(cls, _flatten, _unflatten)
    _REGISTERED_PYTREES.add(cls)


# Module-load time (idempotent) — required for :func:`make_train_step_sharded`.
_register_dataclass_pytree(MuonAdamWState, ("step", "adamw", "muon"))


def make_train_step_sharded(
    mesh: jax.sharding.Mesh,
    param_groups: list[dict],
    optim_state: MuonAdamWState,
) -> Callable:
    """Factory: build a sharding-aware ``nnx.jit`` ed 1-step train function.

    Returns a callable with signature::

        (model, optim_state, idx, targets, lr_mult, muon_momentum, muon_wd)
            -> (loss, new_optim_state)

    matching the existing :func:`train_step` minus the ``mesh``,
    ``param_groups``, and ``optim_state`` arguments — those are captured via
    closure so that non-array metadata (``kind: str``, ``param_keys:
    list[str]``, ``MuonAdamWState`` structure) does not enter the jit tracer.

    The returned callable is decorated with ``@nnx.jit`` and applies
    sharding helpers + :func:`muon_state_sharding`:

    - ``model``: ``in_shardings=None`` (NNX replicated default; a future change
      may shard selected params without touching this signature).
    - ``optim_state``: pytree spec from :func:`muon_state_sharding(mesh,
      optim_state)` — ``state.step`` + ``state.adamw[k][...]`` replicated;
      ``state.muon[k]["momentum_buffer" | "second_momentum_buffer"]`` sharded
      along the stacked first axis (``num_params_in_group``) with
      ``P("data")``. This achieves the **ZeRO-2 variant** memory layout under
      GSPMD (the ``reduce_scatter`` / ``all_gather`` collectives PT's
      ``DistMuonAdamW`` issues by hand are auto-generated for the Muon path).
    - ``idx`` / ``targets``: ``data_parallel_sharding(mesh, batch_axis=0)``
      so the (B, T) token batch is sharded over the mesh's "data" axis.
    - 3 schedule scalars (``lr_mult`` / ``muon_momentum`` / ``muon_wd``):
      ``replicated_sharding(mesh)``.
    - ``loss`` (out): ``replicated_sharding(mesh)`` (scalar, fully replicated).
    - ``new_optim_state`` (out): same Muon-sharded spec as the input
      ``optim_state``.

    Single-device: on a 1-device CPU mesh
    XLA simplifies all sharding constraints to replicated, so the returned
    callable matches an unsharded step for ``loss`` (bit-exact) and within
    ULP for ``weights`` (~3e-8 fp32, well inside the golden
    ``weights_after_N`` comparison tolerances).
    The optimizer state stays fp32 (AdamW bit-exact, Muon within ULP).

    Multi-host : on a v6e-2/4/8 mesh the same sharding spec
    triggers the XLA partitioner to insert ``reduce_scatter`` / ``all_gather``
    collectives equivalent to PT ``DistMuonAdamW`` (optim.py:299-510).
    Each rank holds only ``ceil(num_params_in_group / N)`` Muon stacked
    buffers, dramatically reducing per-rank memory for the optimizer state.
    The sharding specs and end-to-end step behavior are pinned by the
    maintainer's local test net.

    Args:
        mesh: ``jax.sharding.Mesh`` (typically ``make_mesh`` for default
            single-device or ``make_mesh(N)`` for v6e-N). Must have at least
            one axis named ``"data"``.
        param_groups: :func:`setup_optimizer_param_groups` output (with
            ``initial_lr`` populated by :func:`init_train_state`). Captured
            by closure — must be the same object reused across calls for
            jit cache hits.
        optim_state: :func:`init_optim_state` output (zero-initialized
            ``MuonAdamWState``). Captured by closure to build the static
            sharding spec via :func:`muon_state_sharding`. Subsequent calls
            pass actual state values whose **structure** must match this
            template (same dataclass type + same dict keys).

    Returns:
        ``nnx.jit`` ed callable (cached on first call).
    """
    batch_sharding = data_parallel_sharding(mesh, batch_axis=0)
    replicated = replicated_sharding(mesh)
    optim_state_spec = muon_state_sharding(mesh, optim_state)

    @nnx.jit(
        in_shardings=(
            None, # model: NNX (replicated default)
            optim_state_spec, # optim_state: ZeRO-2 variant
            batch_sharding, # idx: (B, T) → batch sharded
            batch_sharding, # targets: (B, T) → batch sharded
            replicated, # lr_mult: scalar
            replicated, # muon_momentum: scalar
            replicated, # muon_wd: scalar
        ),
        out_shardings=(
            replicated, # loss: scalar
            optim_state_spec, # new optim_state: same ZeRO-2 variant
        ),
    )
    def _train_step_sharded(
        model: GPT,
        optim_state: MuonAdamWState,
        idx: jax.Array,
        targets: jax.Array,
        lr_mult: jax.Array,
        muon_momentum: jax.Array,
        muon_wd: jax.Array,
    ) -> tuple[jax.Array, MuonAdamWState]:
        # forward+backward -> schedule-scaled groups -> optimizer step
        # (upstream base_train.py:507-540 single micro_step). ``param_groups``
        # from closure.
        loss, grads = compute_loss_and_grad(model, idx, targets)
        params_state = nnx.state(model, nnx.Param)
        params = nnx_state_to_flat_dict(params_state)
        scaled_groups = _scale_param_groups(
            param_groups, lr_mult, muon_momentum, muon_wd
        )
        new_params, new_optim_state = step_optim(
            optim_state, grads, params, scaled_groups
        )
        _inject_jax_pytree(model, new_params)
        return loss, new_optim_state

    return _train_step_sharded


def make_grad_accum_fns(
    mesh: jax.sharding.Mesh,
    param_groups: list[dict],
    optim_state: MuonAdamWState,
) -> tuple[Callable, Callable, Callable]:
    """Factory: build memory-efficient gradient accumulation functions.

    Returns ``(compute_grads_fn, accumulate_grads_fn, apply_update_fn)`` where:

    - ``compute_grads_fn(model, idx, targets) -> (loss, grads)`` — first
      micro-step (or single-step path).
    - ``accumulate_grads_fn(model, acc_grads, idx, targets) -> (loss, new_acc_grads)``
      — subsequent micro-steps. Uses ``donate_argnums=(1,)`` so JAX reuses
      the ``acc_grads`` buffer for ``new_acc_grads`` (no double allocation).
      Eliminates Python-level ``jax.tree.map(lambda a, g: a + g, ...)`` which
      held both ``acc_grads`` and ``micro_grads`` simultaneously (~11 GB on
      a replicated d24 model).
    - ``apply_update_fn(model, optim_state, grads, lr_mult, mom, wd) -> (optim_state,)``
      — final optim step using accumulated grads.

    The caller does:

        loss, acc_grads = compute_grads_fn(model, idx, targets)
        for micro_step in range(1, grad_accum_steps):
            x, y = next(train_loader)
            loss_i, acc_grads = accumulate_grads_fn(model, acc_grads, x, y)
        # average via tree.map outside (acc_grads → /grad_accum_steps)
        optim_state = apply_update_fn(model, optim_state, avg_grads, ...)

    PT ``base_train.py:510-518`` mirror.
    """
    batch_sharding = data_parallel_sharding(mesh, batch_axis=0)
    replicated = replicated_sharding(mesh)
    optim_state_spec = muon_state_sharding(mesh, optim_state)

    @nnx.jit(
        in_shardings=(
            None, # model
            batch_sharding, # idx
            batch_sharding, # targets
        ),
        out_shardings=(
            replicated, # loss
            None, # grads (dict of arrays, replicated)
        ),
    )
    def _compute_grads(
        model: GPT,
        idx: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        loss, grads = compute_loss_and_grad(model, idx, targets)
        return loss, grads

    @nnx.jit(
        in_shardings=(
            None, # model
            None, # acc_grads (replicated, donated)
            batch_sharding, # idx
            batch_sharding, # targets
        ),
        out_shardings=(
            replicated, # loss
            None, # new_acc_grads (replicated, reuses acc_grads buffer)
        ),
        donate_argnums=(1,), # reuse acc_grads buffer for new_acc_grads
    )
    def _accumulate_grads(
        model: GPT,
        acc_grads: dict[str, jax.Array],
        idx: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        loss, grads = compute_loss_and_grad(model, idx, targets)
        new_acc = jax.tree.map(lambda a, g: a + g, acc_grads, grads)
        return loss, new_acc

    @nnx.jit(
        in_shardings=(
            None, # model
            optim_state_spec, # optim_state
            None, # accumulated grads (replicated)
            replicated, # lr_mult
            replicated, # muon_momentum
            replicated, # muon_wd
        ),
        out_shardings=optim_state_spec,
    )
    def _apply_update(
        model: GPT,
        optim_state: MuonAdamWState,
        acc_grads: dict[str, jax.Array],
        lr_mult: jax.Array,
        muon_momentum: jax.Array,
        muon_wd: jax.Array,
    ) -> MuonAdamWState:
        params_state = nnx.state(model, nnx.Param)
        params = nnx_state_to_flat_dict(params_state)
        scaled_groups = _scale_param_groups(
            param_groups, lr_mult, muon_momentum, muon_wd
        )
        new_params, new_optim_state = step_optim(
            optim_state, acc_grads, params, scaled_groups
        )
        _inject_jax_pytree(model, new_params)
        return new_optim_state

    return _compute_grads, _accumulate_grads, _apply_update


def make_fused_train_step(
    mesh: jax.sharding.Mesh,
    param_groups: list[dict],
    optim_state: MuonAdamWState,
    model: GPT,
    *,
    grad_accum_steps: int,
    donate: bool = True,
) -> tuple[Callable, "nnx.statelib.State"]:
    """Build a fused grad-accum train step.

    The returned function scans all micro-batches and runs the optimizer update
    inside one ``jax.jit`` call. ``idx_all`` and ``targets_all`` have shape
    ``(grad_accum_steps, B, T)``; axis 1 is data-sharded over ``mesh``.
    """
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    replicated = replicated_sharding(mesh)
    micro_batch_sharding = data_parallel_sharding(mesh, batch_axis=1)
    optim_state_spec = muon_state_sharding(mesh, optim_state)

    graphdef, params_state, rest_state = nnx.split(model, nnx.Param, ...)
    params_sharding = jax.tree.map(lambda _: replicated, params_state)

    def _loss_of_params(ps, idx_, targets_):
        m = nnx.merge(graphdef, ps, rest_state)
        return m(idx_, targets=targets_)

    def _fused_train_step(
        params_state,
        optim_state,
        idx_all,
        targets_all,
        lr_mult,
        muon_momentum,
        muon_wd,
    ):
        def _micro(acc_grads_state, micro):
            idx_, targets_ = micro
            loss_i, grads_state = jax.value_and_grad(_loss_of_params)(
                params_state,
                idx_,
                targets_,
            )
            return jax.tree.map(jnp.add, acc_grads_state, grads_state), loss_i

        acc0 = jax.tree.map(jnp.zeros_like, params_state)
        acc_grads_state, losses = jax.lax.scan(
            _micro,
            acc0,
            (idx_all, targets_all),
        )
        loss = jnp.mean(losses)
        inv_accum = jnp.float32(1.0 / grad_accum_steps)
        acc_grads_state = jax.tree.map(lambda g: g * inv_accum, acc_grads_state)

        params = nnx_state_to_flat_dict(params_state)
        grads = nnx_state_to_flat_dict(acc_grads_state)
        scaled_groups = _scale_param_groups(
            param_groups,
            lr_mult,
            muon_momentum,
            muon_wd,
        )
        new_params, new_optim_state = step_optim(
            optim_state,
            grads,
            params,
            scaled_groups,
        )

        m = nnx.merge(graphdef, params_state, rest_state)
        _inject_jax_pytree(m, new_params)
        _, new_params_state, _ = nnx.split(m, nnx.Param, ...)
        return new_params_state, new_optim_state, loss

    fused = jax.jit(
        _fused_train_step,
        in_shardings=(
            params_sharding,
            optim_state_spec,
            micro_batch_sharding,
            micro_batch_sharding,
            replicated,
            replicated,
            replicated,
        ),
        out_shardings=(params_sharding, optim_state_spec, replicated),
        donate_argnums=(0, 1) if donate else (),
    )
    return fused, params_state


__all__ = [
    "get_lr_multiplier",
    "get_muon_momentum",
    "get_weight_decay",
    "init_train_state",
    "make_fused_train_step",
    "make_grad_accum_fns",
    "make_train_step_sharded",
]
