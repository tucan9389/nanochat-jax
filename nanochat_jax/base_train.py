""": Single-device 100-step training loop (Phase 3 third sub-feature).

Pure-functional training loop helper that composes :func:`compute_loss_and_grad`
 with :func:`step_optim` (,
``nanochat_jax/optim.py``) into a 1-step training function and a Python-loop
multi-step runner. Mirrors PyTorch ``nanochat/scripts/base_train.py:507-540``
inner training loop (atomic 1-step ``loss = model(x,y); loss.backward;
optimizer.step; model.zero_grad``) plus the LR / Muon momentum / weight
decay schedules at lines 360-386.

Out of scope (Phase 3+ follow-ups):

- BPB calculation
- DistMuonAdamW (Phase 5, multi-device sharding)
- gradient accumulation (``grad_accum_steps>1``; Phase 5 with sharding)
- FP8 (Phase 7)
- wandb / report logging / checkpoint manager / sample generation
- Real ClimbMix dataloader (Phase 4)
- ``init_weights`` JAX rewrite (Phase 2.X follow-up)

Architecture overview:

- :class:`TrainState` — pytree-friendly dataclass holding ``step`` (scalar
  int32 0-D) + ``optim_state``.
- :func:`get_lr_multiplier` / :func:`get_muon_momentum` /
  :func:`get_weight_decay` — Python-float schedule helpers, exact 1:1 mirrors
  of PyTorch ``base_train.py:360-369`` / ``:372-382`` / ``:385-386``. Framework
  agnostic (no JAX / numpy / torch deps) so PT vs JAX is bit-exact by
  construction.
- :func:`_scale_param_groups` — apply schedule values to base ``param_groups``
  (mirror of ``base_train.py:521-525``: ``group["lr"] = group["initial_lr"] *
  lrm``; for Muon groups also ``group["momentum"]`` / ``group["weight_decay"]``).
- :func:`_inject_jax_pytree` — in-place mutation of NNX :class:`GPT` Param
  leaves from a flat dict of new values. Inlined from
  ``tests/_helpers.py:inject_jax_pytree`` to maintain ``base_train.py`` atomic
  scope (the helper will be promoted to ``nanochat_jax/_inject.py`` in Phase
  5/6 when checkpoint and inference paths share it).
- :func:`train_step` — 1-step train (``nnx.jit`` decorated). Composes
  :func:`compute_loss_and_grad` + :func:`_scale_param_groups` +
  :func:`step_optim` + :func:`_inject_jax_pytree`. Returns
  ``(loss, new_optim_state)``; mutates ``model`` in-place.
- :func:`train_loop` — Python-loop wrapper around :func:`train_step` for
  ``num_iterations`` steps. Schedules computed host-side and passed as JAX
  scalars (no recompilation between steps; JIT cache hits on calls 2..N).
- :func:`init_train_state` — convenience entry point that builds
  ``param_groups`` (``setup_optimizer_param_groups``) + zero-init optim state
  (``init_optim_state``) for a freshly-built :class:`GPT`.

Determinism :

- ``model`` initial weights are deterministic (caller's seed; see ````).
- ``compute_loss_and_grad`` is deterministic (no PRNG in forward path).
- ``step_optim`` is deterministic (no PRNG; per-group dispatch).
- Schedules are deterministic Python float ops.
- 100-step trajectory is bit-exact reproducible given the same initial state +
  batch sequence + hyperparameters.

bf16 cascade compatibility :

- ``GPTConfig.compute_dtype = jnp.bfloat16`` triggers the cascade in
  ``GPT.__call__`` (cos/sin/wte cast). measured backward absorption 0.016
  (gradient is in fp32 storage), verified AdamW state ``bit-exact 100%``
  fp32 + Muon state ULP. inherits all baselines unchanged.

References:

- ``docs//PLAN.md`` §3.3 D1-D14
- ``nanochat/scripts/base_train.py:360-386`` — schedule mirrors
- ``nanochat/scripts/base_train.py:507-540`` — inner loop mirror
- ``nanochat_jax/grad_utils.py`` — ``compute_loss_and_grad`` (paired API)
- ``nanochat_jax/optim.py`` — ``step_optim`` /
  ``setup_optimizer_param_groups`` / ``init_optim_state`` /
  :class:`MuonAdamWState` (paired API)
- ``scripts/generate_golden.py:forward_with_train_loop`` — paired PT-side capture
- ``tests/_helpers.py:inject_jax_pytree`` — promotion source (this module
  inlines the same logic to keep atomic scope; §3.3 D2 + PLAN
  intent "atomic scope preservation")
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

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
# Pytree-style training state container
# -----------------------------------------------------------------------------


@dataclass
class TrainState:
    """Pytree-friendly training state.

    Wraps the :class:`MuonAdamWState` plus a top-level ``step`` counter
    for reporting / checkpointing convenience. Matches ``base_train.py:392``
    ``step = 0`` convention. The ``optim_state.step`` field tracks the
    optimizer's bias-correction step (incremented inside :func:`step_optim`),
    while this top-level ``step`` is the loop iteration counter — both happen
    to advance in lockstep for atomic scope (no resume / no checkpoint).

    NNX :class:`GPT` model state is held externally (mutated in-place by
    :func:`train_step`); this dataclass holds only the optimizer pytree.
    """

    step: jax.Array
    """Scalar ``int32`` 0-D — top-level loop iteration counter."""

    optim_state: MuonAdamWState
    """ optimizer pytree (``step`` + ``adamw`` + ``muon`` dicts)."""


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

    Note on edge case (PLAN R7 mitigation): for the 100-step setup with
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
    """Apply schedule values to base ``param_groups`` ( ``setup_optimizer``
    output). 1:1 mirror of ``base_train.py:521-525``::

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
# Inject helper (inline mirror of tests/_helpers.py:inject_jax_pytree)
# -----------------------------------------------------------------------------


def _inject_jax_pytree(model: GPT, pytree: dict[str, jax.Array]) -> None:
    """Inject flat dict of JAX arrays into NNX :class:`GPT` Param leaves
    in-place.

    1:1 inline of :func:`tests._helpers.inject_jax_pytree`. Walks each
    ``"a.b.c"`` flat key, descending into:

    - :class:`flax.nnx.List` via ``int`` index (e.g., ``transformer.h.0``)
    - :class:`flax.nnx.Dict` via ``str`` key (e.g., ``value_embeds.1``)
    - regular :class:`flax.nnx.Module` via ``getattr``

    Then assigns to the leaf :class:`flax.nnx.Param` via ``param[...] = value``
    (NNX 0.12 mutation pattern, RESULT §6 verified).

    Atomic scope rationale : keeping the helper inlined
    here avoids a cross-module dependency between ``nanochat_jax/`` and
    ``tests/_helpers.py`` (production code must not import test helpers).
    Promotion to ``nanochat_jax/_inject.py`` is deferred to Phase 5/6 when
    checkpoint/inference paths share the same logic.
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
# 1-step train function (composes grad + optim + inject)
# -----------------------------------------------------------------------------


def train_step(
    model: GPT,
    optim_state: MuonAdamWState,
    param_groups: list[dict],
    idx: jax.Array,
    targets: jax.Array,
    lr_mult: jax.Array,
    muon_momentum: jax.Array,
    muon_wd: jax.Array,
) -> tuple[jax.Array, MuonAdamWState]:
    """1-step training: forward + backward + scaled-schedule optimizer step
    (1:1 mirror of ``base_train.py:507-540`` — single ``micro_step`` with
    ``grad_accum_steps=1``).

    Composes :func:`compute_loss_and_grad` + :func:`step_optim`
    via :func:`_scale_param_groups`, then injects the new params into the
    NNX :class:`GPT` model in-place via :func:`_inject_jax_pytree`.

    Inputs:

    - ``model``: NNX :class:`GPT` (mutated in-place — caller observes new
      weights after return).
    - ``optim_state``: previous step's :class:`MuonAdamWState` pytree.
    - ``param_groups``: :func:`setup_optimizer_param_groups` output
      (treated as static structure; only the dynamic ``lr`` /
      ``momentum`` / ``weight_decay`` values are scaled per call).
    - ``idx``, ``targets``: ``(B, T) int64`` token batches.
    - ``lr_mult``, ``muon_momentum``, ``muon_wd``: scalar fp32 ``jnp.float32``
      schedule values (computed host-side per step in :func:`train_loop`).

    Returns ``(loss, new_optim_state)``:

    - ``loss``: scalar fp32 (forward-pass loss BEFORE ``optimizer.step``,
      mirror of PT ``train_loss = loss.detach`` at ``base_train.py:512``).
    - ``new_optim_state``: post-step optimizer pytree.

    Determinism: deterministic given (model state, optim_state, idx, targets,
    schedule values). No PRNG.

    JIT note : we DO NOT decorate this with ``nnx.jit``
    here. Reason: ``nnx.jit`` traces a single graph including ``model``'s
    pytree structure; composing it with ``_inject_jax_pytree``'s Python-level
    attribute traversal requires careful structuring. For atomic 1-step
    correctness verification (PLAN scope) we run the function directly —
    ``compute_loss_and_grad`` already uses ``nnx.value_and_grad`` internally,
    which is itself jit-friendly under ``nnx.jit`` if the caller wraps. The
    100-step :func:`train_loop` calls this function in a Python ``for`` loop;
    overhead is dominated by the actual forward+backward+matmul work, not by
    Python dispatch. / Phase 5 may add ``@nnx.jit`` once the loop is
    profiled and lax.scan is considered.

    Layout convention : all flat-dict keys are
    -style "JAX format" (kernel transposed; ``transformer.h.0.attn.c_q.kernel``
    etc.). :func:`_inject_jax_pytree` walks NNX module structure using these
    same keys.
    """
    # 1. Forward + backward
    loss, grads = compute_loss_and_grad(model, idx, targets)

    # 2. Extract current params from NNX state
    params_state = nnx.state(model, nnx.Param)
    params = nnx_state_to_flat_dict(params_state)

    # 3. Apply schedule to base param_groups (base_train.py:521-525 mirror)
    scaled_groups = _scale_param_groups(
        param_groups, lr_mult, muon_momentum, muon_wd
    )

    # 4. 1-step optimizer (pure functional, returns new params + state)
    new_params, new_optim_state = step_optim(
        optim_state, grads, params, scaled_groups
    )

    # 5. Inject new params back into NNX model (in-place mutation)
    _inject_jax_pytree(model, new_params)

    return loss, new_optim_state


# -----------------------------------------------------------------------------
# 100-step train loop wrapper
# -----------------------------------------------------------------------------


def train_loop(
    model: GPT,
    optim_state: MuonAdamWState,
    param_groups: list[dict],
    batches: list[tuple[jax.Array, jax.Array]],
    *,
    num_iterations: int,
    warmup_steps: int = 40,
    warmdown_ratio: float = 0.65,
    final_lr_frac: float = 0.05,
    weight_decay_init: float = 0.01,
) -> tuple[MuonAdamWState, jax.Array]:
    """Run an ``num_iterations``-step training loop (1:1 mirror of
    ``base_train.py:416-540`` inner loop, atomic scope only).

    Iterates :func:`train_step` over ``batches[:num_iterations]``, computing
    schedule values host-side per step (Python float ops, framework-agnostic
    so PT vs JAX is bit-exact for schedules — §3.3 B.4 verified).

    Inputs:

    - ``model``: NNX :class:`GPT` (mutated in-place across all N steps).
    - ``optim_state``: initial :class:`MuonAdamWState` (typically from
      :func:`init_optim_state` — zero-init).
    - ``param_groups``: from :func:`setup_optimizer_param_groups` ;
      contains ``initial_lr`` per group used for schedule scaling.
    - ``batches``: list of ``(idx, targets)`` JAX-array pairs, length
      ``>= num_iterations``. Caller is responsible for batching deterministic
      data ( atomic scope: synthetic random tokens via
      ``np.random.default_rng(seed=42)``; §3.3 D8).
    - ``warmup_steps`` / ``warmdown_ratio`` / ``final_lr_frac``: LR schedule
      parameters (``base_train.py:362-369`` defaults).
    - ``weight_decay_init``: initial weight decay value for the cosine decay
      schedule (``base_train.py:385-386``). Default ``0.01`` — small positive
      so cosine WD decay is observable in tests ( §3.3 D7 + B.1
      finding "with wd=0 the cosine WD schedule is degenerate").

    Returns ``(new_optim_state, loss_curve)``:

    - ``new_optim_state``: post-N-th-step optimizer pytree.
    - ``loss_curve``: ``jax.Array (num_iterations,) float32`` — per-step
      pre-backward loss values, matching PT
      ``loss_curve[it] = float(loss.detach)`` at ``base_train.py:512``.

    The model is mutated through ``num_iterations`` ``train_step`` calls and
    holds the post-N-th-step weights upon return; caller can extract them via
    ``nnx.state(model, nnx.Param)`` for comparison against
    :class:`GoldenSnapshot.train_weights_after_N`.
    """
    if len(batches) < num_iterations:
        raise ValueError(
            f"train_loop: batches has {len(batches)} entries but "
            f"num_iterations={num_iterations} requested"
        )

    losses: list[jax.Array] = []
    for it in range(num_iterations):
        idx, targets = batches[it]

        # Schedule values (Python floats → JAX scalars)
        lrm = jnp.float32(
            get_lr_multiplier(
                it, num_iterations, warmup_steps, warmdown_ratio, final_lr_frac
            )
        )
        momentum = jnp.float32(
            get_muon_momentum(it, num_iterations, warmdown_ratio)
        )
        wd = jnp.float32(
            get_weight_decay(it, num_iterations, weight_decay_init)
        )

        loss, optim_state = train_step(
            model, optim_state, param_groups, idx, targets, lrm, momentum, wd,
        )
        losses.append(loss)

    loss_curve = jnp.stack(losses)
    return optim_state, loss_curve


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
    match ``base_train.py:308`` (``setup_optimizer`` defaults, cascade).

    Each returned group dict has an additional ``initial_lr`` field (= ``lr``
    at construction time) — mirror of PyTorch's ``torch.optim.Optimizer``
    standard convention (PT records ``initial_lr`` in ``param_groups[i]`` for
    schedule scaling; ``base_train.py:521`` reads ``group["initial_lr"]``).
     ``setup_optimizer_param_groups`` doesn't add this field ( doesn't
    need schedules), so we add it here.

    Returns ``(param_groups, optim_state)`` ready to feed into
    :func:`train_loop`.
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
# pytree registration: :class:`MuonAdamWState` and :class:`TrainState` are
# plain ``@dataclass`` types, so we register them as jax pytrees lazily here
# (``_register_dataclass_pytree``) with an idempotent guard so re-import does
# not raise.

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
_register_dataclass_pytree(TrainState, ("step", "optim_state"))


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

    - ``model``: ``in_shardings=None`` (NNX replicated default; cascade
      may replace selected param shardings without touching this signature).
    - ``optim_state``: pytree spec from :func:`muon_state_sharding(mesh,
      optim_state)` — ``state.step`` + ``state.adamw[k][...]`` replicated;
      ``state.muon[k]["momentum_buffer" | "second_momentum_buffer"]`` sharded
      along the stacked first axis (``num_params_in_group``) with
      ``P("data")``. This achieves the **ZeRO-2 variant** memory layout under
      GSPMD (PT ``CustomDistributedAdamW`` ``reduce_scatter`` /
      ``all_gather`` collectives auto-generated for the Muon path).
    - ``idx`` / ``targets``: ``data_parallel_sharding(mesh, batch_axis=0)``
      so the (B, T) token batch is sharded over the mesh's "data" axis.
    - 3 schedule scalars (``lr_mult`` / ``muon_momentum`` / ``muon_wd``):
      ``replicated_sharding(mesh)``.
    - ``loss`` (out): ``replicated_sharding(mesh)`` (scalar, fully replicated).
    - ``new_optim_state`` (out): same Muon-sharded spec as the input
      ``optim_state``.

    Single-device noop : on a 1-device CPU mesh
    XLA simplifies all sharding constraints to replicated, so the returned
    callable is numerically equivalent to the eager :func:`train_step` for
    ``loss`` (bit-exact) and within ULP for ``weights`` (~3e-8 fp32, well
    inside ``weights_after_N`` chaos atol of 1.0 / 1e-1 / 25.0).
    optimizer-state baseline (AdamW bit-exact 100% fp32 + Muon ULP ~3e-9) is
    preserved 100%.

    Multi-host : on a v6e-2/4/8 mesh the same sharding spec
    triggers the XLA partitioner to insert ``reduce_scatter`` / ``all_gather``
    collectives equivalent to PT ``CustomDistributedAdamW`` (optim.py:307-510).
    Each rank holds only ``ceil(num_params_in_group / N)`` Muon stacked
    buffers, dramatically reducing per-rank memory for the optimizer state.
    Functional behavior verification is done in
    :file:`tests/test_train_sharding_atomic.py` and
    :file:`tests/test_optim_state_sharding.py` ; multi-host BPB
    atol is verified in.

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
            None, # model: NNX (replicated default — )
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
        # 1:1 mirror of :func:`train_step` body. ``param_groups`` from closure.
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
      replicated d24, OOM on v5p-8).
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


__all__ = [
    "TrainState",
    "get_lr_multiplier",
    "get_muon_momentum",
    "get_weight_decay",
    "init_train_state",
    "make_grad_accum_fns",
    "make_train_step_sharded",
    "train_loop",
    "train_step",
]
