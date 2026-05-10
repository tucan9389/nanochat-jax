""" golden test generator (PyTorch side, CPU/GPU).

Generates ``tests/golden/golden_d{1,4_minimal,4_full,12}.npz`` files containing:

- **Weights** (``weight/<state_dict_key>``): PyTorch nanochat ``GPT.state_dict()``
  serialized as numpy (fp32 in this v1 — see PLAN §3.3 D9).
- **Intermediates** (``tensor/<name>``): forward intermediates captured by a
  manual mirror of ``gpt.py:411-476``, with granularity per-file (high for d1,
  block for d4_minimal/d4_full, minimal for d12 — see PLAN §3.3 D7/D9).
- **Inputs** (``input/input_ids``, ``input/targets``): ``int64`` tokens.
- ** Grads** (``grad/<state_dict_key>``): PT autograd gradients for the 3
  tagged entries (d1 + d4_sliding_gqa + d4_sliding_gqa_bf16 cascade).
- ** Weights_after** (``weights_after/<state_dict_key>``): post-1-MuonAdamW-step
  weights for the same 3 entries.
- ** Optim state** (``optim_state/<state_dict_key>/<attr>``): post-1-step
  AdamW (exp_avg, exp_avg_sq, step) + Muon (momentum_buffer, second_momentum_buffer)
  state for the same 3 entries.
- ** Train trajectory** (``train/loss_curve``, ``train/weights_after_N/...``,
  ``train/optim_state_after_N/...``): N-step (default 100) training loop loss
  per-step + final weights + final optim state for the same 3 entries.
- **``_dtype_map``**: JSON mapping ``key → str(dtype)`` for every array. Required
  because ``np.savez_compressed`` does not preserve ``ml_dtypes.bfloat16`` dtype
  labels (saves as ``|V2`` void; raw bytes survive). The loader uses this map to
  restore dtypes via ``arr.view(target_dtype)`` recommendation).
- **``_meta``**: JSON of ``{depth, config, granularity, dtype, minimal, ...}``.

Determinism (PLAN §3.3 D3):
- ``torch.manual_seed(42)`` + ``np.random.seed(42)`` per file
- ``torch.use_deterministic_algorithms(True)``
- ``NANOCHAT_DTYPE=float32`` (forced via ``os.environ`` before nanochat import)

D14 sanity (PLAN §3.3 D7 / R2 mitigation): every generated file passes a
``forward_with_intermediates(...)['logits'] == model.forward(idx)`` bit-exact
check before being written. This guards against drift in the manual mirror.

Usage::

    python scripts/generate_golden.py --out tests/golden/
    python scripts/generate_golden.py --out tests/golden/ --depths 1 4
    python scripts/generate_golden.py --out tests/golden/ --depths 12 --no-minimal

References:
- nanochat/gpt.py:411-476 — ``GPT.forward`` (mirror target)
- nanochat/gpt.py:201-261 — ``init_weights`` (called per file)
- RESULT §6.5 — ``_dtype_map`` recommendation (a)
- INVESTIGATION §3.6 — ``np.savez`` bf16 dtype loss measurement
- 02-verification-strategy.md §2 — Golden Test Set design
- docs//PLAN.md — atomic scope, V1-V5
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

# Set fp32 BEFORE importing nanochat (NANOCHAT_DTYPE detection at module load,
# nanochat/common.py:13-31). On Mac CPU the auto-detected default is also fp32,
# but explicit is reproducible across machines.
# : bf16 entries are processed after `importlib.reload` with NANOCHAT_DTYPE
# overridden to "bfloat16" (PLAN §3.3 D6-D7). See `main()`.
os.environ.setdefault("NANOCHAT_DTYPE", "float32")

import ml_dtypes # noqa: E402 — for bf16 numpy view
import numpy as np
import torch
import torch.nn.functional as F

# : PT MuonAdamW uses @torch.compile(dynamic=False, fullgraph=True) on
# muon_step_fused / adamw_step_fused. When generating multiple golden files
# with different param shapes (d1 vs d4 cascade vs bf16 vs ...), the dynamo
# recompile cache hits its default limit (~8) and raises FailOnRecompileLimitHit.
# Generation is one-shot offline, so we lift the cache to a generous bound
# and fall back to eager on errors. This affects only generation; runtime
# behavior of the fused kernels is unchanged.
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.suppress_errors = True

# Path setup for nanochat (mirrors V9 pattern, INVESTIGATION §1.4(A)).
_NANOCHAT_PATH = Path("~/projects/research/nanochat").expanduser()
if str(_NANOCHAT_PATH) not in sys.path:
    sys.path.insert(0, str(_NANOCHAT_PATH))

from nanochat.common import COMPUTE_DTYPE # noqa: E402
from nanochat.gpt import GPT, GPTConfig, norm # noqa: E402

Granularity = Literal["high", "block", "minimal"]


# -----------------------------------------------------------------------------
# Config + model construction
# -----------------------------------------------------------------------------


def make_config(depth: int) -> GPTConfig:
    """Synthetic small config (matches V1-V3 / V9 sizing).

    Differs from nanochat default (n_embd=768, vocab=32768, sequence_len=2048,
    window_pattern="SSSL"): we use n_embd=64, vocab=100 (padded to 128),
    sequence_len=128, window_pattern="L". Used for d1 / d4_minimal / d4_full / d12.
    Sliding-window-aware variant lives in :func:`make_sliding_config` .
    """
    return GPTConfig(
        sequence_len=128,
        vocab_size=100,
        n_layer=depth,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="L",
    )


def make_sliding_config(depth: int) -> GPTConfig:
    """ sliding-window-aware small config (PLAN §3.3 D1, D7).

    Two changes from :func:`make_config` to make sliding window verification
    meaningful (INVESTIGATION §1, §3.1):

    1. ``sequence_len = 512`` (was 128). Original short_window公式
       ``ceil(seq/4/128)*128`` gives ``128`` for seq=512 (vs ``128 == long`` for
       seq=128, which auto-disables sliding) — function "gotcha ①" resolved.
    2. ``window_pattern = "SSSL"`` (was "L"). 4-char pattern fits ``n_layer=4``
       exactly: layers 0/1/2 short(128), layer 3 long(512, also forced final).

    Function "gotcha ②" (input ``seq_len < short_window`` → PT
    ``_sdpa_attention`` branch 1 falls through) caller ``input_seq_len > 128``
    so resolved (``_GOLDEN_SPEC`` 5-tuple last column).
    """
    return GPTConfig(
        sequence_len=512,
        vocab_size=100,
        n_layer=depth,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        window_pattern="SSSL",
    )


def make_gqa_config(depth: int) -> GPTConfig:
    """ Step 1 GQA-aware small config (PLAN §3.3 D1, D7).

    Differs from :func:`make_config` only in ``n_kv_head < n_head``:
    - ``n_head=4, n_kv_head=2`` (n_rep=2) — minimal GQA broadcast verification.
    - ``sequence_len=128, "L", n_embd=64, vocab=100`` — same as ``d4_minimal`` otherwise.

    PT path: ``F.scaled_dot_product_attention(..., enable_gqa=True)`` — auto-detected
    by ``flash_attention.py:126`` (``enable_gqa = q.size(1) != k.size(1)``).
    JAX path: ``jnp.repeat(k, n_rep, axis=2)`` — explicit broadcast
    .
    """
    return GPTConfig(
        sequence_len=128,
        vocab_size=100,
        n_layer=depth,
        n_head=4,
        n_kv_head=2, # ⭐ Step 1: GQA broadcast (n_rep=2)
        n_embd=64,
        window_pattern="L",
    )


def make_sliding_gqa_config(depth: int) -> GPTConfig:
    """ Step 2 cascade config (sliding + GQA, PLAN §3.3 D1, D7, D10).

    Combines :func:`make_sliding_config` (sequence_len=512, "SSSL") with
    :func:`make_gqa_config` (n_kv_head=2). sliding mask + GQA
    broadcast combined validation — Phase 2 end integrated (d12 real) core invariant
    multi-feature cascade.

     INVESTIGATION §1.2 gotcha ② avoidance: input ``seq_len > short_window``
    required (caller ``_GOLDEN_SPEC`` ``input_seq_len=192`` specified).
    """
    return GPTConfig(
        sequence_len=512,
        vocab_size=100,
        n_layer=depth,
        n_head=4,
        n_kv_head=2, # ⭐ Step 2: GQA broadcast (n_rep=2)
        n_embd=64,
        window_pattern="SSSL", # ⭐ sliding pattern
    )


# -----------------------------------------------------------------------------
# bf16 cascade configs (PLAN §3.3 D4-D5)
#
# PT side cascade enabled ``NANOCHAT_DTYPE=bfloat16`` env var + ``importlib.reload``
# module-level ``COMPUTE_DTYPE`` is changed (nanochat/common.py:13-31). PT GPTConfig
# selfhas dtype field none — module variable ``GPT.__init__`` ``wte/value_embeds.to(
# dtype=COMPUTE_DTYPE)`` (gpt.py:255-261)and ``cos.to(COMPUTE_DTYPE)`` (gpt.py:276)in
# auto cast triggered. therefore this two ``make_*_bf16_config`` fp32 paired thin alias
# (label differs but PT GPTConfig identical — readability + grep-friendly).
#
# NNX side ``GPTConfig.compute_dtype: jnp.dtype = jnp.float32`` field ( new,
# nanochat_jax/gpt.py PLAN §3.3 D1) explicit. NNX-side test cfg when creating
# ``compute_dtype=jnp.bfloat16`` specified.
# -----------------------------------------------------------------------------


def make_bf16_config(depth: int) -> GPTConfig:
    """ Step 1 bf16 standalone config (PLAN §3.3 D4).

    Differs from :func:`make_config` only in PT cascade dtype (set externally
    via ``NANOCHAT_DTYPE=bfloat16`` env + ``importlib.reload``). MHA full causal
     — bf16 standalone effect isolation.

    PT cascade enabled: ``NANOCHAT_DTYPE=bfloat16`` env → ``COMPUTE_DTYPE=torch.bfloat16``
    → ``GPT.__init__`` ``wte/value_embeds.to(dtype=COMPUTE_DTYPE)`` (gpt.py:255-261)
    + ``cos.to(COMPUTE_DTYPE)`` (gpt.py:276) + ``x = x.to(COMPUTE_DTYPE)`` (gpt.py:424).

    NNX-side testin ``cfg.compute_dtype=jnp.bfloat16`` explicit (NNX GPTConfig field).
    """
    return GPTConfig(
        sequence_len=128,
        vocab_size=100,
        n_layer=depth,
        n_head=4,
        n_kv_head=4, # MHA
        n_embd=64,
        window_pattern="L", # full causal
    )


def make_sliding_gqa_bf16_config(depth: int) -> GPTConfig:
    """ Step 2 cascade config (sliding + GQA + bf16, PLAN §3.3 D5).

    Combines :func:`make_sliding_gqa_config` (sequence_len=512, "SSSL", n_kv_head=2)
    with bf16 cascade. **Phase 2 end integrated d12 real config enter before cascade validation**:
     sliding + GQA + bf16 all sub-feature combined.

     cascade ULP equivalent found (Step 1 = Step 2 fp32 max abs diff = 1.43e-6) bf16
    in preserved if: Step 1 (`d4_bf16_minimal`) max ≈ Step 2 max → d12 integrated risk
    -50% decrease.

     gotcha ② avoidance: input ``seq_len > short_window`` (192 = 1.5×128).
    """
    return GPTConfig(
        sequence_len=512,
        vocab_size=100,
        n_layer=depth,
        n_head=4,
        n_kv_head=2, # ⭐ GQA
        n_embd=64,
        window_pattern="SSSL", # ⭐ sliding
    )


def make_d12_real_config(depth: int = 12) -> GPTConfig:
    """I-02 Phase 2 end integrated d12 real config (PLAN §3.3 D1).

    PT default mirror (nanochat/gpt.py:29-39 GPTConfig defaults directly grep validation):

    - sequence_len=2048
    - vocab_size=32768 (full real, .gitignore + generate-on-demand)
    - n_layer=12
    - n_head=6, n_kv_head=6 (MHA, n_head==n_kv_head)
    - n_embd=768
    - window_pattern="SSSL"

    PT cascade enabled: ``NANOCHAT_DTYPE=bfloat16`` env + ``importlib.reload`` after
    call → ``COMPUTE_DTYPE=torch.bfloat16`` ``GPT.__init__`` ``wte/value_embeds.to(
    dtype=COMPUTE_DTYPE)`` (gpt.py:255-261) + ``cos.to(COMPUTE_DTYPE)`` (gpt.py:276)
    + ``x = x.to(COMPUTE_DTYPE)`` (gpt.py:424)in auto cast triggered.

    NNX-side testin ``cfg.compute_dtype=jnp.bfloat16`` explicit (NNX GPTConfig field).
    ``_helpers.py`` ``build_gpt_from_snapshot`` ``snap.meta["dtype"]`` auto
    detection .

     gotcha ② avoidance: input ``seq_len > short_window``. d12 realin
    ``short_window = ceil(2048/4/128)*128 = 512`` → caller
    ``input_seq_len=768`` (1.5×short, / pattern) specified.

    Note (.npz size): vocab=32768 + n_embd=768 + n_layer=12 → weights ~756MB
    (bf16 wte/value_embeds 336MB + fp32 block body/lm_head 420MB) + intermediates
    ~244MB (logits 200MB fp32 + body bf16 44MB) → total ~1GB raw, ~500MB compressed.
    .gitignore (PLAN Q-01 (B) decision).
    """
    return GPTConfig(
        sequence_len=2048,
        vocab_size=32768, # PT default precise mirror
        n_layer=depth,
        n_head=6,
        n_kv_head=6, # MHA (n_head==n_kv_head, PT default)
        n_embd=768,
        window_pattern="SSSL", # ⭐ sliding
    )


def zero_extra_features(model: GPT) -> None:
    """Zero VE / smear / backout / x0 effects (PLAN §3.3 D8 — Q-02 (A)).

    Sets four parameters to zero after ``init_weights()``:
    - ``smear_lambda`` → smear additive contribution = 0 (gpt.py:431)
    - ``backout_lambda`` → backout subtract = 0 (gpt.py:459)
    - ``x0_lambdas`` → x0 blending = 0 (gpt.py:452)
    - ``value_embeds[i].weight`` → VE contribution = 0 (gpt.py:95: ``gate * ve = gate * 0 = 0``)

    GPT code is NOT modified. ``ve_gate.weight`` keeps its small-uniform init;
    it's irrelevant when ``value_embeds`` is zero. This satisfies the
    02-verification-strategy.md §3 Phase 3 Step 0 intent ("d1, no VE/smear/backout").
    """
    with torch.no_grad():
        model.smear_lambda.zero_()
        model.backout_lambda.zero_()
        model.x0_lambdas.zero_()
        for ve in model.value_embeds.values():
            ve.weight.zero_()


# -----------------------------------------------------------------------------
# Manual forward mirror (PLAN §3.3 D7) — gpt.py:411-476
# -----------------------------------------------------------------------------


def forward_with_intermediates(
    model: GPT,
    idx: torch.Tensor,
    targets: torch.Tensor | None = None,
    *,
    granularity: Granularity = "block",
) -> dict[str, torch.Tensor]:
    """Manual mirror of ``GPT.forward`` (gpt.py:411-476) capturing intermediates.

    Calls the same submodules in the same order as ``GPT.forward`` (no
    behavioral change); the only delta is recording named intermediate
    tensors per the requested granularity:

    - ``"minimal"``: ``x_after_wte``, ``logits``, ``loss`` (used for d12)
    - ``"block"``: minimal + ``x_after_smear``, ``x0``, ``block_{0..n-1}_out``,
      ``x_after_backout``, ``x_final_norm`` (used for d4_minimal / d4_full)
    - ``"high"``: block + ``x_after_norm0``, ``block_{0..n-1}_input``,
      ``logits_pre_softcap`` (used for d1)

    The caller is responsible for the D14 mirror-parity sanity check
    (``forward_with_intermediates(...)['logits'] == model.forward(idx)``
    bit-exactly) before writing the .npz.

    Assumes ``T > 1`` (training path; matches gpt.py:430 assertion) and
    ``kv_cache is None`` (gpt.py:428 branch).
    """
    cfg = model.config
    ints: dict[str, torch.Tensor] = {}
    B, T = idx.size()
    assert T > 1, " generator expects T > 1 (training path; gpt.py:430)"

    # Rotary slices (gpt.py:420)
    cos_sin = (model.cos[:, :T], model.sin[:, :T])

    # 1. wte + cast (gpt.py:423-424)
    x = model.transformer.wte(idx)
    x = x.to(COMPUTE_DTYPE)
    ints["x_after_wte"] = x.detach().clone()

    # 2. First norm (gpt.py:425)
    x = norm(x)
    if granularity == "high":
        ints["x_after_norm0"] = x.detach().clone()

    # 3. Smear (gpt.py:430-432, training path with T > 1)
    gate = model.smear_lambda.to(x.dtype) * torch.sigmoid(
        model.smear_gate(x[:, 1:, :24])
    )
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
    if granularity in ("high", "block"):
        ints["x_after_smear"] = x.detach().clone()

    # 4. Save x0 (gpt.py:447)
    x0 = x
    if granularity in ("high", "block"):
        ints["x0"] = x0.detach().clone()

    # 5. Trunk loop (gpt.py:448-456)
    n_layer = cfg.n_layer
    backout_layer = n_layer // 2
    x_backout: torch.Tensor | None = None
    for i, block in enumerate(model.transformer.h):
        x_block_in = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        if granularity == "high":
            ints[f"block_{i}_input"] = x_block_in.detach().clone()
        ve = (
            model.value_embeds[str(i)](idx).to(x.dtype)
            if str(i) in model.value_embeds
            else None
        )
        x = block(x_block_in, ve, cos_sin, model.window_sizes[i], None)
        if granularity in ("high", "block"):
            ints[f"block_{i}_out"] = x.detach().clone()
        if i == backout_layer:
            x_backout = x

    # 6. Backout subtract (gpt.py:458-459)
    if x_backout is not None:
        x = x - model.backout_lambda.to(x.dtype) * x_backout
    if granularity in ("high", "block"):
        ints["x_after_backout"] = x.detach().clone()

    # 7. Final norm (gpt.py:460)
    x = norm(x)
    if granularity in ("high", "block"):
        ints["x_final_norm"] = x.detach().clone()

    # 8. lm_head + softcap (gpt.py:463-467)
    softcap_v = 15
    logits = model.lm_head(x)
    logits = logits[..., : cfg.vocab_size]
    logits = logits.float()
    if granularity == "high":
        ints["logits_pre_softcap"] = logits.detach().clone()
    logits = softcap_v * torch.tanh(logits / softcap_v)
    ints["logits"] = logits.detach().clone()

    # 9. Loss (gpt.py:469-473)
    if targets is not None:
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="mean",
        )
        ints["loss"] = loss.detach().clone()

    return ints


# -----------------------------------------------------------------------------
# : PT-side gradient capture
# -----------------------------------------------------------------------------


def forward_with_grad(
    model: GPT,
    idx: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """: capture per-parameter gradients via PT autograd.

    Mirror of ``base_train.py:511-517`` ``loss = model(x, y); loss.backward()``,
    factored into a reusable helper. Returns a dict of ``{state_dict_key: grad}``
    cloned tensors (caller can detach + numpy-convert as needed).

    Important: this function MUST NOT be wrapped in ``torch.no_grad()`` (that
    would disable autograd). Caller must NOT pre-call ``torch.no_grad`` either.

    bf16 cascade compatibility: PT autograd preserves param dtype on grad
    ( follow-up M3 measurement: wte/value_embeds bf16 → grad bf16,
    Linear/scalar fp32 → grad fp32). Our ``_torch_to_numpy`` handles bf16 via
    uint16 view — dtype preserved bit-exact in .npz.

    Determinism: caller's ``torch.manual_seed(42)`` + ``torch.use_deterministic_algorithms(True)``
    apply (set in :func:`generate_one`). Reproduces bit-exact across runs.
    """
    model.zero_grad()
    loss = model(idx, targets)
    loss.backward()
    grads: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        # detach + clone so caller can mutate model state freely afterward
        grads[name] = p.grad.detach().clone()
    return grads


# -----------------------------------------------------------------------------
# : PT-side 1-step MuonAdamW capture
# -----------------------------------------------------------------------------


def forward_with_optim_step(
    model: GPT,
    idx: torch.Tensor,
    targets: torch.Tensor,
    *,
    optim_kwargs: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """: capture per-parameter weights + optimizer state after 1 MuonAdamW step.

    Mirror of ``base_train.py:511-525``::

        loss = model(x, y)
        loss.backward()
        optimizer.step()

    Returns ``(weights_after, optim_state)``:

    - ``weights_after``: ``{state_dict_key: weight_tensor}`` — all named params
      after 1 step (PT layout, fp32 / bf16 cascade dtype preserved).
    - ``optim_state``: ``{state_dict_key/attr_name: state_tensor}`` flat dict.
        - AdamW groups: per-param ``{exp_avg, exp_avg_sq, step}`` (PT optim.py:208-213).
        - Muon groups: group-level ``{momentum_buffer, second_momentum_buffer}``
          stored under the FIRST param's key (PT optim.py:241-254).
        - ``step`` (int counter) is stored as ``int64`` 0-D tensor for
          uniform .npz handling.

    Important caveats:
    - Mutates ``model`` (via ``optimizer.step()``). Caller must NOT reuse the
      same model afterwards. ``generate_one`` calls this AFTER capturing
      weights/intermediates/grads and BEFORE writing the .npz.
    - MUST NOT be wrapped in ``torch.no_grad()`` (would disable autograd).
    - bf16 cascade compatibility: PT autograd preserves param dtype on grad
      (M3 measurement ); optim state buffers are ``torch.zeros_like(p)``
      (PT optim.py:210-211, 248) so dtype = param dtype mixed (bf16 wte +
      fp32 Linear/scalars).

    Determinism: caller's ``torch.manual_seed(42)`` +
    ``torch.use_deterministic_algorithms(True)`` apply (set in
    :func:`generate_one`). PT MuonAdamW + Polar Express + NorMuon all
    deterministic given (model state, idx, targets, hyperparams).

    Hyperparams (defaults match ``base_train.py:308`` / ``gpt.py:369-409``):
    ``unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
    weight_decay=0.0, scalar_lr=0.5``. Use ``optim_kwargs`` to override.
    """
    # zero grad + forward + backward
    model.zero_grad()
    loss = model(idx, targets)
    loss.backward()

    # Setup optimizer with defaults (base_train.py:308 mirror)
    kw = optim_kwargs or {}
    optimizer = model.setup_optimizer(
        unembedding_lr=kw.get("unembedding_lr", 0.004),
        embedding_lr=kw.get("embedding_lr", 0.2),
        matrix_lr=kw.get("matrix_lr", 0.02),
        weight_decay=kw.get("weight_decay", 0.0),
        scalar_lr=kw.get("scalar_lr", 0.5),
    )
    optimizer.step()

    # Capture weights_after (post-step)
    weights_after: dict[str, torch.Tensor] = {
        n: p.detach().clone() for n, p in model.named_parameters()
    }

    # Capture optim_state (flat dict keyed by `<state_dict_key>/<attr>`)
    # PT optim.py:285-293 — opt.state[p] is a dict per param. AdamW populates
    # state for every param in group; Muon populates state only on first param
    # of group (group-level buffers). Both patterns yield clean flat keys.
    optim_state: dict[str, torch.Tensor] = {}
    name_by_param = {id(p): n for n, p in model.named_parameters()}
    for p, state in optimizer.state.items():
        name = name_by_param.get(id(p))
        if name is None:
            continue # defensive (should not happen for properly registered params)
        for attr_name, attr_val in state.items():
            if isinstance(attr_val, torch.Tensor):
                optim_state[f"{name}/{attr_name}"] = attr_val.detach().clone()
            elif attr_name == "step":
                # int counter → int64 0-D tensor for uniform .npz storage
                optim_state[f"{name}/{attr_name}"] = torch.tensor(
                    attr_val, dtype=torch.int64
                )
            # else: skip (None or unsupported types)

    return weights_after, optim_state


# -----------------------------------------------------------------------------
# : PT-side N-step train loop capture
# -----------------------------------------------------------------------------


def _pt_get_lr_multiplier(
    it: int,
    num_iterations: int,
    warmup_steps: int = 40,
    warmdown_ratio: float = 0.65,
    final_lr_frac: float = 0.05,
) -> float:
    """Linear warmup + constant + linear warmdown (base_train.py:360-369 1:1 mirror).

    Note: For 100-step setup (warmup=40, warmdown_ratio=0.65), warmdown_iters=65
    so num_iter - warmdown_iters = 35 < 40 = warmup. The warmdown branch wins
    (PLAN R7 mitigation): step 0-39 warmup, step 40-99 warmdown (constant region empty).
    """
    import math # local to keep top-level imports stable
    _ = math # silence unused (math used below in caller path)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_steps:
        return (it + 1) / warmup_steps
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


def _pt_get_muon_momentum(
    it: int,
    num_iterations: int,
    warmdown_ratio: float = 0.65,
) -> float:
    """Muon momentum schedule 0.85 → 0.97 → 0.90 (base_train.py:372-382 1:1 mirror).

    For 100-step setup, it < 400 always (100 < 400) → first branch only,
    momentum interpolates 0.85 → 0.85 + 99/400 × 0.12 ≈ 0.880.
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


def _pt_get_weight_decay(
    it: int,
    num_iterations: int,
    weight_decay_init: float,
) -> float:
    """Cosine decay to zero (base_train.py:385-386 1:1 mirror)."""
    import math
    return weight_decay_init * 0.5 * (1.0 + math.cos(math.pi * it / num_iterations))


def forward_with_train_loop(
    model: GPT,
    idx_first: torch.Tensor,
    targets_first: torch.Tensor,
    *,
    num_iterations: int = 100,
    batch_seed: int = 42,
    optim_kwargs: dict[str, float] | None = None,
    schedule_kwargs: dict[str, float] | None = None,
) -> tuple[
    np.ndarray,
    list[dict[str, torch.Tensor]],
    list[dict[str, torch.Tensor]],
    list[dict[str, torch.Tensor]],
]:
    """: capture per-step training trajectory (grad, weights, optim state for ALL N steps).

    Mirror of ``base_train.py:507-540`` inner training loop (atomic 1-step
    forward + backward + schedule update + optimizer.step + zero_grad).
    FP8/wandb/eval/sample/checkpoint/DDP/grad_accum_steps>1 branch all excluded — atomic scope.

    **Per-step capture for strict algorithmic verification ( redesign,
    chaos divergence finding)**:

    Continuous PT-vs-JAX 100-step exact match is impossible due to chaotic
    dynamics (random data + ULP differences in matmul reduction order between
    cuBLAS (PT) and XLA (JAX) amplify exponentially through forward/backward
    chains). Instead we capture per-step (grad, out_weights, out_optim_state)
    so JAX-side tests can verify each step's algorithmic correctness
    INDEPENDENTLY by anchoring at the PT-stored input state — bypassing
    chaotic accumulation. This delivers strict atol on every
    one of the 100 steps.

    Inputs:
    - ``idx_first``, ``targets_first``: first batch (typically the same as
      ``input/input_ids``/``input/targets`` so step 0 forward overlaps the /
      1-step baseline for sanity checks).
    - ``num_iterations``: number of training steps (PLAN Q-03 (A) default 100).
    - ``batch_seed``: separate ``np.random.default_rng`` seed for the N-1
      remaining batches (deterministic; see PLAN D8/D13).
    - ``optim_kwargs``: forwarded to ``model.setup_optimizer`` (defaults match
      ``base_train.py:308`` / ). NOTE: ``weight_decay`` here is the
      ``weight_decay_init`` passed into the cosine schedule (per PLAN D7) —
      the optimizer's per-step ``group["weight_decay"]`` is OVERWRITTEN every
      iteration via the schedule, so the initial value passed to setup_optimizer
      acts as the schedule init.
    - ``schedule_kwargs``: ``warmup_steps`` (default 40), ``warmdown_ratio``
      (default 0.65), ``final_lr_frac`` (default 0.05), ``weight_decay_init``
      (default 0.01 — small positive so cosine WD decay is observable; PLAN
      D7 + B.1 finding "wd=0if cosine wd schedule PT vs JAX bit-exact validation
      meaning is weak").

    Returns:
    - ``loss_curve``: ``np.ndarray (num_iterations,)`` fp32 (each step's
      pre-backward loss, ``loss.detach().item()`` mirror — base_train.py:512).
    - ``per_step_grads``: list of length ``num_iterations``. Each entry is
      ``{state_dict_key: grad_tensor}`` — gradients computed at the START of
      step N (BEFORE optimizer step), captured via PT autograd. Used by tests
      to feed PT-stored grad into JAX optim for strict isolation.
    - ``per_step_weights_after``: list of length ``num_iterations``. Each entry
      is ``{state_dict_key: weight_tensor}`` — params AFTER step N's
      ``optimizer.step()``. ``per_step_weights_after[N]`` = input weights for
      step N+1. ``per_step_weights_after[-1]`` = final weights after all N steps.
    - ``per_step_optim_state_after``: list of length ``num_iterations``. Each
      entry is ``{state_dict_key/attr: state_tensor}`` — optim state AFTER
      step N (AdamW exp_avg/exp_avg_sq/step + Muon momentum_buffer/
      second_momentum_buffer, same flat-key convention as
      :func:`forward_with_optim_step` D3).

    Important caveats:
    - Mutates ``model`` (via ``optimizer.step()`` × N). Caller must NOT reuse the
      same model afterwards. ``generate_one`` calls this AFTER capturing
      weights/intermediates/grads/-1-step BEFORE writing the .npz.
    - MUST NOT be wrapped in ``torch.no_grad()`` (would disable autograd).
    - bf16 cascade compatibility: identical to (param dtype preserved
      through grad / state buffers; PT autograd standard).

    Determinism: caller's ``torch.manual_seed(seed)`` +
    ``torch.use_deterministic_algorithms(True)`` apply (set in
    :func:`generate_one`). Per-step PT MuonAdamW + Polar Express + NorMuon
    + schedule values all deterministic given (model state, batches, hyperparams).

    Cost (per .npz):
    - d1 fp32 100-step: ~30-60s (Mac arm64 CPU)
    - d4 sliding+GQA fp32 100-step: ~2-3min
    - d4 sliding+GQA bf16 100-step: ~3-5min (bf16 promote cascade)
    """
    import math
    _ = math # silence unused import (used inside _pt_get_*)

    optim_kw = dict(optim_kwargs or {})
    sched_kw = dict(schedule_kwargs or {})
    warmup_steps = int(sched_kw.get("warmup_steps", 40))
    warmdown_ratio = float(sched_kw.get("warmdown_ratio", 0.65))
    final_lr_frac = float(sched_kw.get("final_lr_frac", 0.05))
    weight_decay_init = float(sched_kw.get("weight_decay_init", 0.01))

    # Setup optimizer (base_train.py:308 mirror; same defaults as D2 path)
    # NOTE: pass weight_decay_init as the initial WD so PT's setup_optimizer
    # records `initial_lr` correctly. The per-step group["weight_decay"] is
    # then overwritten every iteration via schedule (base_train.py:527).
    optimizer = model.setup_optimizer(
        unembedding_lr=optim_kw.get("unembedding_lr", 0.004),
        embedding_lr=optim_kw.get("embedding_lr", 0.2),
        matrix_lr=optim_kw.get("matrix_lr", 0.02),
        weight_decay=optim_kw.get("weight_decay", weight_decay_init),
        scalar_lr=optim_kw.get("scalar_lr", 0.5),
    )

    # Build N batches: first one = caller's (idx_first, targets_first) for
    # / sanity overlap; remaining N-1 from default_rng(batch_seed)
    # (B.5 verified deterministic).
    cfg = model.config
    B, T = idx_first.shape
    rng = np.random.default_rng(batch_seed)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = [(idx_first, targets_first)]
    for _ in range(num_iterations - 1):
        idx_np = rng.integers(0, cfg.vocab_size, size=(B, T), dtype=np.int64)
        tgt_np = rng.integers(0, cfg.vocab_size, size=(B, T), dtype=np.int64)
        # Mask a fraction with ignore_index=-1 (random tokens that hit value 0
        # become -1 — exercises the ignore_index path, which / also
        # exercise for the first batch).
        tgt_np[tgt_np == 0] = -1
        batches.append((torch.from_numpy(idx_np), torch.from_numpy(tgt_np)))

    # Run the N-step training loop (base_train.py:507-540 mirror) with per-step capture
    loss_curve = np.zeros(num_iterations, dtype=np.float32)
    per_step_grads: list[dict[str, torch.Tensor]] = []
    per_step_weights_after: list[dict[str, torch.Tensor]] = []
    per_step_optim_state_after: list[dict[str, torch.Tensor]] = []
    name_by_param = {id(p): n for n, p in model.named_parameters()}

    for it, (idx, tgts) in enumerate(batches):
        # Forward + backward (base_train.py:511-517)
        model.zero_grad(set_to_none=True)
        loss = model(idx, tgts)
        loss_curve[it] = float(loss.detach())
        loss.backward()

        # Capture grad BEFORE optim step
        grads_step: dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            grads_step[n] = p.grad.detach().clone()
        per_step_grads.append(grads_step)

        # Schedule values (base_train.py:520-527 mirror — Python float ops)
        lrm = _pt_get_lr_multiplier(
            it, num_iterations, warmup_steps, warmdown_ratio, final_lr_frac,
        )
        muon_momentum = _pt_get_muon_momentum(it, num_iterations, warmdown_ratio)
        muon_wd = _pt_get_weight_decay(it, num_iterations, weight_decay_init)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd

        optimizer.step()

        # Capture weights AFTER optim step (clone — model is mutated in-place
        # by optimizer.step() and by next iteration's backward, so we MUST
        # clone here; same view-aliasing concern as forward_with_optim_step
        # §3.3 D2)
        weights_after_step: dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters()
        }
        per_step_weights_after.append(weights_after_step)

        # Capture optim_state AFTER optim step (same flat-key convention as )
        optim_state_step: dict[str, torch.Tensor] = {}
        for p, state in optimizer.state.items():
            name = name_by_param.get(id(p))
            if name is None:
                continue
            for attr_name, attr_val in state.items():
                if isinstance(attr_val, torch.Tensor):
                    optim_state_step[f"{name}/{attr_name}"] = attr_val.detach().clone()
                elif attr_name == "step":
                    optim_state_step[f"{name}/{attr_name}"] = torch.tensor(
                        attr_val, dtype=torch.int64
                    )
        per_step_optim_state_after.append(optim_state_step)

    return loss_curve, per_step_grads, per_step_weights_after, per_step_optim_state_after


# -----------------------------------------------------------------------------
# : BPB evaluation capture
# -----------------------------------------------------------------------------


def forward_with_bpb_eval(
    model: GPT,
    idx_first: torch.Tensor,
    targets_first: torch.Tensor,
    *,
    eval_steps: int = 5,
    batch_seed: int = 42,
    slow_path_at_step: int = 4,
    token_bytes_max: int = 8,
) -> dict[str, np.ndarray]:
    """: capture BPB evaluation trajectory — PT
    ``nanochat/loss_eval.py:evaluate_bpb`` (65 LOC) end-to-end snapshot.

    Mirrors ``loss_eval.py:9-65``: per-step ``loss_2d = model(x, y,
    loss_reduction='none').view(-1)`` + ``num_bytes2d = token_bytes[y]``
    weighting + ``total_nats / (log(2) × total_bytes)``.

    eval-only forward (no autograd, no optim mutation) → does NOT mutate
    ``model``. Caller may run this BEFORE ``forward_with_optim_step`` /
    ``forward_with_train_loop`` (which DO mutate) without rebuilding.

    Inputs:

    - ``idx_first``, ``targets_first``: first batch (typically the same
      ``input/input_ids`` / ``input/targets`` so step 0 forward matches /
       1-step baseline for sanity overlap).
    - ``eval_steps``: number of eval batches .
    - ``batch_seed``: ``np.random.default_rng`` seed for the remaining
      ``eval_steps - 1`` batches and for ``token_bytes`` .
    - ``slow_path_at_step``: index of the batch where ``targets`` are partially
      set to ``-1`` (ignore_index) — exercises ``evaluate_bpb`` slow path
      (PT ``loss_eval.py:18-32``). §3.3 D10 (default 4: last batch).
    - ``token_bytes_max``: upper bound (exclusive) for dummy ``token_bytes``
      generation (``rng.integers(0, token_bytes_max, vocab_size)``). PLAN
      0018 §3.3 D9 default 8 (12.5% sparse 0 → ``num_bytes2d > 0`` mask sanity).

    Returns dict with these arrays :

    - ``input_ids``: ``(eval_steps, B, T)`` int64 — eval batch input_ids
    - ``targets``: ``(eval_steps, B, T)`` int64 — eval batch targets (with
      ``-1`` at ``slow_path_at_step`` per D10)
    - ``token_bytes``: ``(vocab_size,)`` int32 — dummy dataset (D9)
    - ``per_step_loss_2d``: ``(eval_steps, B*T)`` fp32 — PT ``loss2d.view(-1)``
      after each ``model(x, y, loss_reduction='none')``
    - ``total_nats``: ``()`` fp32 — accumulated nats (PT line 38/45)
    - ``total_bytes``: ``()`` int64 — accumulated bytes (PT line 39/46;
      stored as int64 for PT-mirror exact byte-count even though
      ``evaluate_bpb`` JAX side uses int32 — atomic test scale always fits)
    - ``expected_bpb``: ``()`` fp32 — final BPB scalar (PT line 64)

    Notes
    -----
    - Each returned array is already a numpy ndarray (suitable for direct
      ``arrays[f"bpb_eval/{k}"] = ...`` in ``generate_one``).
    - ``loss_2d`` is computed in fp32 (PT ``F.cross_entropy(reduction='none')``
      promotes to fp32 even with bf16 logits because ``logits = logits.float()``
      at gpt.py:466 before cross_entropy — same path as the existing /
      forward that produces ``tensor/loss``).
    - Determinism: caller's ``torch.manual_seed(seed)`` +
      ``torch.use_deterministic_algorithms(True)`` apply (set in
      ``generate_one``). Plus our local ``np.random.default_rng(batch_seed)``
      for reproducible eval batches + token_bytes across runs.
    """
    cfg = model.config
    B, T = idx_first.shape
    vocab_size = cfg.vocab_size

    # §3.3 D9: dummy token_bytes (Phase 4 tokenizer dep avoid).
    # rng.integers(0, max=8) → values in [0, 7]. ~12.5% will be 0 (special
    # token simulation, sanity for `(num_bytes2d > 0)` mask). Stored as int32
    # (D4 atomic; no overflow risk at test scale).
    rng = np.random.default_rng(batch_seed)
    token_bytes_np = rng.integers(
        0, token_bytes_max, size=(vocab_size,), dtype=np.int32
    )
    token_bytes_pt = torch.from_numpy(token_bytes_np.astype(np.int64))

    # Build eval_steps batches:
    # batch 0 = (idx_first, targets_first) — overlap with / baseline
    # batch 1.. = rng.integers(0, vocab_size) for input_ids/targets
    # batch[slow_path_at_step] gets ~25% of positions overwritten with -1
    eval_input_ids: list[np.ndarray] = []
    eval_targets: list[np.ndarray] = []

    # Batch 0: caller's first
    eval_input_ids.append(idx_first.detach().cpu().numpy().astype(np.int64).copy())
    eval_targets.append(targets_first.detach().cpu().numpy().astype(np.int64).copy())

    for n in range(1, eval_steps):
        idx_np = rng.integers(0, vocab_size, size=(B, T), dtype=np.int64)
        tgt_np = rng.integers(0, vocab_size, size=(B, T), dtype=np.int64)
        if n == slow_path_at_step:
            # §3.3 D10 — slow path branch coverage. Mark ~25%
            # (every 4th flat index) as -1 (ignore_index). Deterministic
            # pattern (NOT rng-based) so the slow path is reliably hit.
            flat = tgt_np.reshape(-1)
            flat[::4] = -1
            tgt_np = flat.reshape(B, T)
        eval_input_ids.append(idx_np)
        eval_targets.append(tgt_np)

    # Per-step capture: loss2d + accumulator updates (PT loss_eval.py:11-46
    # 1:1 mirror, single-rank skip)
    per_step_loss_2d_list: list[np.ndarray] = []
    total_nats_pt = torch.tensor(0.0, dtype=torch.float32)
    total_bytes_pt = torch.tensor(0, dtype=torch.int64)

    with torch.no_grad():
        for n in range(eval_steps):
            x = torch.from_numpy(eval_input_ids[n])
            y = torch.from_numpy(eval_targets[n])
            # PT loss_eval.py:13 — model(x, y, loss_reduction='none')
            loss2d = model(x, y, loss_reduction="none") # (B*T,) fp32
            loss2d_flat = loss2d.view(-1)
            per_step_loss_2d_list.append(loss2d_flat.detach().cpu().numpy())

            y_flat = y.view(-1)
            # PT loss_eval.py:16 — fast vs slow branch
            if (y_flat.int() < 0).any():
                valid = y_flat >= 0
                y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
                num_bytes2d = torch.where(
                    valid,
                    token_bytes_pt[y_safe],
                    torch.zeros_like(y_flat, dtype=token_bytes_pt.dtype),
                )
            else:
                num_bytes2d = token_bytes_pt[y_flat]
            total_nats_pt += (loss2d_flat * (num_bytes2d > 0)).sum().to(
                total_nats_pt.dtype
            )
            total_bytes_pt += num_bytes2d.sum()

    # PT loss_eval.py:57-64 — host-side division
    import math as _math
    total_nats_host = float(total_nats_pt.item())
    total_bytes_host = int(total_bytes_pt.item())
    if total_bytes_host == 0:
        expected_bpb = float("inf")
    else:
        expected_bpb = total_nats_host / (_math.log(2) * total_bytes_host)

    # Pack arrays (numpy, ready for arrays[f"bpb_eval/{k}"] in generate_one)
    return {
        "input_ids": np.stack(eval_input_ids, axis=0), # (eval_steps, B, T)
        "targets": np.stack(eval_targets, axis=0), # (eval_steps, B, T)
        "token_bytes": token_bytes_np, # (vocab_size,) int32
        "per_step_loss_2d": np.stack(
            per_step_loss_2d_list, axis=0
        ), # (eval_steps, B*T) fp32
        "total_nats": np.float32(total_nats_host), # () fp32
        "total_bytes": np.int64(total_bytes_host), # () int64
        "expected_bpb": np.float32(expected_bpb), # () fp32
    }


# -----------------------------------------------------------------------------
# .npz dump with `_dtype_map` JSON key (PLAN §3.3 D5)
# -----------------------------------------------------------------------------


def dump_npz_with_dtype_map(
    path: Path, arrays: dict[str, np.ndarray], meta: dict[str, Any]
) -> None:
    """Dump arrays + meta to a compressed .npz with a JSON ``_dtype_map``.

    ``np.savez_compressed`` loses ``ml_dtypes.bfloat16`` dtype labels (stored
    as ``|V2`` void; raw bytes survive). We work around this by storing the
    per-key dtype labels in a reserved ``_dtype_map`` key (JSON of
    ``{key: str(dtype)}``). The loader uses this map to restore the dtype
    via ``arr.view(target_dtype)`` recommendation).

    Reserved keys ``_dtype_map`` and ``_meta`` must not appear in ``arrays``.
    """
    if "_dtype_map" in arrays or "_meta" in arrays:
        raise ValueError(
            "Reserved keys '_dtype_map' / '_meta' must not appear in arrays"
        )
    dtype_map = {k: str(v.dtype) for k, v in arrays.items()}
    np.savez_compressed(
        path,
        **arrays,
        _dtype_map=np.array(json.dumps(dtype_map)),
        _meta=np.array(json.dumps(meta)),
    )


def _torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert ``torch.Tensor`` → ``np.ndarray``, with bf16 support.

    PyTorch ``.numpy()`` does not support ``torch.bfloat16`` directly. path
    uses ``view(torch.uint16)`` to expose raw bytes, then ``.view(ml_dtypes.bfloat16)``
    on the numpy side — bit-exact dtype preservation (no fp32 intermediate cast,
    no ULP loss). INVESTIGATION §3.5 raw-byte pattern.

    fp32 / int64 / etc. paths unchanged — direct ``.numpy()`` is dtype-preserving.
    """
    if t.dtype == torch.bfloat16:
        # uint16 view preserves bit pattern; ml_dtypes.bfloat16 reinterprets.
        return t.detach().cpu().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    return t.detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Per-file generator (PLAN §3.3 D9)
# -----------------------------------------------------------------------------


def generate_one(
    out_dir: Path,
    depth: int,
    granularity: Granularity,
    minimal: bool,
    *,
    label: str,
    seq_len: int = 64,
    batch_size: int = 2,
    seed: int = 42,
    with_grad: bool = False,
    with_optim_step: bool = False,
    with_train_loop: bool = False,
    with_bpb_eval: bool = False,
    train_num_iter: int = 100,
    bpb_eval_steps: int = 5,
) -> Path:
    """Generate one ``golden_<label>.npz`` and return its path.

    Includes the D14 mirror-parity sanity check; raises ``AssertionError`` on
    drift between ``forward_with_intermediates`` and ``model.forward`` (rather
    than silently writing a possibly-wrong golden).

    Config selection priority (I-02 d12 real + + + cascade):
    0. label ``"d12_sliding_bf16"`` AND ``depth=12`` → :func:`make_d12_real_config`
       (I-02 Phase 2 end integrated, sequence_len=2048, vocab=32768, "SSSL", n_kv_head=6
       MHA, n_embd=768, bf16 via env) ⭐⭐
    1. label ``"sliding"`` AND ``"gqa"`` AND ``"bf16"`` → :func:`make_sliding_gqa_bf16_config`

    2. label ``"sliding"`` AND ``"gqa"`` → :func:`make_sliding_gqa_config`

    3. label ``"sliding"`` → :func:`make_sliding_config`

    4. label ``"gqa"`` → :func:`make_gqa_config`

    5. label ``"bf16"`` → :func:`make_bf16_config`

    6. otherwise → :func:`make_config`
       (legacy, sequence_len=128, "L", n_kv_head=4)

     bf16 cascade: PT GPTConfig selfhas dtype field none — caller
    (``main()``) ``NANOCHAT_DTYPE=bfloat16`` env + ``importlib.reload``
    module-level ``COMPUTE_DTYPE`` is changed so cascade enabled. this function module-level
    ``GPT/GPTConfig`` lookup so reload + re-call when new GPT/GPTConfig used.
    """
    out_path = out_dir / f"golden_{label}.npz"

    # Determinism (PLAN §3.3 D3)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    if "d12" in label and "sliding" in label and "bf16" in label:
        cfg = make_d12_real_config(depth) # I-02 Phase 2 end integrated ⭐⭐
    elif "sliding" in label and "gqa" in label and "bf16" in label:
        cfg = make_sliding_gqa_bf16_config(depth)
    elif "sliding" in label and "gqa" in label:
        cfg = make_sliding_gqa_config(depth)
    elif "sliding" in label:
        cfg = make_sliding_config(depth)
    elif "gqa" in label:
        cfg = make_gqa_config(depth)
    elif "bf16" in label:
        cfg = make_bf16_config(depth)
    else:
        cfg = make_config(depth)
    model = GPT(cfg, pad_vocab_size_to=64)
    model.init_weights()
    if minimal:
        zero_extra_features(model)
    model.eval()

    # Inputs (numpy → torch; numpy seed already set above)
    input_ids_np = np.random.randint(
        0, cfg.vocab_size, size=(batch_size, seq_len), dtype=np.int64
    )
    targets_np = np.random.randint(
        0, cfg.vocab_size, size=(batch_size, seq_len), dtype=np.int64
    )
    idx = torch.from_numpy(input_ids_np)
    tgts = torch.from_numpy(targets_np)

    # D14 sanity: mirror parity (logits + loss bit-exact with model.forward)
    with torch.no_grad():
        ref_logits = model.forward(idx)
        ref_loss = model.forward(idx, targets=tgts)
        mirror_min = forward_with_intermediates(
            model, idx, targets=tgts, granularity="minimal"
        )
    if not torch.equal(ref_logits, mirror_min["logits"]):
        max_diff = (ref_logits - mirror_min["logits"]).abs().max().item()
        raise AssertionError(
            f"D14 mirror parity (logits) failed for {label}: max diff {max_diff}"
        )
    if not torch.equal(ref_loss, mirror_min["loss"]):
        diff = (ref_loss - mirror_min["loss"]).abs().max().item()
        raise AssertionError(
            f"D14 mirror parity (loss) failed for {label}: diff {diff}"
        )

    # Real run at requested granularity (deterministic, identical for shared keys)
    with torch.no_grad():
        ints = forward_with_intermediates(
            model, idx, targets=tgts, granularity=granularity
        )

    # : optional gradient capture .
    # Done AFTER no-grad forward to avoid double work; uses fresh autograd path.
    grads: dict[str, torch.Tensor] = {}
    if with_grad:
        grads = forward_with_grad(model, idx, tgts)

    # : optional BPB evaluation capture .
    # eval-only forward (no autograd, no optim mutation) → does NOT mutate
    # model. Run BEFORE (which DOES mutate via optimizer.step()) so the
    # captured per-token losses match the same initial weights as grad
    # and the existing `tensor/loss` baseline. also runs eval at start
    # of training (step 0 baseline = same model state).
    bpb_eval_arrays: dict[str, np.ndarray] = {}
    if with_bpb_eval:
        bpb_eval_arrays = forward_with_bpb_eval(
            model, idx, tgts, eval_steps=bpb_eval_steps,
        )

    # CRITICAL: capture initial weights BEFORE optimizer.step() .
    # `forward_with_optim_step` mutates model via `optimizer.step()` (in-place
    # `p.add_`/`p.sub_` ops on parameter storage). Since ``_torch_to_numpy``
    # returns a zero-copy view (`.detach().cpu().numpy()` shares storage on
    # CPU), we MUST clone the tensors before extracting numpy arrays — else
    # the captured arrays would alias the post-step values, silently breaking
    # the `weight/` baseline . The clone is explicit and
    # cheap (only triggered for the 3 with_optim_step entries).
    initial_weights_pt = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_weights = {k: _torch_to_numpy(v) for k, v in initial_weights_pt.items()}

    # : optional 1-step MuonAdamW capture .
    # Done AFTER weights/intermediates/grads since this MUTATES model state.
    weights_after: dict[str, torch.Tensor] = {}
    optim_state: dict[str, torch.Tensor] = {}
    if with_optim_step:
        weights_after, optim_state = forward_with_optim_step(model, idx, tgts)

    # : optional N-step train loop capture .
    # Done LAST since it MUTATES model state through N optimizer.step() calls.
    # CRITICAL: must be on a FRESH model (not the one mutated by above) so
    # that step 0 of the train loop matches the / 1-step baseline at
    # the same initial weights. We rebuild the model here.
    #
    # Per-step capture (chaos divergence finding redesign): capture per-step
    # (grad, out_weights, out_optim_state) for ALL N steps so JAX-side tests
    # can verify each step's algorithmic correctness INDEPENDENTLY by anchoring
    # at PT-stored input state. This delivers strict atol on every one of the
    # N steps (no chaos amplification accumulation).
    train_loss_curve = np.zeros(0, dtype=np.float32)
    train_per_step_grads: list[dict[str, torch.Tensor]] = []
    train_per_step_weights_after: list[dict[str, torch.Tensor]] = []
    train_per_step_optim_state_after: list[dict[str, torch.Tensor]] = []
    if with_train_loop:
        # Rebuild model from scratch (deterministic given seed, / path
        # also rebuilds when needed). zero_extra_features mirror.
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        train_model = GPT(cfg, pad_vocab_size_to=64)
        train_model.init_weights()
        if minimal:
            zero_extra_features(train_model)
        train_model.train() # base_train.py: model.train() before loop

        (
            train_loss_curve,
            train_per_step_grads,
            train_per_step_weights_after,
            train_per_step_optim_state_after,
        ) = forward_with_train_loop(
            train_model, idx, tgts, num_iterations=train_num_iter,
        )

    # Final state derived from last entry (backward compat with sister tests
    # and existing post-100-step Lenient atol regression patterns).
    train_weights_after_N: dict[str, torch.Tensor] = (
        train_per_step_weights_after[-1] if train_per_step_weights_after else {}
    )
    train_optim_state_after_N: dict[str, torch.Tensor] = (
        train_per_step_optim_state_after[-1] if train_per_step_optim_state_after else {}
    )

    # Build .npz contents (weight/ uses pre-step snapshot — baseline preserved)
    arrays: dict[str, np.ndarray] = {}
    for k, v in initial_weights.items():
        arrays[f"weight/{k}"] = v
    for k, v in ints.items():
        arrays[f"tensor/{k}"] = _torch_to_numpy(v)
    for k, v in grads.items():
        arrays[f"grad/{k}"] = _torch_to_numpy(v)
    for k, v in weights_after.items():
        arrays[f"weights_after/{k}"] = _torch_to_numpy(v)
    for k, v in optim_state.items():
        arrays[f"optim_state/{k}"] = _torch_to_numpy(v)
    if with_train_loop:
        arrays["train/loss_curve"] = train_loss_curve
        # Final state (backward compat for chaos baseline post-100-step tests)
        for k, v in train_weights_after_N.items():
            arrays[f"train/weights_after_N/{k}"] = _torch_to_numpy(v)
        for k, v in train_optim_state_after_N.items():
            arrays[f"train/optim_state_after_N/{k}"] = _torch_to_numpy(v)
        # Per-step arrays (NEW for strict per-step verification): each value
        # is shape (N, *original_shape) — first axis is the timestep index.
        # Stack along new axis 0 across N timesteps. We assume all timesteps
        # have the same set of keys (they do for grads/weights/optim_state
        # since the model topology is fixed across the loop).
        if train_per_step_grads:
            grad_keys = list(train_per_step_grads[0].keys())
            for k in grad_keys:
                stacked = torch.stack(
                    [train_per_step_grads[i][k] for i in range(train_num_iter)]
                )
                arrays[f"train/per_step_grads/{k}"] = _torch_to_numpy(stacked)
        if train_per_step_weights_after:
            weight_keys = list(train_per_step_weights_after[0].keys())
            for k in weight_keys:
                stacked = torch.stack(
                    [train_per_step_weights_after[i][k] for i in range(train_num_iter)]
                )
                arrays[f"train/per_step_weights_after/{k}"] = _torch_to_numpy(stacked)
        if train_per_step_optim_state_after:
            # First step's optim state has all keys (AdamW gets state on first
            # call; Muon gets state for first param of each group on first call
            # too). Use union of all-step keys to be safe.
            state_keys: set[str] = set()
            for s in train_per_step_optim_state_after:
                state_keys.update(s.keys())
            for k in sorted(state_keys):
                # Some keys may be missing in some timesteps (defensive); fill
                # with first-available shape's zeros if missing. In practice
                # all keys present in step 0 are present in every step (PT
                # MuonAdamW populates state on first call and reuses).
                tensors_for_key = []
                template = None
                for s in train_per_step_optim_state_after:
                    if k in s:
                        tensors_for_key.append(s[k])
                        if template is None:
                            template = s[k]
                if len(tensors_for_key) != train_num_iter:
                    raise AssertionError(
                        f"per_step_optim_state_after key {k!r} only present in "
                        f"{len(tensors_for_key)}/{train_num_iter} steps — "
                        f"PT optim state structure changed mid-loop?"
                    )
                stacked = torch.stack(tensors_for_key)
                arrays[f"train/per_step_optim_state_after/{k}"] = _torch_to_numpy(stacked)
    arrays["input/input_ids"] = input_ids_np
    arrays["input/targets"] = targets_np

    # : BPB evaluation arrays .
    if with_bpb_eval:
        for k, v in bpb_eval_arrays.items():
            arrays[f"bpb_eval/{k}"] = v

    # : dtype meta reflects actual COMPUTE_DTYPE (env-driven cascade dtype,
    # not the .npz storage dtype). bf16 entries store transformer body tensors
    # as bf16 + logits/loss as fp32 (PT mirror gpt.py:466).
    cascade_dtype = "bfloat16" if "bf16" in label else "float32"

    meta = {
        "depth": depth,
        "config": {
            "sequence_len": cfg.sequence_len,
            "vocab_size": cfg.vocab_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_kv_head": cfg.n_kv_head,
            "n_embd": cfg.n_embd,
            "window_pattern": cfg.window_pattern,
            "padded_vocab_size": ((cfg.vocab_size + 63) // 64) * 64,
        },
        "granularity": granularity,
        "dtype": cascade_dtype, # : "bfloat16" for bf16 entries
        "minimal": minimal,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "seed": seed,
        "n_weight_keys": sum(1 for k in arrays if k.startswith("weight/")),
        "n_tensor_keys": sum(1 for k in arrays if k.startswith("tensor/")),
        "n_grad_keys": sum(1 for k in arrays if k.startswith("grad/")),
        "n_weights_after_keys": sum(
            1 for k in arrays if k.startswith("weights_after/")
        ),
        "n_optim_state_keys": sum(
            1 for k in arrays if k.startswith("optim_state/")
        ),
        "with_grad": with_grad,
        "with_optim_step": with_optim_step,
        "with_train_loop": with_train_loop,
        "with_bpb_eval": with_bpb_eval,
        "train_num_iter": train_num_iter if with_train_loop else 0,
        "bpb_eval_steps": bpb_eval_steps if with_bpb_eval else 0,
        "n_train_weights_after_N_keys": sum(
            1 for k in arrays if k.startswith("train/weights_after_N/")
        ),
        "n_train_optim_state_after_N_keys": sum(
            1 for k in arrays if k.startswith("train/optim_state_after_N/")
        ),
        "n_train_per_step_grads_keys": sum(
            1 for k in arrays if k.startswith("train/per_step_grads/")
        ),
        "n_train_per_step_weights_after_keys": sum(
            1 for k in arrays if k.startswith("train/per_step_weights_after/")
        ),
        "n_train_per_step_optim_state_after_keys": sum(
            1 for k in arrays if k.startswith("train/per_step_optim_state_after/")
        ),
        "n_bpb_eval_keys": sum(
            1 for k in arrays if k.startswith("bpb_eval/")
        ),
    }

    dump_npz_with_dtype_map(out_path, arrays, meta)
    return out_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

# (depth, granularity, minimal, label, input_seq_len, with_grad)
# - v1 (4 entries): seq_len=64 (small input, window="L" no sliding).
# - (1 entry): "d4_sliding_minimal" requires input seq_len > short_window=128
# to actually engage sliding (PT _sdpa_attention branch 3); 192 is 1.5x short.
# - Step 1 (1 entry): "d4_gqa_minimal" — n_head=4/n_kv_head=2 (n_rep=2),
# window="L" → seq_len=64 (sliding agnostic).
# - Step 2 cascade (1 entry): "d4_sliding_gqa_minimal" — sliding+GQA combined,
# seq_len=192 .
# - Step 1 (1 entry): "d4_bf16_minimal" — MHA full causal + bf16 cascade
# (NANOCHAT_DTYPE=bfloat16 enabled, importlib.reload after handled, main()).
# - Step 2 cascade (1 entry): "d4_sliding_gqa_bf16_minimal" — ++
# combined, Phase 2 end integrated d12 enter before validation.
# - I-02 Phase 2 end integrated (1 entry): "d12_sliding_bf16" — d12 real config
# (PT default mirror: sequence_len=2048, vocab=32768, n_layer=12, n_embd=768,
# n_head=6, n_kv_head=6 MHA, "SSSL", bf16). seq_len=768 (1.5×short_window=512,
# gotcha ② avoid). .gitignore + generate-on-demand (PLAN Q-01 (B), ~700MB
# raw / ~500MB compressed). ⭐⭐ Phase 3 enter trigger.
# - with_grad (3 entries marked True): d1 fp32 (smallest sanity), d4_sliding_gqa
# fp32 cascade , d4_sliding_gqa_bf16 cascade .
# Other entries with_grad=False (forward-only, atomic scope).
# - with_optim_step (3 entries marked True): same 3 configs as grad.
# Captures `weights_after/<key>` + `optim_state/<key>/<attr>` after 1
# MuonAdamW step. Mutates model — must run after grad capture (which itself
# doesn't mutate weights).
# with_train_loop (3 entries marked True): same 3 configs as grad and
# optim_step. Captures `train/loss_curve` (N=100,) + `train/weights_after_N/<key>`
# + `train/optim_state_after_N/<key>/<attr>` after N-step train loop. Mutates
# model — generate_one rebuilds a fresh model for the train loop run so the
# / 1-step baseline at the same initial weights is preserved.
# with_bpb_eval (3 entries marked True): same 3 configs. Captures
# `bpb_eval/{input_ids, targets, token_bytes, per_step_loss_2d, total_nats,
# total_bytes, expected_bpb}` from PT `evaluate_bpb` 1:1 mirror. eval-only
# forward (no autograd, no optim mutation) → same model state as grad.
_GOLDEN_SPEC: list[
    tuple[int, Granularity, bool, str, int, bool, bool, bool, bool]
] = [
    # (depth, granularity, minimal, label, input_seq_len, with_grad, with_optim_step, with_train_loop, with_bpb_eval)
    (1, "high", False, "d1", 64, True, True, True, True), # + + + ⭐⭐⭐
    (4, "block", True, "d4_minimal", 64, False, False, False, False),
    (4, "block", False, "d4_full", 64, False, False, False, False),
    (12, "minimal", False, "d12", 64, False, False, False, False),
    (4, "block", True, "d4_sliding_minimal", 192, False, False, False, False), #
    (4, "block", True, "d4_gqa_minimal", 64, False, False, False, False), # Step 1
    (4, "block", True, "d4_sliding_gqa_minimal", 192, True, True, True, True), # + + + + ⭐⭐⭐
    (4, "block", True, "d4_bf16_minimal", 64, False, False, False, False), # Step 1 ⭐
    (4, "block", True, "d4_sliding_gqa_bf16_minimal", 192, True, True, True, True), # + + + + ⭐⭐⭐
    (12, "block", True, "d12_sliding_bf16", 768, False, False, False, False), # I-02 Phase 2 end ⭐⭐⭐
]


def main() -> None:
    # : declare globals up front so the bf16 reload branch can rebind them
    # (Python requires `global` before any use within the function).
    global GPT, GPTConfig, norm, COMPUTE_DTYPE # noqa: PLW0603
    parser = argparse.ArgumentParser(description=" golden test generator")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/golden"),
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="*",
        default=None,
        help="Subset of depths (1, 4, 12). Default: all.",
    )
    parser.add_argument(
        "--no-minimal",
        action="store_true",
        help="Skip d4_minimal variant (debug)",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[] Output dir: {args.out.resolve()}")
    print(f"[] COMPUTE_DTYPE (initial) = {COMPUTE_DTYPE}")

    selected: list[tuple[int, Granularity, bool, str, int, bool, bool, bool, bool]] = []
    for spec in _GOLDEN_SPEC:
        depth, _gran, minimal, _label, _seq_len, _with_grad, _with_optim, _with_train, _with_bpb = spec
        if args.depths is not None and depth not in args.depths:
            continue
        if args.no_minimal and minimal:
            continue
        selected.append(spec)

    # (PLAN §3.3 D7): split fp32 / bf16 entries; bf16 entries first
    # ``importlib.reload`` ``COMPUTE_DTYPE`` ``torch.bfloat16`` first.
    # ``nanochat`` module load when ``COMPUTE_DTYPE`` decided so (common.py:31),
    # env var changedonly effect none. reload + reimport new module-level binding.
    fp32_entries = [s for s in selected if "bf16" not in s[3]]
    bf16_entries = [s for s in selected if "bf16" in s[3]]

    def _generate_batch(entries: list[tuple[int, Granularity, bool, str, int, bool, bool, bool, bool]]) -> None:
        for depth, gran, minimal, label, input_seq_len, with_grad, with_optim_step, with_train_loop, with_bpb_eval in entries:
            tags = []
            if with_grad:
                tags.append("grad")
            if with_optim_step:
                tags.append("optim")
            if with_train_loop:
                tags.append("train")
            if with_bpb_eval:
                tags.append("bpb")
            tag_str = f" +{'+'.join(tags)}" if tags else ""
            print(
                f"[] Generating golden_{label}.npz "
                f"(depth={depth}, granularity={gran}, minimal={minimal}, "
                f"input_seq_len={input_seq_len}{tag_str}) …"
            )
            path = generate_one(
                args.out, depth, gran, minimal,
                label=label, seq_len=input_seq_len,
                with_grad=with_grad, with_optim_step=with_optim_step,
                with_train_loop=with_train_loop,
                with_bpb_eval=with_bpb_eval,
            )
            size_kb = path.stat().st_size / 1024
            print(f"[] → {path.name}: {size_kb:.1f} KB")

    # 1. fp32 entries (current state, NANOCHAT_DTYPE=float32)
    if fp32_entries:
        print(f"[] Phase 1: {len(fp32_entries)} fp32 entries")
        _generate_batch(fp32_entries)

    # 2. bf16 entries : reload nanochat with NANOCHAT_DTYPE=bfloat16
    if bf16_entries:
        print(f"\n[] Phase 2: {len(bf16_entries)} bf16 entries ")
        print("[] Switching to NANOCHAT_DTYPE=bfloat16 + importlib.reload …")
        os.environ["NANOCHAT_DTYPE"] = "bfloat16"
        import nanochat.common
        import nanochat.gpt

        importlib.reload(nanochat.common)
        importlib.reload(nanochat.gpt)
        # Rebind module-level names so generate_one (and make_*_config) pick up
        # the reloaded classes / module variables.
        from nanochat.common import COMPUTE_DTYPE # noqa: F811
        from nanochat.gpt import GPT, GPTConfig, norm # noqa: F811

        print(f"[] COMPUTE_DTYPE (after reload) = {COMPUTE_DTYPE}")
        assert str(COMPUTE_DTYPE) == "torch.bfloat16", (
            f" reload sanity: expected torch.bfloat16, got {COMPUTE_DTYPE}"
        )
        _generate_batch(bf16_entries)
    print(f"[] Done. {len(selected)} file(s).")


if __name__ == "__main__":
    main()
