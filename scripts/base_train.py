"""Train a base model on multiple TPU hosts (data-parallel).

Composes the JAX-native sharding primitives in :mod:`nanochat_jax.sharding`,
the optimizer in :mod:`nanochat_jax.optim`, and the training loop helpers in
:mod:`nanochat_jax.base_train` into a self-contained training entry point.
The validation BPB pass uses :func:`nanochat_jax.loss_eval.evaluate_bpb`
(multi-host aware via ``jax.experimental.multihost_utils``).

Out of scope (Phase 6+ follow-up):

- gradient accumulation (default ``grad_accum_steps=1``)
- checkpoint save / resume
- wandb logging (master ``jax.process_index == 0`` print only)
- mixed precision (FP8 GradScaler) — bf16/fp32 only
- CORE metric

Composes:

- :func:`jax.distributed.initialize` — multi-host coordinator (auto-detect)
- :func:`nanochat_jax.common.setup_distributed_env_vars` — JAX semantic → PT
  env-var bridge so dataloader can read rank info
- :func:`nanochat_jax.sharding.make_mesh` — 1D ``("data",)`` mesh over
  ``jax.device_count``
- :func:`nanochat_jax.base_train.init_train_state` — model + param_groups +
  zero-init optim_state
- :func:`nanochat_jax.base_train.make_train_step_sharded` — +
  factory (closure capture mesh + param_groups + optim_state)
- 100-step Python loop with host-side schedules
- NaN guard + training divergence check
- Per-step weights save for M6 metric (single-host vs multi-host comparison)

References:

- PT mirror: ``nanochat/scripts/base_train.py:507-540``

Usage::

    # CPU dry run (Phase 0)
    python scripts/base_train.py --depth=2 --num-iterations=2 \\
        --device-batch-size=2 --seq-len=128 --no-distributed

    # single-host TPU v6e-2/4 (Phase 1/2 sanity)
    python scripts/base_train.py --depth=12 --num-iterations=100 \\
        --device-batch-size=4 --weights-out=weights_v6e2.npz

    # multi-host TPU v6e-8 (Phase 3 M6 trigger, run on all workers)
    gcloud compute tpus tpu-vm ssh ... --worker=all --command="\\
        python scripts/base_train.py --depth=12 --num-iterations=100 \\
            --device-batch-size=4 --weights-out=weights_v6e8.npz"
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Allow `python scripts/base_train.py` from the project root .
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from nanochat_jax.base_train import ( # noqa: E402 — sys.path mutation above
    get_lr_multiplier,
    get_muon_momentum,
    get_weight_decay,
    init_train_state,
    make_grad_accum_fns,
    make_train_step_sharded,
)
from nanochat_jax.common import setup_distributed_env_vars # noqa: E402
from nanochat_jax.gpt import GPT, GPTConfig # noqa: E402
from nanochat_jax.grad_utils import nnx_state_to_flat_dict # noqa: E402
from nanochat_jax.loss_eval import evaluate_bpb # noqa: E402 —
from nanochat_jax.perf import compute_mfu, get_peak_bf16_tflops # noqa: E402
from nanochat_jax.sharding import get_process_info, make_mesh # noqa: E402


def make_d12_config(
    *,
    depth: int = 12,
    sequence_len: int = 2048,
    vocab_size: int = 32768,
    n_head: int = 6,
    n_kv_head: int = 6,
    n_embd: int = 768,
    window_pattern: str = "SSSL",
    bf16: bool = True,
    attn_impl: str = "xla",
) -> GPTConfig:
    """d12 default config (I-02 cascade — §3.11 ).

    Defaults match ``scripts/generate_golden.py:make_d12_real_config``
    (n_layer=12, MHA, bf16) for downstream BPB / golden comparison.

    **Backward-compat alias** for :func:`make_config` .
    Direct use is preserved for / regression (n_embd=768 hardcoded).
    """
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        window_pattern=window_pattern,
        compute_dtype=jnp.bfloat16 if bf16 else jnp.float32,
        attn_impl=attn_impl,
    )


def make_config(
    *,
    depth: int,
    aspect_ratio: int = 64,
    head_dim: int = 128,
    sequence_len: int = 2048,
    vocab_size: int = 32768,
    window_pattern: str = "SSSL",
    bf16: bool = True,
    attn_impl: str = "xla",
) -> GPTConfig:
    """ : PT 1:1 mirror of ``nanochat/scripts/base_train.py:build_model_meta``.

    Generalize from :func:`make_d12_config`, supporting any depth with the
    same ``aspect_ratio`` / ``head_dim`` convention as PT base_train.py:

    - ``base_dim = depth * aspect_ratio``
    - ``n_embd = ceil(base_dim / head_dim) * head_dim`` (round up to multiple of head_dim)
    - ``n_head = n_embd // head_dim``
    - ``n_kv_head = n_head`` (MHA default — PT base_train.py mirror)

    For ``depth=24, aspect_ratio=64, head_dim=128`` (Karpathy d24 default):
    - ``n_embd = 1536`` (= 24 × 64)
    - ``n_head = 12`` (= 1536 / 128)
    - ``n_kv_head = 12`` (MHA)
    - matches Karpathy LEADERBOARD Run 1 exactly ✅

    For ``depth=12, aspect_ratio=64, head_dim=128`` :
    - ``n_embd = 768``, ``n_head = 6``, ``n_kv_head = 6`` (defaults preserved).
    """
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = n_embd // head_dim
    return GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=n_head,
        n_kv_head=n_head, # MHA — PT base_train.py mirror
        n_embd=n_embd,
        window_pattern=window_pattern,
        compute_dtype=jnp.bfloat16 if bf16 else jnp.float32,
        attn_impl=attn_impl,
    )


def make_synthetic_batch(
    rng: np.random.Generator, batch_size: int, seq_len: int, vocab_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``(idx, targets)`` synthetic random tokens.

     cascade pattern (random data, deterministic seed). Returns numpy int64
    arrays of shape ``(batch_size, seq_len)``; caller converts to JAX arrays.

    Phase 5 atomic scope uses synthetic data (PLAN §3.7 ):

    - Real ClimbMix dataloader integration is ✅ Done but cross-rank token
      ordering for M6 metric is sensitive to rank-stride ( row-group
      sharding). Synthetic random data lets us isolate **sharding numerical
      correctness** from data ordering effects.
    - (M7 BPB) will use real dataloader.

    Returns ``(idx, targets)`` where ``idx[i, t] = tokens[i, t]`` and
    ``targets[i, t] = tokens[i, t+1]`` (next-token prediction, PT mirror).
    """
    tokens = rng.integers(0, vocab_size, size=(batch_size, seq_len + 1), dtype=np.int64)
    return tokens[:, :-1], tokens[:, 1:]


def save_weights(model: GPT, path: str) -> None:
    """Extract NNX :class:`Param` leaves and save as ``.npz`` for comparison.

    Uses :func:`nnx_state_to_flat_dict` to produce a flat
    string-keyed dict matching weight_converter format. The .npz can be
    loaded by :func:`numpy.load` and compared element-wise across runs
    (single-host vs multi-host) to compute per-step weights diff for M6.
    """
    params_state = nnx.state(model, nnx.Param)
    flat_dict = nnx_state_to_flat_dict(params_state)
    np_dict = {k: np.asarray(v) for k, v in flat_dict.items()}
    # Sanitize keys: numpy savez silently rewrites '.' to '_' but we want the
    # original keys so M6 comparison can use the same names. We use savez with
    # a wrapping function and explicit keyword rebind.
    np.savez(path, **{_safe_key(k): v for k, v in np_dict.items()})



def _safe_key(k: str) -> str:
    """numpy savez uses keyword args; replace '.' with '__' to round-trip."""
    return k.replace(".", "__")


def restore_key(safe_k: str) -> str:
    """Inverse of :func:`_safe_key` — restore original dotted key."""
    return safe_k.replace("__", ".")


def _rolling_delete_old_checkpoints(
    checkpoint_dir: str, *, keep_last: int = 2
) -> int:
    """Keep only the last N checkpoints to prevent disk full (T06 fix).

    Looks for ``model_<step:06d>.pt`` files in ``checkpoint_dir`` and removes
    the oldest, keeping ``keep_last`` newest. Also removes corresponding
    ``meta_<step:06d>.json``. Idempotent — no-op if fewer than ``keep_last``.

    Per OPERATIONAL-TRAPS T06 cookbook (work 0036). Master-only call.
    Defensive against concurrent deletion races (OSError → continue).

    Returns:
        Number of checkpoint pairs deleted.
    """
    import re

    pattern = re.compile(r"^model_(\d{6})\.pt$")
    ckpts: list[tuple[int, str]] = []
    for fname in os.listdir(checkpoint_dir):
        m = pattern.match(fname)
        if m:
            ckpts.append((int(m.group(1)), fname))
    ckpts.sort()
    if len(ckpts) <= keep_last:
        return 0
    deleted = 0
    for _step, fname in ckpts[:-keep_last]:
        model_path = os.path.join(checkpoint_dir, fname)
        try:
            os.remove(model_path)
        except OSError:
            continue # concurrent del race — defensive
        meta_fname = fname.replace("model_", "meta_").replace(".pt", ".json")
        meta_path = os.path.join(checkpoint_dir, meta_fname)
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except OSError:
                pass
        deleted += 1
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="multi-TPU training + full pretrain")
    parser.add_argument("--depth", type=int, default=12, help="Model depth (n_layer)")
    # : generalize d24 config (aspect_ratio + head_dim)
    parser.add_argument(
        "--use-pt-mirror-config", action="store_true",
        help=" : use PT 1:1 mirror build_model_meta (aspect_ratio + head_dim). "
        "Default: backward-compat to make_d12_config (n_embd=768 hardcoded for / cascade).",
    )
    parser.add_argument(
        "--aspect-ratio", type=int, default=64,
        help=" : with --use-pt-mirror-config, n_embd = depth * aspect_ratio "
        "(rounded to head_dim multiple). Karpathy default 64. d24 → n_embd=1536.",
    )
    parser.add_argument(
        "--head-dim", type=int, default=128,
        help=" : with --use-pt-mirror-config, target head dimension. "
        "Karpathy default 128.",
    )
    parser.add_argument(
        "--attn-impl", type=str, default="xla", choices=["xla", "splash"],
        help=" v4 (work 0036): attention implementation. "
        "'xla' (default) = jax.nn.dot_product_attention(implementation='xla') "
        "(regression preserved, / baseline). "
        "'splash' = splash_attention_kernel TPU FlashAttention "
        ".",
    )
    parser.add_argument(
        "--num-iterations", type=int, default=-1,
        help="Total training steps. -1 = use --target-param-data-ratio. "
        "/ backward-compat: explicit positive value overrides ratio.",
    )
    # : auto num_iterations from target_param_data_ratio
    parser.add_argument(
        "--target-param-data-ratio", type=float, default=-1.0,
        help=" : target tokens / num_scaling_params ratio. -1 = disabled. "
        "Karpathy Run 1 d24 = 12 (Chinchilla=20). target_tokens = ratio * "
        "num_scaling_params (transformer_matrices + lm_head, PT 1:1 mirror). "
        "num_iterations = target_tokens // total_batch_size.",
    )
    parser.add_argument(
        "--device-batch-size", type=int, default=4, help="Per-device batch size"
    )
    parser.add_argument(
        "--total-batch-size", type=int, default=-1,
        help=" : total batch size in tokens. -1 = device_batch_size × seq_len × "
        "jax.device_count × grad_accum_steps. Karpathy d24 default = 524,288 (B_REF). "
        "Used to compute num_iterations from target_tokens.",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help=" : gradient accumulation steps. Default 1. "
        "Use >1 for v6e-8 + total_batch=524288 (Karpathy mirror, grad_accum=2).",
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize mesh + model + optimizer + dataloader, then exit. "
        "Useful for cheap end-to-end smoke tests on TPU.",
    )
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-frac", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.28) # Karpathy default; code scales → ~0.042 for d24
    # : explicit lr/wd args for Karpathy mirror (defaults match
    # nanochat_jax.base_train.init_train_state — / regression preserved).
    parser.add_argument(
        "--matrix-lr", type=float, default=0.02,
        help=" : Muon LR for matrix params. Karpathy default 0.02.",
    )
    parser.add_argument(
        "--embedding-lr", type=float, default=0.2,
        help=" : Adam LR for token embedding. nanochat init_train_state "
        "default 0.2. Karpathy d24 default = 0.3 .",
    )
    parser.add_argument(
        "--unembedding-lr", type=float, default=0.004,
        help=" : Adam LR for lm_head. nanochat init_train_state default "
        "0.004. Karpathy d24 default = 0.008 .",
    )
    parser.add_argument(
        "--scalar-lr", type=float, default=0.5,
        help=" : Adam LR for scalar params. Karpathy default 0.5.",
    )
    parser.add_argument(
        "--weights-out",
        type=str,
        default=None,
        help="Optional final weights .npz path (M6 comparison source)",
    )
    parser.add_argument(
        "--per-step-weights-out",
        type=str,
        default=None,
        help="Optional dir for per-step weights .npz (M6 metric only — ~756MB × N steps)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save per-step weights every N steps (default 1=every step). "
        "Step 0 + final step always saved. Use 50 for {0, 50, 99} sample only "
        "(suitable for d12 large model where per-step disk = ~60-100GB).",
    )
    parser.add_argument(
        "--no-distributed",
        action="store_true",
        help="Skip jax.distributed.initialize (CPU dry run)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bf16 compute (default; cascade). --no-bf16 disables.",
    )
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    # v8 (work 0039) — A/B test toggles for residual ~16% CORE gap.
    # Default OFF preserves // v5/v6/v7 baseline regression.
    parser.add_argument(
        "--cast-embeddings-bf16",
        action="store_true",
        default=False,
        help=" v8 (work 0039): cast wte and value_embeds Params to "
        "compute_dtype after init_weights (PT 1:1 mirror of gpt.py:258-261). "
        "Permanently stores embeddings as bf16 in bf16 mode → matches PT "
        "memory layout AND optimizer state precision (bf16 quantization noise "
        "may act as implicit regularization). Default off preserves baseline.",
    )
    parser.add_argument(
        "--polar-express-bf16",
        action="store_true",
        default=False,
        help=" v8 (work 0039): cast Muon gradient to bf16 before "
        "Polar Express NS iterations (PT 1:1 mirror of optim.py:117). "
        "Default off preserves baseline (NS in fp32). Affects all 24 layer "
        "Muon groups × 16,704 steps.",
    )
    # 0040 v9 fix candidate (Option B/C): TPU default fp32 matmul = bf16
    # internal acc, lm_head 197x noise amp (work 0040 RESULT.md §0.1). 'highest' =
    # 6-pass bf16 = near-fp32 acc (~2.5x slower step). For full retrain to
    # validate Option B, pass --matmul-precision highest.
    parser.add_argument(
        "--matmul-precision",
        type=str,
        default=None,
        choices=[None, "default", "high", "highest"],
        help="Override JAX default matmul precision (jax.config update). "
        "TPU default = bf16 internal acc (Fastest). 'highest' = 6-pass bf16 "
        "= near-fp32 acc (~2.5x slower). For v9 retrain (0040 Option B) "
        "use --matmul-precision=highest.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N steps (master process only).",
    )
    # (Phase 5 fifth sub-feature) -- val BPB measurement
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help=": use real ClimbMix dataloader instead of synthetic random batches "
        "(default: synthetic, path). Required for M7 BPB comparison.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=": skip training (num_iterations forced to 0) and run val BPB "
        "evaluation only on randomly-initialized weights. Use for sanity checks.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=20,
        help=": number of val batches for BPB measurement .",
    )
    parser.add_argument(
        "--bpb-out",
        type=str,
        default=None,
        help=": path for val BPB JSON result (master process only). "
        "If None, BPB measurement is skipped .",
    )
    # : intermediate checkpoint + resume
    parser.add_argument(
        "--checkpoint-every", type=int, default=-1,
        help=" : save intermediate checkpoint every N steps. -1 = disabled "
        "(only final saved if --checkpoint-dir set). 2000 recommended for spot "
        "preemption safety (PLAN R-7 cascade).",
    )
    parser.add_argument(
        "--keep-last-checkpoints", type=int, default=2,
        help=" v5 (T06 fix per OPERATIONAL-TRAPS, work 0037): keep only the "
        "last N intermediate checkpoints (rolling delete). Default 2 = ~11GB max "
        "disk for d24 (5.5GB × 2). v4 hit disk full @ step 8000 with "
        "16 ckpts × 5.5GB = 88GB / 100GB worker disk.",
    )
    parser.add_argument(
        "--resume-from-step", type=int, default=-1,
        help=" : resume training from step N checkpoint. -1 = disabled (start "
        "fresh). Loads model + optim_state + dataloader_state + loop_state from "
        "checkpoint_dir.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help=" : directory for intermediate + final checkpoints. Default = "
        "~/.cache/nanochat-tpu/base_checkpoints/{model_tag} if model_tag given.",
    )
    parser.add_argument(
        "--model-tag", type=str, default=None,
        help=" : model tag for checkpoint dir name. Default = "
        "'d{depth}_jax_seed{seed}' if not given.",
    )
    # : periodic val_bpb measurement during training
    parser.add_argument(
        "--eval-every", type=int, default=0,
        help=" : evaluate val_bpb every N steps. 0 = disabled (default). "
        "250 = Karpathy default. Checkpoint always saved BEFORE eval runs.",
    )
    parser.add_argument(
        "--eval-tokens", type=int, default=80 * 524_288,
        help=" : number of tokens per validation eval. Karpathy default "
        "80 * 524288 = 41,943,040. eval_steps = eval_tokens // (device_batch * seq_len * device_count).",
    )
    parser.add_argument(
        "--no-final-eval", action="store_true",
        help=" : suppress the forced final-step val_bpb eval that the "
        "periodic-eval branch fires unconditionally on is_last_step. Periodic "
        "evals at multiples of --eval-every continue to fire. No effect when "
        "--eval-every=0. Saves ~5-40 min of duplicate eval when an external "
        "base_eval.py pass runs after training (e.g. runs/d24.sh).",
    )
    parser.add_argument(
        "--token-bytes-path",
        type=str,
        default=None,
        help=": explicit path to token_bytes.npy. If None, uses "
        "``nanochat_jax.tokenizer.get_token_bytes`` (default tokenizer dir).",
    )
    args = parser.parse_args()

    # 0a. matmul precision override (TPU default fp32 matmul actually uses
    # bf16 internal accumulation; see OPS docs). MUST apply BEFORE
    # jax.distributed.initialize / jax.devices so all subsequent compilation
    # respects the precision setting.
    # See /RESULT.md §6.2 Option B.
    if args.matmul_precision is not None:
        jax.config.update("jax_default_matmul_precision", args.matmul_precision)
        print(f"[info] jax_default_matmul_precision = {args.matmul_precision} "
              "(0040 v9 fix candidate)")

    # 1. Multi-host coordinator. On single-process JAX (CPU / single-host
    # TPU) this is a no-op. On v6e-8 2-host pod slice it must be called from
    # all hosts to set up the cluster.
    if not args.no_distributed:
        try:
            jax.distributed.initialize()
        except RuntimeError as exc: # already-initialized or single-process
            print(
                f"[warn] jax.distributed.initialize skipped: {exc}",
                file=sys.stderr,
            )

    # 2. PT env-var bridge for dataloader
    setup_distributed_env_vars()

    process_idx, process_count, local_device_count = get_process_info()
    is_master = process_idx == 0

    if is_master:
        print(f"[info] jax.devices = {jax.devices()}")
        print(
            f"[info] process_index={process_idx} process_count={process_count} "
            f"local_device_count={local_device_count} "
            f"jax.device_count={jax.device_count()}"
        )
        print(
            f"[info] depth={args.depth} aspect_ratio={args.aspect_ratio} head_dim={args.head_dim} "
            f"device_batch_size={args.device_batch_size} seq_len={args.seq_len} "
            f"bf16={args.bf16} seed={args.seed}"
        )

    # 3. config + model init (deterministic seed). generalize:
    # --use-pt-mirror-config activates make_config (aspect_ratio + head_dim).
    # Default = make_d12_config for backward-compat with / regression
    # (n_embd=768 hardcoded for any depth).
    if args.use_pt_mirror_config:
        # path: PT 1:1 mirror of build_model_meta
        config = make_config(
            depth=args.depth,
            aspect_ratio=args.aspect_ratio,
            head_dim=args.head_dim,
            sequence_len=args.seq_len,
            vocab_size=args.vocab_size,
            bf16=args.bf16,
            attn_impl=args.attn_impl,
        )
    else:
        # / backward-compat (n_embd=768 hardcoded)
        config = make_d12_config(
            depth=args.depth,
            sequence_len=args.seq_len,
            vocab_size=args.vocab_size,
            bf16=args.bf16,
            attn_impl=args.attn_impl,
        )
    rngs = nnx.Rngs(args.seed)
    model = GPT(config, rngs=rngs)
    model.init_weights(
        seed=args.seed,
        cast_embeddings_to_compute_dtype=args.cast_embeddings_bf16,
    )
    if is_master and args.cast_embeddings_bf16:
        wte_dtype = model.transformer.wte.embedding.value.dtype
        print(
            f"[info] v8: cast_embeddings_bf16=ON — wte.dtype = {wte_dtype}"
        )

    # 4. total_batch_size + Karpathy-style LR/WD muP scaling → param_groups + optim_state
    # Batch size must be known before optimizer init (weight decay depends on it).
    grad_accum_steps = max(int(args.grad_accum_steps), 1)
    if args.total_batch_size > 0:
        total_batch_size_tokens = int(args.total_batch_size)
    else:
        total_batch_size_tokens = (
            args.device_batch_size * args.seq_len
            * max(jax.device_count(), 1) * grad_accum_steps
        )

    # Karpathy muP scaling (PT base_train.py:287-315):
    # batch_lr_scale = sqrt(B / B_ref) — η ∝ √(B/B_ref) for AdamW/Muon
    # weight_decay_scaled = wd × sqrt(B/B_ref) × (D_ref / D_target)
    # where D_ref/D_target = d12_scaling_params / d_X_scaling_params (ratio cancels)
    # This uses the T_epoch framework: λ ∝ √(B/B_ref) × (D_ref/D).
    _B_REF = 2**19 # 524288 tokens — Karpathy muP reference batch
    _scaling_counts = model.num_scaling_params()
    _d_x_scaling = _scaling_counts["transformer_matrices"] + _scaling_counts["lm_head"]
    # Standard d12 reference (depth=12, aspect_ratio=64, head_dim=128, vocab=32768):
    # transformer_matrices=84,935,088 + lm_head=25,165,824 = 110,100,912
    _D12_SCALING = 110_100_912
    _batch_lr_scale = (total_batch_size_tokens / _B_REF) ** 0.5
    weight_decay_scaled = args.weight_decay * _batch_lr_scale * (_D12_SCALING / _d_x_scaling)

    if is_master:
        n_params = sum(int(np.prod(v.shape)) for v in nnx_state_to_flat_dict(nnx.state(model, nnx.Param)).values())
        print(f"[info] model params (Param leaves) = {n_params:,} ({n_params / 1e6:.2f}M)")
        # : detailed param breakdown + scaling_params
        scaling_counts = _scaling_counts
        scaling_params = _d_x_scaling
        print(f"[info] scaling_params (transformer_matrices + lm_head) = {scaling_params:,} ({scaling_params/1e6:.2f}M)")
        print(f"[info] wte={scaling_counts['wte']/1e6:.2f}M, value_embeds={scaling_counts['value_embeds']/1e6:.2f}M, "
              f"lm_head={scaling_counts['lm_head']/1e6:.2f}M, "
              f"transformer_matrices={scaling_counts['transformer_matrices']/1e6:.2f}M, "
              f"scalars={scaling_counts['scalars']}")
        print(
            f"[info] muP scaling: batch_lr_scale={_batch_lr_scale:.4f} "
            f"weight_decay {args.weight_decay:.5f} → {weight_decay_scaled:.5f} "
            f"(×{_D12_SCALING / _d_x_scaling:.4f} d12/model ratio)"
        )

    # v8 (work 0039): optional Polar Express bf16 cast for Muon NS.
    polar_express_dtype = (
        config.compute_dtype if args.polar_express_bf16 else None
    )
    if is_master and polar_express_dtype is not None:
        print(
            f"[info] v8: polar_express_bf16=ON — Muon NS dtype = {polar_express_dtype}"
        )
    param_groups, optim_state = init_train_state(
        model,
        weight_decay=weight_decay_scaled,
        matrix_lr=args.matrix_lr * _batch_lr_scale,
        embedding_lr=args.embedding_lr * _batch_lr_scale,
        unembedding_lr=args.unembedding_lr * _batch_lr_scale,
        scalar_lr=args.scalar_lr * _batch_lr_scale,
        polar_express_dtype=polar_express_dtype,
    )

    # + : compute num_iterations (total_batch_size already above)
    if int(args.num_iterations) > 0:
        num_iterations = int(args.num_iterations)
        horizon_source = "explicit --num-iterations"
    elif args.target_param_data_ratio > 0:
        scaling_params = (
            model.num_scaling_params()["transformer_matrices"]
            + model.num_scaling_params()["lm_head"]
        )
        target_tokens = int(args.target_param_data_ratio * scaling_params)
        num_iterations = target_tokens // total_batch_size_tokens
        horizon_source = (
            f"--target-param-data-ratio={args.target_param_data_ratio} "
            f"(target_tokens={target_tokens:,}, scaling_params={scaling_params:,})"
        )
    else:
        # Backward-compat: default num_iterations was 100 (overridden by
        # --num-iterations arg historically)
        num_iterations = 100
        horizon_source = "default 100 "

    if is_master:
        print(
            f"[info] num_iterations = {num_iterations:,} "
            f"(total_batch_size = {total_batch_size_tokens:,} tokens, "
            f"grad_accum_steps = {grad_accum_steps}, source = {horizon_source})"
        )

    # 5. Mesh + sharded train_step factory
    mesh = make_mesh() # default: jax.device_count
    # v4 (work 0036): register mesh for Splash multi-chip shard_map.
    # Must be set BEFORE jit-tracing the train_step so the mesh is captured
    # as a closed-over constant. ``set_splash_mesh(None)`` is a no-op for the
    # default ``attn_impl='xla'`` path.
    if args.attn_impl == "splash":
        from nanochat_jax.gpt import set_splash_mesh
        set_splash_mesh(mesh)
    train_step = make_train_step_sharded(mesh, param_groups, optim_state)
    compute_grads_fn, accumulate_grads_fn, apply_update_fn = (None, None, None)
    if grad_accum_steps > 1:
        compute_grads_fn, accumulate_grads_fn, apply_update_fn = make_grad_accum_fns(
            mesh, param_groups, optim_state
        )
    if is_master:
        print(f"[info] mesh.devices.shape = {mesh.devices.shape} axis_names = {mesh.axis_names}")

    # 6. Batch generator — synthetic or real ClimbMix
    np_rng = np.random.default_rng(args.seed)
    train_loader = None
    val_loader = None
    tokenizer = None
    token_bytes = None

    if args.use_real_data or args.bpb_out is not None:
        # : import lazily so synthetic-only runs don't pay the deps
        from nanochat_jax.dataloader import ( # noqa: E402
            tokenizing_distributed_data_loader_bos_bestfit,
        )
        from nanochat_jax.tokenizer import get_token_bytes, get_tokenizer # noqa: E402

        tokenizer = get_tokenizer()
        if args.token_bytes_path is None:
            token_bytes_np = get_token_bytes()
        else:
            token_bytes_np = np.load(args.token_bytes_path)
        token_bytes = jnp.asarray(token_bytes_np)
        if is_master:
            print(
                f"[info] tokenizer vocab_size={tokenizer.get_vocab_size()} "
                f"token_bytes.shape={token_bytes.shape} "
                f"token_bytes.dtype={token_bytes.dtype}"
            )
            assert tokenizer.get_vocab_size() == config.vocab_size, (
                f"Tokenizer vocab_size {tokenizer.get_vocab_size()} != "
                f"model vocab_size {config.vocab_size}"
            )

    if args.use_real_data:
        # : per-process dataloader. Each process_index streams a disjoint
        # rank shard of train shards. Batch size per device, but
        # the global batch is device_batch_size × jax.device_count across all
        # data-parallel devices. So **each process** yields B = device_batch_size
        # × local_device_count rows per step (PT mirror).
        per_process_train_batch = args.device_batch_size * max(local_device_count, 1)
        train_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, per_process_train_batch, args.seq_len, split="train"
        )
        if is_master:
            print(
                f"[info] train_loader: split=train per_process_batch="
                f"{per_process_train_batch} seq_len={args.seq_len}"
            )

    # 7. Training loop.
    # Optional intermediate checkpoint + periodic val_bpb eval.
    if args.dry_run:
        if is_master:
            print(
                "[dry-run] All init complete (mesh, model, optim, dataloader). "
                "Exiting before the training loop."
            )
        return 0
    losses: list[float] = []
    t0 = time.time()
    if args.per_step_weights_out is not None and is_master:
        os.makedirs(args.per_step_weights_out, exist_ok=True)

    # MFU bookkeeping (Karpathy nanochat parity).
    num_flops_per_token = int(model.estimate_flops_per_token())
    peak_tflops_per_chip = get_peak_bf16_tflops()
    num_chips_total = max(jax.device_count(), 1)
    tokens_per_step = args.device_batch_size * args.seq_len * num_chips_total * grad_accum_steps
    if is_master:
        device_kind = jax.devices()[0].device_kind if jax.devices() else "unknown"
        print(
            f"[info] flops/token = {num_flops_per_token:.3e}, "
            f"peak BF16 = {peak_tflops_per_chip if peak_tflops_per_chip else '<unknown>'} TFLOPS/chip "
            f"({device_kind}), tokens/step = {tokens_per_step:,}"
        )

    # : resolve checkpoint_dir
    model_tag = args.model_tag or f"d{args.depth}_jax_seed{args.seed}"
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    elif (
        args.checkpoint_every > 0
        or args.resume_from_step >= 0
        or args.model_tag is not None
    ):
        checkpoint_dir = os.path.expanduser(
            f"~/.cache/nanochat-tpu/base_checkpoints/{model_tag}"
        )
    else:
        checkpoint_dir = None
    if checkpoint_dir is not None and is_master:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"[info] checkpoint_dir = {checkpoint_dir} (model_tag={model_tag})")

    # : resume from checkpoint (if requested)
    start_step = 0
    if args.resume_from_step >= 0:
        if checkpoint_dir is None:
            raise ValueError(
                "--resume-from-step requires --checkpoint-dir or --model-tag"
            )
        from nanochat_jax.checkpoint_manager import load_checkpoint
        if is_master:
            print(f"[info] Resuming from step {args.resume_from_step} in {checkpoint_dir}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            checkpoint_dir, args.resume_from_step, load_optimizer=False,
        )
        # Inject PT state_dict into NNX model (T07 fix per OPERATIONAL-TRAPS,
        # work 0037 Phase 0b). `inject_pt_linear_weight` is a single-Linear
        # helper, NOT a full-GPT injector — calling it on a GPT + dict raises
        # AttributeError at weight_converter.py:_torch_to_numpy. Use the public
        # `inject_pt_state_dict` from checkpoint_manager ( cascade verified
        # NNX `tree_flatten_with_path` + `nnx.update` path).
        from nanochat_jax.checkpoint_manager import inject_pt_state_dict
        from nanochat_jax.weight_converter import pt_state_dict_to_jax
        jax_dict = pt_state_dict_to_jax(model_data, target_dtype=None)
        inject_pt_state_dict(model, jax_dict)
        start_step = args.resume_from_step + 1
        if is_master:
            print(f"[info] Resumed: starting from step {start_step}")

    num_iter = 0 if args.eval_only else num_iterations

    # : periodic val_bpb eval setup (eval_steps + training_log)
    eval_steps_periodic = 0
    if args.eval_every > 0:
        eval_steps_periodic = max(
            args.eval_tokens // (
                args.device_batch_size * args.seq_len * max(jax.device_count(), 1)
            ),
            1,
        )
        if is_master:
            print(
                f"[info] eval_every={args.eval_every} eval_tokens={args.eval_tokens:,} "
                f"→ eval_steps_periodic={eval_steps_periodic}"
            )
    training_log_path = None
    if checkpoint_dir is not None and is_master:
        training_log_path = os.path.join(checkpoint_dir, "training_log.jsonl")

    def _eval_val_bpb_now(step: int) -> float | None:
        """ : periodic val_bpb measurement during training (master + all hosts)."""
        if not args.use_real_data:
            return None
        per_process_val_batch = args.device_batch_size * max(local_device_count, 1)
        val_loader_eval = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, per_process_val_batch, args.seq_len, split="val"
        )
        return float(evaluate_bpb(model, val_loader_eval, eval_steps_periodic, token_bytes))

    def _save_intermediate_checkpoint(step: int, val_bpb: float | None) -> None:
        """ : save intermediate checkpoint (master only writes meta + model)."""
        if checkpoint_dir is None:
            return
        from nanochat_jax.checkpoint_manager import save_checkpoint
        meta = {
            "step": step,
            "val_bpb": val_bpb,
            # T14 fix per OPERATIONAL-TRAPS (work 0036): model_config dict for
            # downstream consumers (base_eval, chat_cli, v5 resume).
            # Karpathy 1:1 mirror — see external/nanochat/scripts/base_train.py:486
            # `meta={"model_config": model_config_kwargs, ...}`.
            # compute_dtype + attn_impl omitted intentionally — set explicitly
            # at load. Backward-compat: deconstructed
            # args (depth/aspect_ratio/head_dim/...) preserved below.
            "model_config": {
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_kv_head": config.n_kv_head,
                "n_embd": config.n_embd,
                "vocab_size": config.vocab_size,
                "sequence_len": config.sequence_len,
                "window_pattern": config.window_pattern,
            },
            "depth": args.depth,
            "aspect_ratio": args.aspect_ratio,
            "head_dim": args.head_dim,
            "device_batch_size": args.device_batch_size,
            "max_seq_len": args.seq_len,
            "total_batch_size": total_batch_size_tokens,
            "num_iterations": num_iter,
            "matrix_lr": args.matrix_lr,
            "embedding_lr": args.embedding_lr,
            "unembedding_lr": args.unembedding_lr,
            "scalar_lr": args.scalar_lr,
            "weight_decay": args.weight_decay,
            "weight_decay_scaled": float(weight_decay_scaled),
            "warmup_steps": args.warmup_steps,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_frac": args.final_lr_frac,
            "seed": args.seed,
            "loop_state": {
                "min_val_bpb": min(losses) if losses else None,
                "smooth_train_loss": float(np.mean(losses[-100:])) if losses else None,
                "total_training_time": time.time() - t0,
            },
        }
        save_checkpoint(
            checkpoint_dir, step, model,
            optim_state=None, # cascade — optim shard not saved
            meta_data=meta, rank=process_idx,
        )
        if is_master:
            print(f"[info] Saved intermediate checkpoint at step {step}")
            # T06 fix per OPERATIONAL-TRAPS (work 0036) — rolling delete
            # to prevent disk full on long-running spot training. v4
            # cascade hit 16 ckpts × 5.5GB = 88GB / 100GB worker disk →
            # torch.save fail @ step 8000. keep_last=2 = ~11GB max disk.
            keep_last = max(int(getattr(args, "keep_last_checkpoints", 2)), 1)
            deleted = _rolling_delete_old_checkpoints(
                checkpoint_dir, keep_last=keep_last
            )
            if deleted > 0:
                print(
                    f"[info] Rolling delete: removed {deleted} old "
                    f"checkpoint(s), kept last {keep_last}"
                )

    for step in range(start_step, num_iter):
        step_start_time = time.time()
        # 7a. Batch + forward/backward (with gradient accumulation).
        global_batch_size = args.device_batch_size * max(jax.device_count(), 1)

        # 7b. Schedules (host-side Python floats → JAX scalars)
        lrm = jnp.float32(
            get_lr_multiplier(
                step,
                num_iterations,
                args.warmup_steps,
                args.warmdown_ratio,
                args.final_lr_frac,
            )
        )
        mom = jnp.float32(
            get_muon_momentum(step, num_iterations, args.warmdown_ratio)
        )
        wd = jnp.float32(
            get_weight_decay(step, num_iterations, weight_decay_scaled)
        )

        if grad_accum_steps <= 1:
            # Fast path: no accumulation (same as before)
            if args.use_real_data:
                inputs_np, targets_np = next(train_loader)
                idx_np = inputs_np
            else:
                idx_np, targets_np = make_synthetic_batch(
                    np_rng, global_batch_size, args.seq_len, config.vocab_size
                )
            idx = jnp.asarray(idx_np)
            targets = jnp.asarray(targets_np)
            loss, optim_state = train_step(
                model, optim_state, idx, targets, lrm, mom, wd
            )
        else:
            # Gradient accumulation: PT base_train.py:510-518 mirror.
            # Memory-efficient: first micro-step via compute_grads_fn,
            # subsequent micro-steps via accumulate_grads_fn (donate_argnums=1
            # reuses acc_grads buffer in jit, no Python-level double allocation).
            total_loss = 0.0
            acc_grads = None
            for micro_step in range(grad_accum_steps):
                if args.use_real_data:
                    inputs_np, targets_np = next(train_loader)
                    idx_np = inputs_np
                else:
                    idx_np, targets_np = make_synthetic_batch(
                        np_rng, global_batch_size, args.seq_len, config.vocab_size
                    )
                idx = jnp.asarray(idx_np)
                targets = jnp.asarray(targets_np)
                if acc_grads is None:
                    micro_loss, acc_grads = compute_grads_fn(model, idx, targets)
                else:
                    micro_loss, acc_grads = accumulate_grads_fn(
                        model, acc_grads, idx, targets,
                    )
                total_loss += float(micro_loss)
            # Average gradients (PT normalizes loss; equivalent to averaging grads)
            inv_accum = jnp.float32(1.0 / grad_accum_steps)
            acc_grads = jax.tree.map(lambda g: g * inv_accum, acc_grads)
            loss = jnp.float32(total_loss / grad_accum_steps)
            optim_state = apply_update_fn(
                model, optim_state, acc_grads, lrm, mom, wd
            )

        # 7d. NaN guard
        loss_val = float(loss)
        if not np.isfinite(loss_val):
            raise ValueError(
                f"Non-finite loss at step {step}: {loss_val} (NaN/Inf — "
                f" Catastrophic, § reset criteria)"
            )
        if loss_val > 100.0:
            raise ValueError(
                f"Loss explosion at step {step}: {loss_val} > 100 "
                f""
            )
        losses.append(loss_val)

        # 7e. Per-step weights save (M6 metric, master only). Step 0 + final
        # always saved; intermediate steps every --save-every steps.
        if args.per_step_weights_out is not None and is_master:
            save_this_step = (
                step == 0
                or step == num_iterations - 1
                or (args.save_every > 0 and step % args.save_every == 0)
            )
            if save_this_step:
                step_path = os.path.join(
                    args.per_step_weights_out, f"step_{step:04d}.npz"
                )
                save_weights(model, step_path)

        # 7f. Logging (master only, )
        step_dt = time.time() - step_start_time
        if is_master and (step % args.log_every == 0 or step == num_iterations - 1):
            elapsed = time.time() - t0
            mfu_str = ""
            # Skip step 0 (JIT compile dominates) — real MFU starts at step 1.
            if step > 0 and peak_tflops_per_chip is not None:
                mfu = compute_mfu(
                    num_flops_per_token=num_flops_per_token,
                    tokens_per_step=tokens_per_step,
                    step_seconds=step_dt,
                    peak_tflops_per_chip=peak_tflops_per_chip,
                    num_chips=num_chips_total,
                )
                mfu_str = f" dt={step_dt*1000:.0f}ms mfu={mfu*100:.1f}%"
            print(
                f"[step {step:4d}] loss={loss_val:.6f} "
                f"lrm={float(lrm):.4f} mom={float(mom):.4f} wd={float(wd):.4f} "
                f"elapsed={elapsed:.1f}s{mfu_str}",
                flush=True,
            )

        # 7g. : checkpoint save BEFORE eval — ckpt safe even if eval crashes.
        # Save at: last_step OR (step > 0 AND step != resume_from_step AND
        # checkpoint_every > 0 AND step % checkpoint_every == 0). PT 1:1 mirror.
        is_last_step = step == num_iterations - 1
        save_this = (
            is_last_step
            or (
                step > 0
                and step != args.resume_from_step
                and args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
            )
        )
        if save_this and checkpoint_dir is not None:
            # val_bpb not yet available — eval runs after save. Recorded in
            # training_log.jsonl after eval completes; meta.json val_bpb = null.
            _save_intermediate_checkpoint(step, val_bpb=None)

        # 7h. : periodic val_bpb measurement AFTER checkpoint save.
        # NOTE (work 0036 Phase 6 cascade): initial eval at step 0 triggers a
        # massive JIT compile (~50min for d24+B=8+Splash+multi-host) since the
        # eval graph is structurally different from the train graph. Skip eval
        # at step 0 — the first eval lands at step==eval_every.
        # --no-final-eval suppresses the is_last_step branch only; if last step
        # happens to align with the periodic schedule, that path still fires.
        val_bpb_periodic = None
        fire_periodic = (
            args.eval_every > 0 and step > 0 and (step % args.eval_every == 0)
        )
        fire_final = (
            args.eval_every > 0 and is_last_step and not args.no_final_eval
        )
        if fire_periodic or fire_final:
            eval_t0 = time.time()
            val_bpb_periodic = _eval_val_bpb_now(step)
            eval_elapsed = time.time() - eval_t0
            if is_master and val_bpb_periodic is not None:
                print(
                    f"[step {step:4d}] val_bpb={val_bpb_periodic:.6f} "
                    f"(eval_steps={eval_steps_periodic} elapsed={eval_elapsed:.1f}s)",
                    flush=True,
                )
                if training_log_path is not None:
                    with open(training_log_path, "a") as f:
                        f.write(json.dumps({
                            "step": step,
                            "loss": loss_val,
                            "val_bpb": val_bpb_periodic,
                            "lrm": float(lrm),
                            "mom": float(mom),
                            "wd": float(wd),
                            "elapsed": time.time() - t0,
                        }) + "\n")

    # 8. Final weights save
    if args.weights_out is not None and is_master:
        save_weights(model, args.weights_out)
        print(f"[info] Saved final weights to {args.weights_out}")

    if is_master and num_iter > 0:
        elapsed = time.time() - t0
        print(
            f"[done] {num_iter} steps in {elapsed:.1f}s "
            f"(avg {elapsed / max(num_iter, 1):.3f}s/step)"
        )
        print(f"[done] losses[:5] = {losses[:5]}")
        print(f"[done] losses[-5:] = {losses[-5:]}")
        print(f"[done] final loss = {losses[-1]:.6f}")

    # 9. : val BPB measurement (M7 trigger). Runs on master + all hosts;
    # ``evaluate_bpb`` does the cross-host all-reduce internally
    # (loss_eval.py:196+). PT mirror: base_train.py:421-435 final eval.
    if args.bpb_out is not None:
        per_process_val_batch = args.device_batch_size * max(local_device_count, 1)
        val_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, per_process_val_batch, args.seq_len, split="val"
        )
        if is_master:
            print(
                f"[info] val_loader: split=val per_process_batch="
                f"{per_process_val_batch} seq_len={args.seq_len} "
                f"eval_steps={args.eval_steps}"
            )

        eval_t0 = time.time()
        val_bpb = evaluate_bpb(model, val_loader, args.eval_steps, token_bytes)
        eval_elapsed = time.time() - eval_t0

        if is_master:
            print(
                f"[done] val_bpb={val_bpb:.6f} "
                f"(eval_steps={args.eval_steps} elapsed={eval_elapsed:.1f}s)"
            )
            result = {
                "val_bpb": val_bpb,
                "num_iterations": num_iter,
                "eval_steps": args.eval_steps,
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.seq_len,
                "seed": args.seed,
                "depth": args.depth,
                "vocab_size": int(config.vocab_size),
                "dtype": str(config.compute_dtype),
                "process_count": int(jax.process_count()),
                "device_count": int(jax.device_count()),
                "use_real_data": bool(args.use_real_data),
                "eval_only": bool(args.eval_only),
                "wall_time_train_s": (time.time() - t0 - eval_elapsed) if num_iter > 0 else 0.0,
                "wall_time_eval_s": eval_elapsed,
                "final_train_loss": losses[-1] if losses else None,
            }
            with open(args.bpb_out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[info] Saved val BPB result to {args.bpb_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
