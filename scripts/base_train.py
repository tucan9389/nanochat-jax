"""Train a nanochat-style base model on TPU or CPU.

This entry point keeps the public training contract close to upstream
``nanochat/scripts/base_train.py`` while adding JAX/TPU-specific execution
controls for sharding, checkpointing, gradient accumulation, Splash Attention,
and precision.

Usage::

    # CPU dry run
    python scripts/base_train.py --depth=2 --num-iterations=2 \\
        --device-batch-size=2 --seq-len=128 --no-distributed

    # Single-host TPU smoke
    python scripts/base_train.py --depth=12 --num-iterations=100 \\
        --device-batch-size=4 --use-real-data --model-tag=d12_smoke
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
    make_fused_train_step,
    make_grad_accum_fns,
    make_train_step_sharded,
)
from nanochat_jax.base_train_config import ( # noqa: E402
    D12_SCALING_PARAMS,
    compute_batch_lr_scale,
    compute_total_batch_size_tokens,
    compute_weight_decay_scaled,
    make_config,
    make_d12_config,
    resolve_num_iterations,
)
from nanochat_jax.common import get_base_dir, setup_distributed_env_vars # noqa: E402
from nanochat_jax.gpt import GPT # noqa: E402
from nanochat_jax.grad_utils import nnx_state_to_flat_dict # noqa: E402
from nanochat_jax.loss_eval import evaluate_bpb # noqa: E402 —
from nanochat_jax.perf import compute_mfu, get_peak_bf16_tflops # noqa: E402
from nanochat_jax.report import log_report_safe # noqa: E402
from nanochat_jax.sharding import get_process_info, make_mesh # noqa: E402


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
    ``meta_<step:06d>.json`` and ``optim_<step:06d>_rank*.pt`` files.
    Idempotent: no-op if fewer than ``keep_last``. Master-only call.
    Defensive against concurrent deletion races (OSError -> continue).

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
        optim_prefix = f"optim_{_step:06d}_rank"
        for optim_fname in os.listdir(checkpoint_dir):
            if not optim_fname.startswith(optim_prefix):
                continue
            if not optim_fname.endswith(".pt"):
                continue
            try:
                os.remove(os.path.join(checkpoint_dir, optim_fname))
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
        help="Attention implementation. "
        "'xla' (default) = jax.nn.dot_product_attention(implementation='xla') "
        "; 'splash' = Splash Attention TPU kernel.",
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
        "Use >1 when per-device batch is smaller than the target token batch.",
    )
    parser.add_argument(
        "--grad-accum-impl",
        type=str,
        default="loop",
        choices=["loop", "fused"],
        help="Gradient accumulation implementation. 'loop' keeps the existing "
        "per-microbatch JIT path; 'fused' scans microbatches and applies the "
        "optimizer inside one JIT call.",
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
        help="Optional final weights .npz path.",
    )
    parser.add_argument(
        "--per-step-weights-out",
        type=str,
        default=None,
        help="Optional dir for per-step weights .npz snapshots.",
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
    parser.add_argument(
        "--cast-embeddings-bf16",
        action="store_true",
        default=False,
        help="Store token and value embeddings in compute dtype after init. "
        "In bf16 mode this keeps embeddings and optimizer state in the same "
        "precision regime as the PyTorch reference path.",
    )
    parser.add_argument(
        "--polar-express-bf16",
        action="store_true",
        default=False,
        help="Cast Muon gradients to bf16 before Polar Express Newton-Schulz "
        "iterations. Default off keeps Newton-Schulz in fp32.",
    )
    # TPU default fp32 matmul can use bf16 internal accumulation. ``highest``
    # requests a slower, higher-accuracy accumulation path when needed.
    parser.add_argument(
        "--matmul-precision",
        type=str,
        default=None,
        choices=[None, "default", "high", "highest"],
        help="Override JAX default matmul precision (jax.config update). "
        "TPU default = bf16 internal acc (Fastest). 'highest' = 6-pass bf16 "
        "= near-fp32 acc (~2.5x slower).",
    )
    parser.add_argument(
        "--lm-head-precision",
        type=str,
        default=None,
        choices=[None, "default", "high", "highest"],
        help="Override only the lm_head dot_general precision while leaving "
        "global matmul precision unchanged.",
    )
    parser.add_argument(
        "--splash-block-q",
        type=int,
        default=None,
        help="Splash Attention block_q override. None keeps Splash defaults.",
    )
    parser.add_argument(
        "--splash-block-kv",
        type=int,
        default=None,
        help="Splash Attention block_kv override. None keeps Splash defaults.",
    )
    parser.add_argument(
        "--splash-block-kv-compute",
        type=int,
        default=None,
        help="Splash Attention block_kv_compute override. None uses min(block_kv, 128).",
    )
    parser.add_argument(
        "--ve-grad-impl",
        type=str,
        default="scatter",
        choices=["scatter", "onehot", "segsum", "segsum_fp32"],
        help="Value-embedding backward implementation. Default scatter is the "
        "baseline; onehot can improve TPU throughput at large vocab/sequence shapes.",
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
        "(default: synthetic). Required for meaningful quality evaluation.",
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
        "preemption safety.",
    )
    parser.add_argument(
        "--keep-last-checkpoints", type=int, default=2,
        help="Keep only the last N intermediate checkpoints. Default 2 bounds "
        "disk usage during long spot runs.",
    )
    parser.add_argument(
        "--resume-from-step", type=int, default=-1,
        help=" : resume training from step N checkpoint. -1 = disabled (start "
        "fresh). Loads model + optim_state + dataloader_state + loop_state from "
        "checkpoint_dir.",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        default=-1,
        help="Stop after completing this train step. -1 disables.",
    )
    parser.add_argument(
        "--no-final-checkpoint",
        action="store_true",
        help="Suppress the final/stop checkpoint. Periodic --checkpoint-every "
        "checkpoints still fire.",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help=" : directory for intermediate + final checkpoints. Default = "
        "get_base_dir()/base_checkpoints/{model_tag} if model_tag given.",
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
        "base_eval.py pass runs after training.",
    )
    parser.add_argument(
        "--token-bytes-path",
        type=str,
        default=None,
        help=": explicit path to token_bytes.npy. If None, uses "
        "``nanochat_jax.tokenizer.get_token_bytes`` (default tokenizer dir).",
    )
    args = parser.parse_args()

    # 0a. Matmul precision override. MUST apply BEFORE
    # jax.distributed.initialize / jax.devices so all subsequent compilation
    # respects the precision setting.
    if args.matmul_precision is not None:
        jax.config.update("jax_default_matmul_precision", args.matmul_precision)
        print(f"[info] jax_default_matmul_precision = {args.matmul_precision}")

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
        print(
            f"[info] attn_impl={args.attn_impl} "
            f"lm_head_precision={args.lm_head_precision} "
            f"splash_blocks=({args.splash_block_q}, {args.splash_block_kv}, "
            f"{args.splash_block_kv_compute}) "
            f"ve_grad_impl={args.ve_grad_impl}"
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
            lm_head_precision=args.lm_head_precision,
            splash_block_q=args.splash_block_q,
            splash_block_kv=args.splash_block_kv,
            splash_block_kv_compute=args.splash_block_kv_compute,
            ve_grad_impl=args.ve_grad_impl,
        )
    else:
        # / backward-compat (n_embd=768 hardcoded)
        config = make_d12_config(
            depth=args.depth,
            sequence_len=args.seq_len,
            vocab_size=args.vocab_size,
            bf16=args.bf16,
            attn_impl=args.attn_impl,
            lm_head_precision=args.lm_head_precision,
            splash_block_q=args.splash_block_q,
            splash_block_kv=args.splash_block_kv,
            splash_block_kv_compute=args.splash_block_kv_compute,
            ve_grad_impl=args.ve_grad_impl,
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
            f"[info] cast_embeddings_bf16=ON — wte.dtype = {wte_dtype}"
        )

    # 4. total_batch_size + Karpathy-style LR/WD muP scaling → param_groups + optim_state
    # Batch size must be known before optimizer init (weight decay depends on it).
    grad_accum_steps = max(int(args.grad_accum_steps), 1)
    total_batch_size_tokens = compute_total_batch_size_tokens(
        total_batch_size=args.total_batch_size,
        device_batch_size=args.device_batch_size,
        seq_len=args.seq_len,
        device_count=jax.device_count(),
        grad_accum_steps=grad_accum_steps,
    )

    # Karpathy muP scaling (PT base_train.py:287-315):
    # batch_lr_scale = sqrt(B / B_ref) — η ∝ √(B/B_ref) for AdamW/Muon
    # weight_decay_scaled = wd × sqrt(B/B_ref) × (D_ref / D_target)
    # where D_ref/D_target = d12_scaling_params / d_X_scaling_params (ratio cancels)
    # This uses the T_epoch framework: λ ∝ √(B/B_ref) × (D_ref/D).
    _scaling_counts = model.num_scaling_params()
    _d_x_scaling = _scaling_counts["transformer_matrices"] + _scaling_counts["lm_head"]
    # Standard d12 reference (depth=12, aspect_ratio=64, head_dim=128, vocab=32768):
    # transformer_matrices=84,935,088 + lm_head=25,165,824 = 110,100,912
    _batch_lr_scale = compute_batch_lr_scale(total_batch_size_tokens)
    weight_decay_scaled = compute_weight_decay_scaled(
        weight_decay=args.weight_decay,
        total_batch_size_tokens=total_batch_size_tokens,
        d_x_scaling_params=_d_x_scaling,
    )

    n_params = None
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
            f"(×{D12_SCALING_PARAMS / _d_x_scaling:.4f} d12/model ratio)"
        )

    polar_express_dtype = (
        config.compute_dtype if args.polar_express_bf16 else None
    )
    if is_master and polar_express_dtype is not None:
        print(
            f"[info] polar_express_bf16=ON — Muon NS dtype = {polar_express_dtype}"
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
    num_iterations, horizon_source = resolve_num_iterations(
        num_iterations=args.num_iterations,
        target_param_data_ratio=args.target_param_data_ratio,
        scaling_params=_d_x_scaling,
        total_batch_size_tokens=total_batch_size_tokens,
    )

    if is_master:
        print(
            f"[info] num_iterations = {num_iterations:,} "
            f"(total_batch_size = {total_batch_size_tokens:,} tokens, "
            f"grad_accum_steps = {grad_accum_steps}, "
            f"grad_accum_impl = {args.grad_accum_impl}, source = {horizon_source})"
        )

    # 5. Mesh + sharded train_step factory
    mesh = make_mesh() # default: jax.device_count
    # Register the mesh before jit-tracing the train_step so Splash Attention
    # can capture the multi-chip shard_map mesh as a closed-over constant.
    # ``set_splash_mesh(None)`` is a no-op for the default ``attn_impl='xla'``.
    if args.attn_impl == "splash":
        from nanochat_jax.gpt import set_splash_mesh
        set_splash_mesh(mesh)
    train_step = make_train_step_sharded(mesh, param_groups, optim_state)
    compute_grads_fn, accumulate_grads_fn, apply_update_fn = (None, None, None)
    fused_step_fn = None
    fused_state = None
    fused_state_dirty = False
    if grad_accum_steps > 1 and args.grad_accum_impl == "fused":
        if is_master:
            print(
                "[info] fused grad accumulation enabled "
                "(single jax.jit + lax.scan + optimizer update)"
            )
        fused_step_fn, fused_state = make_fused_train_step(
            mesh,
            param_groups,
            optim_state,
            model,
            grad_accum_steps=grad_accum_steps,
        )
    elif grad_accum_steps > 1:
        compute_grads_fn, accumulate_grads_fn, apply_update_fn = make_grad_accum_fns(
            mesh, param_groups, optim_state
        )
    if is_master:
        print(f"[info] mesh.devices.shape = {mesh.devices.shape} axis_names = {mesh.axis_names}")

    # 6. Batch generator — synthetic or real ClimbMix
    np_rng = np.random.default_rng(args.seed)
    train_loader = None
    train_loader_state = None
    make_train_loader = None
    val_loader = None
    tokenizer = None
    token_bytes = None

    if args.use_real_data or args.bpb_out is not None:
        # : import lazily so synthetic-only runs don't pay the deps
        from nanochat_jax.dataloader import ( # noqa: E402
            tokenizing_distributed_data_loader_bos_bestfit,
            tokenizing_distributed_data_loader_with_state_bos_bestfit,
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

        def make_train_loader(resume_state_dict=None):
            nonlocal train_loader_state
            loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
                tokenizer,
                per_process_train_batch,
                args.seq_len,
                split="train",
                resume_state_dict=resume_state_dict,
            )
            for inputs, targets, state_dict in loader:
                train_loader_state = dict(state_dict)
                yield inputs, targets

        train_loader = make_train_loader()
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
            log_report_safe(
                section="Base model training",
                data=[
                    vars(args),
                    {
                        "dry_run": True,
                        "Number of parameters": n_params,
                        "Calculated number of iterations": num_iterations,
                        "Number of training tokens": total_batch_size_tokens * num_iterations,
                        "Total batch size": total_batch_size_tokens,
                        "JAX world size": int(jax.process_count()),
                        "JAX device count": int(jax.device_count()),
                        "checkpoint_dir": None,
                    },
                ],
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
        checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", model_tag)
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
            checkpoint_dir,
            args.resume_from_step,
            load_optimizer=True,
            rank=process_idx,
        )
        if optimizer_data is None:
            raise FileNotFoundError(
                "Optimizer checkpoint is required for --resume-from-step but "
                f"was not found for rank {process_idx} at step {args.resume_from_step}"
            )
        # Inject the PyTorch-style state_dict into the NNX model. The linear
        # layer helper is intentionally not used here; full-model injection goes
        # through checkpoint_manager so GPT module paths are mapped correctly.
        from nanochat_jax.checkpoint_manager import inject_pt_state_dict
        from nanochat_jax.weight_converter import pt_state_dict_to_jax
        jax_dict = pt_state_dict_to_jax(model_data, target_dtype=None)
        inject_pt_state_dict(model, jax_dict)
        from nanochat_jax.checkpoint_manager import _pt_dict_to_muon_adamw_state
        optim_state = _pt_dict_to_muon_adamw_state(optimizer_data)
        if args.use_real_data:
            resume_train_loader_state = (
                meta_data.get("train_loader_state")
                or meta_data.get("dataloader_state")
            )
            if resume_train_loader_state is None or make_train_loader is None:
                raise KeyError(
                    "train_loader_state is required for real-data --resume-from-step"
                )
            train_loader = make_train_loader(resume_train_loader_state)
            if is_master:
                print(
                    "[info] Resumed train loader state "
                    f"{resume_train_loader_state}"
                )
        else:
            synthetic_rng_state = meta_data.get("synthetic_rng_state")
            if synthetic_rng_state is None:
                raise KeyError(
                    "synthetic_rng_state is required for synthetic --resume-from-step"
                )
            np_rng.bit_generator.state = synthetic_rng_state
        start_step = args.resume_from_step + 1
        if is_master:
            print(f"[info] Resumed: starting from step {start_step}")

        train_step = make_train_step_sharded(mesh, param_groups, optim_state)
        compute_grads_fn, accumulate_grads_fn, apply_update_fn = (None, None, None)
        fused_step_fn = None
        fused_state = None
        fused_state_dirty = False
        if grad_accum_steps > 1 and args.grad_accum_impl == "fused":
            fused_step_fn, fused_state = make_fused_train_step(
                mesh,
                param_groups,
                optim_state,
                model,
                grad_accum_steps=grad_accum_steps,
            )
        elif grad_accum_steps > 1:
            compute_grads_fn, accumulate_grads_fn, apply_update_fn = make_grad_accum_fns(
                mesh, param_groups, optim_state
            )

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
            # model_config dict for downstream consumers such as base_eval,
            # chat_cli, and resume paths. Karpathy 1:1 mirror:
            # external/nanochat/scripts/base_train.py:486
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
                "attn_impl": config.attn_impl,
                "lm_head_precision": config.lm_head_precision,
                "splash_block_q": config.splash_block_q,
                "splash_block_kv": config.splash_block_kv,
                "splash_block_kv_compute": config.splash_block_kv_compute,
                "ve_grad_impl": config.ve_grad_impl,
            },
            "depth": args.depth,
            "aspect_ratio": args.aspect_ratio,
            "head_dim": args.head_dim,
            "device_batch_size": args.device_batch_size,
            "max_seq_len": args.seq_len,
            "total_batch_size": total_batch_size_tokens,
            "grad_accum_steps": grad_accum_steps,
            "grad_accum_impl": args.grad_accum_impl,
            "num_iterations": num_iter,
            "executed_num_iterations": step + 1,
            "stop_after_step": args.stop_after_step,
            "last_executed_step": step,
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
            "train_loader_state": train_loader_state,
            "synthetic_rng_state": (
                np_rng.bit_generator.state if not args.use_real_data else None
            ),
            "loop_state": {
                "min_val_bpb": min(losses) if losses else None,
                "smooth_train_loss": float(np.mean(losses[-100:])) if losses else None,
                "total_training_time": time.time() - t0,
            },
        }
        save_checkpoint(
            checkpoint_dir, step, model,
            optim_state=optim_state,
            meta_data=meta, rank=process_idx,
        )
        if is_master:
            print(f"[info] Saved intermediate checkpoint at step {step}")
            # Keep only the newest local checkpoints so long spot runs do not
            # fill the TPU VM boot disk. Durable copies should be synced to GCS.
            keep_last = max(int(getattr(args, "keep_last_checkpoints", 2)), 1)
            deleted = _rolling_delete_old_checkpoints(
                checkpoint_dir, keep_last=keep_last
            )
            if deleted > 0:
                print(
                    f"[info] Rolling delete: removed {deleted} old "
                    f"checkpoint(s), kept last {keep_last}"
                )

    def _sync_fused_model_state() -> None:
        nonlocal fused_state_dirty
        if fused_state_dirty and fused_state is not None:
            nnx.update(model, fused_state)
            fused_state_dirty = False

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

        if grad_accum_steps > 1 and args.grad_accum_impl == "fused":
            if args.use_real_data:
                idx_chunks = []
                target_chunks = []
                for _ in range(grad_accum_steps):
                    inputs_np, targets_np = next(train_loader)
                    idx_chunks.append(np.array(inputs_np, copy=True))
                    target_chunks.append(np.array(targets_np, copy=True))
                idx_all = jnp.asarray(np.stack(idx_chunks, axis=0))
                targets_all = jnp.asarray(np.stack(target_chunks, axis=0))
            else:
                idx_np, targets_np = make_synthetic_batch(
                    np_rng,
                    global_batch_size * grad_accum_steps,
                    args.seq_len,
                    config.vocab_size,
                )
                idx_all = jnp.asarray(
                    idx_np.reshape(grad_accum_steps, global_batch_size, args.seq_len)
                )
                targets_all = jnp.asarray(
                    targets_np.reshape(grad_accum_steps, global_batch_size, args.seq_len)
                )
            fused_state, optim_state, loss = fused_step_fn(
                fused_state,
                optim_state,
                idx_all,
                targets_all,
                lrm,
                mom,
                wd,
            )
            fused_state_dirty = True
        elif grad_accum_steps <= 1:
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
            _sync_fused_model_state()
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
        stop_requested = args.stop_after_step >= 0 and step >= args.stop_after_step
        save_this = (
            ((is_last_step or stop_requested) and not args.no_final_checkpoint)
            or (
                step > 0
                and step != args.resume_from_step
                and args.checkpoint_every > 0
                and step % args.checkpoint_every == 0
            )
        )
        if save_this and checkpoint_dir is not None:
            _sync_fused_model_state()
            # val_bpb not yet available — eval runs after save. Recorded in
            # training_log.jsonl after eval completes; meta.json val_bpb = null.
            _save_intermediate_checkpoint(step, val_bpb=None)

        # 7h. : periodic val_bpb measurement AFTER checkpoint save.
        # Initial eval at step 0 triggers a large JIT compile because the eval
        # graph is structurally different from the train graph. Skip eval at
        # step 0; the first eval lands at step == eval_every.
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
            _sync_fused_model_state()
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

        if stop_requested:
            if is_master:
                print(f"[info] stop_after_step={args.stop_after_step} reached")
            break

    _sync_fused_model_state()

    # 8. Final weights save
    if args.weights_out is not None and is_master:
        save_weights(model, args.weights_out)
        print(f"[info] Saved final weights to {args.weights_out}")

    if is_master and num_iter > 0:
        elapsed = time.time() - t0
        executed_steps = len(losses)
        print(
            f"[done] {executed_steps} executed steps in {elapsed:.1f}s "
            f"(avg {elapsed / max(executed_steps, 1):.3f}s/step)"
        )
        print(f"[done] losses[:5] = {losses[:5]}")
        print(f"[done] losses[-5:] = {losses[-5:]}")
        print(f"[done] final loss = {losses[-1]:.6f}")

    # 9. : val BPB measurement (M7 trigger). Runs on master + all hosts;
    # ``evaluate_bpb`` does the cross-host all-reduce internally
    # (loss_eval.py:196+). PT mirror: base_train.py:421-435 final eval.
    final_bpb_report = None
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
                "resume_from_step": (
                    int(args.resume_from_step)
                    if int(args.resume_from_step) >= 0
                    else None
                ),
                "start_step": int(start_step),
                "executed_steps_this_process": len(losses),
                "wall_time_train_s": (time.time() - t0 - eval_elapsed) if losses else 0.0,
                "wall_time_eval_s": eval_elapsed,
                "final_train_loss": losses[-1] if losses else None,
            }
            with open(args.bpb_out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"[info] Saved val BPB result to {args.bpb_out}")
            final_bpb_report = result

    if is_master:
        total_elapsed = time.time() - t0
        executed_steps = len(losses)
        last_completed_step = start_step + executed_steps - 1 if executed_steps else start_step - 1
        represented_steps = max(last_completed_step + 1, 0)
        report_setup = {
            "dry_run": False,
            "model_tag": model_tag,
            "checkpoint_dir": checkpoint_dir,
            "Number of parameters": n_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Resume from step": (
                int(args.resume_from_step)
                if int(args.resume_from_step) >= 0
                else None
            ),
            "Start step": start_step,
            "Last completed step": last_completed_step,
            "Executed number of iterations": executed_steps,
            "Total model iterations represented": represented_steps,
            "Training tokens executed this process": total_batch_size_tokens * executed_steps,
            "Training tokens represented by checkpoint": total_batch_size_tokens * represented_steps,
            "Tokens : Scaling params ratio": (
                (total_batch_size_tokens * represented_steps) / _d_x_scaling
                if _d_x_scaling
                else None
            ),
            "JAX world size": int(jax.process_count()),
            "JAX device count": int(jax.device_count()),
            "warmup_steps": args.warmup_steps,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_frac": args.final_lr_frac,
        }
        report_outcome = {
            "Final train loss": losses[-1] if losses else None,
            "Total training time s": total_elapsed,
            "Average step time s": (
                total_elapsed / max(executed_steps, 1)
                if executed_steps
                else None
            ),
        }
        if final_bpb_report is not None:
            report_outcome["Final validation bpb"] = final_bpb_report.get("val_bpb")
        log_report_safe(
            section="Base model training",
            data=[vars(args), report_setup, report_outcome],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
