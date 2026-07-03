"""Supervised Fine-Tuning entry point.

Loads a pretrained base model, builds an SFT data mixture, and trains the
model on conversation-formatted data with a mask that supervises only the
assistant's responses. Uses ``--sft-scope`` to choose between mixtures of
varying size.

Scopes (``--sft-scope``):
- ``full`` (default): SmolTalk + CustomJSON x2 + MMLU x N + GSM8K x N +
  SimpleSpelling + SpellingBee (~1M rows). Mirrors upstream nanochat.
- ``reduced``: smaller subsets of the above (~150K rows) for cheaper TPU runs.
- ``min``: ~12K rows for Mac CPU sanity.

Usage::

    # Mac CPU 5-step sanity (--chatcore-every -1 skips the expensive
    # generative ChatCORE eval that otherwise still fires on the last step)
    python -m scripts.chat_sft --model-tag d12_jax_seed42 --num-iterations 5 \\
        --device-batch-size 2 --max-seq-len 256 --no-distributed --chatcore-every -1

    # Short TPU run
    python -m scripts.chat_sft --model-tag d12_jax_seed42 --num-iterations 100 \\
        --device-batch-size 4 --max-seq-len 1024 --bf16
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

# Allow `python scripts/chat_sft.py` from the project root .
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from nanochat_jax.train_core import ( # noqa: E402 — sys.path mutation above
    init_train_state,
    make_grad_accum_fns,
    make_train_step_sharded,
)
from nanochat_jax.checkpoint_manager import ( # noqa: E402
    load_model,
    load_optimizer_state,
    save_checkpoint,
)
from nanochat_jax.common import ( # noqa: E402
    download_file_with_lock,
    get_base_dir,
    setup_distributed_env_vars,
)
from nanochat_jax.loss_eval import evaluate_bpb # noqa: E402
from nanochat_jax.report import log_report_safe # noqa: E402
from nanochat_jax.sharding import make_mesh # noqa: E402
from nanochat_jax.tasks import CustomJSON, SmolTalk, TaskMixture # noqa: E402
from nanochat_jax.tasks.gsm8k import GSM8K # noqa: E402 — lazy path
from nanochat_jax.tasks.mmlu import MMLU # noqa: E402 — lazy path
from nanochat_jax.tasks.spellingbee import SimpleSpelling, SpellingBee # noqa: E402
from nanochat_jax.tokenizer import get_token_bytes # noqa: E402

if TYPE_CHECKING:
    from nanochat_jax.tasks import Task


# Identity conversations URL (auto-download path)
IDENTITY_CONVERSATIONS_URL = (
    "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
)


# -----------------------------------------------------------------------------
# DummyWandb
# -----------------------------------------------------------------------------


class DummyWandb:
    """No-op logging stub (mirrors upstream's DummyWandb shape).

    This port does not integrate wandb; `wandb_run.log({...})` and
    `wandb_run.finish()` are always no-ops.
    """

    def log(self, _data: dict) -> None: # noqa: D401
        pass

    def finish(self) -> None: # noqa: D401
        pass


# -----------------------------------------------------------------------------
# Master logger
# -----------------------------------------------------------------------------


def _print0(message: str) -> None:
    """Master-only print."""
    if jax.process_index() == 0:
        print(message, flush=True)


def capture_attention_impl_state(model) -> list[tuple[object, str, object]]:
    """Capture attention impl fields that ChatEval may need to switch to XLA."""
    state: list[tuple[object, str, object]] = []
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "attn_impl"):
        state.append((config, "attn_impl", getattr(config, "attn_impl")))

    transformer = getattr(model, "transformer", None)
    for block in getattr(transformer, "h", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "attn_impl"):
            state.append((attn, "attn_impl", getattr(attn, "attn_impl")))
    return state


def set_attention_impl_for_chat_eval(model) -> list[tuple[object, str, object]]:
    """Temporarily use XLA attention for variable-length ChatCORE prompts."""
    state = capture_attention_impl_state(model)
    for obj, attr, old_value in state:
        if old_value == "splash":
            setattr(obj, attr, "xla")
    return state


def restore_attention_impl_state(state: list[tuple[object, str, object]]) -> None:
    """Restore attention impl fields captured by ``set_attention_impl_for_chat_eval``."""
    for obj, attr, old_value in state:
        setattr(obj, attr, old_value)


# -----------------------------------------------------------------------------
# Schedules
# -----------------------------------------------------------------------------


def get_lr_multiplier_progress(
    progress: float,
    *,
    warmup_ratio: float = 0.0,
    warmdown_ratio: float = 0.5,
    final_lr_frac: float = 0.0,
) -> float:
    """Linear warmup + constant + linear warmdown (mirrors upstream chat_sft.py).

    Uses ``progress`` (0→1 normalized) instead of step-indexed (upstream uses
    ``progress = consumed / dataset_size`` since num_iterations may be unknown).

    Args:
        progress: 0→1 normalized progress through the epoch.
        warmup_ratio: fraction of total progress used for linear warmup.
        warmdown_ratio: fraction of total progress used for linear warmdown.
        final_lr_frac: final learning rate as fraction of base lr.

    Returns:
        learning rate multiplier in [0, 1].
    """
    if progress < warmup_ratio:
        return (progress + 1e-8) / warmup_ratio
    elif progress <= 1.0 - warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - warmdown_ratio)) / warmdown_ratio
        return (1 - decay) * 1.0 + decay * final_lr_frac


def get_muon_momentum_sft(it: int) -> float:
    """SFT Muon momentum schedule (mirrors upstream chat_sft.py).

    Simple linear interpolation from 0.85 to 0.95 over first 300 steps,
    then constant 0.95.

    Note: differs from train_core.get_muon_momentum (which has 3-phase warmdown).
    """
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


# -----------------------------------------------------------------------------
# Batch semantics
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchSemantics:
    """Resolved SFT batch math.

    ``device_batch_size`` is per accelerator device. The dataloader emits
    ``per_process_batch_size`` rows, because a single JAX process owns
    ``local_device_count`` devices on single-host TPU slices such as v6e-8.
    """

    device_batch_size: int
    max_seq_len: int
    total_batch_size: int
    per_process_batch_size: int
    process_count: int
    local_device_count: int
    device_count: int
    tokens_per_microbatch_per_device: int
    tokens_per_microbatch_per_process: int
    world_tokens_per_microbatch: int
    grad_accum_steps: int


def resolve_batch_semantics(
    *,
    device_batch_size: int,
    max_seq_len: int,
    total_batch_size: int | None,
    process_count: int,
    local_device_count: int,
    device_count: int,
) -> BatchSemantics:
    """Resolve nanochat SFT batch accounting.

    The upstream token batch is per-device rows x sequence length x all devices
    x gradient accumulation. Using only ``jax.process_count()`` is wrong on
    single-host multi-chip TPU slices, where ``process_count == 1`` but
    ``device_count`` can be 8.
    """
    if device_batch_size <= 0:
        raise ValueError("device_batch_size must be positive")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")

    process_count = max(int(process_count), 1)
    local_device_count = max(int(local_device_count), 1)
    device_count = max(int(device_count), 1)

    per_process_batch_size = device_batch_size * local_device_count
    tokens_per_microbatch_per_device = device_batch_size * max_seq_len
    tokens_per_microbatch_per_process = per_process_batch_size * max_seq_len
    world_tokens_per_microbatch = tokens_per_microbatch_per_device * device_count

    if total_batch_size is None:
        total_batch_size = world_tokens_per_microbatch
    else:
        total_batch_size = int(total_batch_size)
        if total_batch_size <= 0:
            total_batch_size = world_tokens_per_microbatch

    if total_batch_size % world_tokens_per_microbatch != 0:
        grad_accum_steps = max(1, total_batch_size // world_tokens_per_microbatch)
        total_batch_size = grad_accum_steps * world_tokens_per_microbatch
    else:
        grad_accum_steps = total_batch_size // world_tokens_per_microbatch

    return BatchSemantics(
        device_batch_size=device_batch_size,
        max_seq_len=max_seq_len,
        total_batch_size=total_batch_size,
        per_process_batch_size=per_process_batch_size,
        process_count=process_count,
        local_device_count=local_device_count,
        device_count=device_count,
        tokens_per_microbatch_per_device=tokens_per_microbatch_per_device,
        tokens_per_microbatch_per_process=tokens_per_microbatch_per_process,
        world_tokens_per_microbatch=world_tokens_per_microbatch,
        grad_accum_steps=grad_accum_steps,
    )


# -----------------------------------------------------------------------------
# SFT data generator
# -----------------------------------------------------------------------------


def sft_data_generator_bos_bestfit(
    *,
    dataset: "Task",
    tokenizer,
    device_batch_size: int,
    max_seq_len: int,
    ddp_rank: int,
    ddp_world_size: int,
    state: dict,
    num_iterations: int,
    buffer_size: int = 100,
):
    """BOS-aligned best-fit packing dataloader (mirrors upstream chat_sft.py).

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).

    Yields ``(inputs, targets)`` numpy int64 arrays, shape ``(B, T)``.

    The ``state`` dict is mutated in-place:

    - ``state["last_step"]``: bool — toggled True when num_iterations reached or
      consumption exhausts dataset
    - ``state["approx_progress"]``: float — 0→1 progress (for lr schedule)
    - ``state["current_epoch"]``: int — current epoch counter

    Args:
        dataset: :class:`Task` (typically TaskMixture).
        tokenizer: tokenizer with ``render_conversation()`` + ``get_bos_token_id()``.
        device_batch_size: per-device batch size.
        max_seq_len: max sequence length (row capacity is max_seq_len + 1).
        ddp_rank: process rank.
        ddp_world_size: total processes.
        state: mutable state dict (as described).
        num_iterations: max iterations (-1 = full epoch).
        buffer_size: conversation buffer size for best-fit (default 100).
    """
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = max_seq_len + 1 # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: list of (token_ids, loss_mask) tuples
    conv_buffer: list[tuple[list[int], list[int]]] = []
    cursor = ddp_rank # Each rank processes different conversations (for fetching)
    consumed = ddp_rank # Track actual consumption separately from buffering
    epoch = 1
    it = 0 # iteration counter

    def refill_buffer() -> None:
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            ids, mask = tokenizer.render_conversation(conversation)
            conv_buffer.append((ids, mask))
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1

    while True:
        rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        row_lengths: list[int] = [] # actual content length (excluding padding)
        for _ in range(device_batch_size):
            row: list[int] = []
            mask_row: list[int] = []
            padded = False
            content_len = row_capacity # default if no padding
            while len(row) < row_capacity:
                # Ensure buffer has conversations
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirely
                best_idx = -1
                best_len = 0
                for i, (conv, _) in enumerate(conv_buffer):
                    conv_len = len(conv)
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv)
                    mask_row.extend(conv_mask)
                    consumed += ddp_world_size # Track actual consumption
                else:
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    content_len = len(row)
                    row.extend([bos_token] * remaining) # Pad with BOS tokens
                    mask_row.extend([0] * remaining)
                    padded = True
                    break # Row is now full (with padding)

            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # Stopping condition
        it += 1
        if 0 < num_iterations <= it:
            state["last_step"] = True

        # Update progress (0→1)
        state["current_epoch"] = epoch
        if num_iterations > 0:
            state["approx_progress"] = it / num_iterations
        else:
            state["approx_progress"] = consumed / dataset_size
        # Trigger last_step when consumed enough (matches PT)
        if consumed >= dataset_size and num_iterations <= 0:
            state["last_step"] = True

        # Build numpy arrays
        batch = np.array(rows, dtype=np.int64)
        inputs = batch[:, :-1].astype(np.int32)
        targets = batch[:, 1:].astype(np.int64)

        # Apply loss mask: assistant-completion positions get target,
        # non-completion positions get -1 (ignore_index).
        mask_arr = np.array(mask_rows, dtype=np.int8)
        mask_targets = mask_arr[:, 1:] # shifted by 1 to align with targets
        targets[mask_targets == 0] = -1

        # Mask out padding positions in targets
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1:] = -1

        yield inputs, targets


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SFT (Supervised Fine-Tuning) for nanochat-jax"
    )
    # Model loading
    parser.add_argument("--model-tag", type=str, default=None,
                        help="model tag to load from (default: largest in base_checkpoints)")
    parser.add_argument("--model-step", type=int, default=None,
                        help="model step to load from (default: last step)")
    parser.add_argument("--output-model-tag", type=str, default=None,
                        help="model tag to save under in chatsft_checkpoints "
                             "(default: --model-tag or d{depth})")
    parser.add_argument("--load-optimizer", type=int, default=1,
                        help="warm-start optimizer from base checkpoint (0=no, 1=yes). "
                             "Upstream default 1; missing optimizer files fall back to cold start.")
    # Training horizon
    parser.add_argument("--num-iterations", type=int, default=-1,
                        help="number of optimization steps (-1 = full epoch). "
                             "Upstream default -1.")
    # Batch sizes (default: inherit from pretrained checkpoint)
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="max context length (default: inherit from pretrain)")
    parser.add_argument("--device-batch-size", type=int, default=None,
                        help="per-device batch size (default: 4 if not set)")
    parser.add_argument("--total-batch-size", type=int, default=None,
                        help="total batch size in tokens (default: inherit from pretrain, "
                             "else device_bs * max_seq_len * jax.device_count())")
    # Optimization (default: inherit from pretrained checkpoint user_config or fallbacks)
    parser.add_argument("--embedding-lr", type=float, default=None,
                        help="learning rate for embedding parameters (Adam)")
    parser.add_argument("--unembedding-lr", type=float, default=None,
                        help="learning rate for unembedding parameters (Adam)")
    parser.add_argument("--matrix-lr", type=float, default=None,
                        help="learning rate for matrix parameters (Muon)")
    parser.add_argument("--init-lr-frac", type=float, default=0.8,
                        help="initial LR as fraction of base LR")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="ratio of iterations for LR warmup")
    parser.add_argument("--warmdown-ratio", type=float, default=0.5,
                        help="ratio of iterations for LR warmdown")
    parser.add_argument("--final-lr-frac", type=float, default=0.0,
                        help="final LR as fraction of initial LR")
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=-1,
                        help="evaluate val bpb every N steps (-1 = only at end)")
    parser.add_argument("--eval-steps", type=int, default=4,
                        help="number of val batches for BPB measurement")
    parser.add_argument("--chatcore-every", type=int, default=200,
                        help="evaluate ChatCORE metric every N steps (-1 = disable). "
                             "Upstream default 200.")
    parser.add_argument("--chatcore-max-cat", type=int, default=-1,
                        help="max problems per categorical task for ChatCORE")
    parser.add_argument("--chatcore-max-sample", type=int, default=24,
                        help="max problems per generative task for ChatCORE")
    # Data mixture
    parser.add_argument("--mmlu-epochs", type=int, default=3,
                        help="number of epochs of MMLU in training mixture "
                             "(teaches Multiple Choice). Upstream default 3.")
    parser.add_argument("--gsm8k-epochs", type=int, default=4,
                        help="number of epochs of GSM8K in training mixture "
                             "(teaches Math and Tool Use). Upstream default 4.")
    parser.add_argument("--sft-scope", type=str, default="full",
                        choices=["full", "reduced", "min"],
                        help="SFT training mixture scope:\n"
                             " full -- 7-source mixture (~1.07M rows)\n"
                             " reduced -- ~150K rows\n"
                             " min -- minimum (~12K rows, Mac CPU sanity)")
    # Distributed
    parser.add_argument("--no-distributed", action="store_true",
                        help="Skip jax.distributed.initialize() (CPU dry run)")
    # Compute precision
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bf16 compute (matches the published TPU run; CPU runs stay fp32)")
    # Logging
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N steps (master process only).")
    args = parser.parse_args()
    raw_user_config = vars(args).copy()

    # ---------------------------------------------------------------------
    # Compute / distributed init
    # ---------------------------------------------------------------------
    if not args.no_distributed:
        try:
            jax.distributed.initialize()
        except RuntimeError as e:
            _print0(f"WARNING: jax.distributed.initialize() failed ({e}); "
                    f"continuing single-process.")
    setup_distributed_env_vars()
    mesh = make_mesh()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    device_count = jax.device_count()
    rank = jax.process_index()
    is_master = rank == 0

    if is_master:
        print("=" * 80)
        print(f" chat_sft - process {rank}/{process_count}")
        print(f" mesh: {mesh}")
        print(f" devices: {jax.devices()}")
        print(f" local_device_count: {local_device_count} | device_count: {device_count}")
        print(f" bf16: {args.bf16}")
        print("=" * 80)

    # ---------------------------------------------------------------------
    # wandb (stubbed out — DummyWandb mirrors upstream's dummy run object)
    # ---------------------------------------------------------------------
    wandb_run = DummyWandb()  # wandb dependency removed

    # ---------------------------------------------------------------------
    # Load base model.
    # ---------------------------------------------------------------------
    compute_dtype = jnp.bfloat16 if args.bf16 else jnp.float32
    model, tokenizer, meta = load_model(
        "base",
        compute_dtype=compute_dtype,
        model_tag=args.model_tag,
        step=args.model_step,
    )
    base_model_tag = args.model_tag
    base_model_step = meta.get("step")
    _print0(f"Loaded base model: tag={base_model_tag}, step={base_model_step}, "
            f"vocab_size={meta['model_config']['vocab_size']}, "
            f"sequence_len={meta['model_config']['sequence_len']}, "
            f"compute_dtype={compute_dtype}")
    if getattr(model.config, "attn_impl", "xla") == "splash":
        from nanochat_jax.gpt import set_splash_mesh

        set_splash_mesh(mesh)
        _print0("Registered Splash Attention mesh for multi-chip TPU execution")

    # ---------------------------------------------------------------------
    # Inherit hyperparams from the pretrained checkpoint (mirrors upstream chat_sft.py)
    # ---------------------------------------------------------------------
    pretrain_user_config = meta.get("user_config", {})
    for name, fallback, source in [
        ("max_seq_len", 2048, None),  # special-cased below (model_config.sequence_len)
        ("device_batch_size", 4, meta),
        ("total_batch_size", None, meta), # computed if None
        ("embedding_lr", 0.3, pretrain_user_config),
        ("unembedding_lr", 0.004, pretrain_user_config),
        ("matrix_lr", 0.02, pretrain_user_config),
    ]:
        arg_val = getattr(args, name)
        # max_seq_len: inherit from model_config.sequence_len if present
        if name == "max_seq_len":
            pretrain_val = meta["model_config"].get("sequence_len", fallback)
        else:
            pretrain_val = source.get(name) if source else None
        if arg_val is None:
            resolved = pretrain_val if pretrain_val is not None else fallback
            setattr(args, name, resolved)
            _print0(f"Inherited {name}={resolved}")
        elif pretrain_val is not None and arg_val != pretrain_val:
            _print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained {pretrain_val}")

    depth = meta["model_config"]["n_layer"]
    requested_total_batch_size = args.total_batch_size
    batch_semantics = resolve_batch_semantics(
        device_batch_size=args.device_batch_size,
        max_seq_len=args.max_seq_len,
        total_batch_size=args.total_batch_size,
        process_count=process_count,
        local_device_count=local_device_count,
        device_count=device_count,
    )
    args.total_batch_size = batch_semantics.total_batch_size
    grad_accum_steps = batch_semantics.grad_accum_steps
    if requested_total_batch_size != args.total_batch_size:
        _print0(f"Resolved total_batch_size={args.total_batch_size:,} "
                f"(requested/inherited={requested_total_batch_size})")
    _print0(f"Rows / micro-batch / process: {batch_semantics.per_process_batch_size:,} "
            f"({args.device_batch_size} per device x {batch_semantics.local_device_count} local devices)")
    _print0(f"Tokens / micro-batch / device: {batch_semantics.tokens_per_microbatch_per_device:,}")
    _print0(f"Tokens / micro-batch / process: {batch_semantics.tokens_per_microbatch_per_process:,}")
    _print0(f"Tokens / micro-batch (world): {batch_semantics.world_tokens_per_microbatch:,}")
    _print0(f"Total batch size {args.total_batch_size:,} → grad_accum_steps: {grad_accum_steps}")
    if grad_accum_steps > 1:
        _print0(f"Using gradient accumulation: {grad_accum_steps} micro-batches "
                f"per optimizer step (loop)")

    # ---------------------------------------------------------------------
    # token_bytes
    # ---------------------------------------------------------------------
    token_bytes_arr = get_token_bytes()
    token_bytes = jnp.asarray(token_bytes_arr)

    # ---------------------------------------------------------------------
    # Initialize optimizer state
    # ---------------------------------------------------------------------
    param_groups, optim_state = init_train_state(
        model,
        embedding_lr=args.embedding_lr,
        unembedding_lr=args.unembedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=0.0, # SFT default
    )

    # Optional warm-start
    if args.load_optimizer:
        try:
            warm_state = load_optimizer_state(
                "base",
                rank=rank,
                model_tag=args.model_tag,
                step=args.model_step,
            )
            if warm_state is not None:
                optim_state = warm_state
                # LRs are scheduled per-step in the train loop (reading initial_lr
                # set by init_train_state), so there is nothing to restore here.
                _print0("Loaded base optimizer state (warm-start, LRs reset to SFT initial)")
            else:
                _print0("WARNING: base optimizer state not found — cold-start optim_state")
        except FileNotFoundError as e:
            _print0(f"WARNING: load_optimizer_state failed ({e}) — cold-start optim_state")

    # Override the initial learning rate as a fraction of the base learning rate
    for g in param_groups:
        g["lr"] = g["lr"] * args.init_lr_frac
        g["initial_lr"] = g["lr"]
    _print0(f"Applied init_lr_frac={args.init_lr_frac} to all param_groups")

    # ---------------------------------------------------------------------
    # SFT data mixture: CustomJSON identity plus upstream conversation tasks.
    # ---------------------------------------------------------------------
    base_dir = get_base_dir()
    identity_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
    if not os.path.exists(identity_filepath):
        _print0(f"identity_conversations.jsonl not found, downloading from {IDENTITY_CONVERSATIONS_URL}")
        try:
            download_file_with_lock(IDENTITY_CONVERSATIONS_URL, identity_filepath)
        except Exception as e:
            _print0(f"WARNING: download failed ({e}). SFT will run with empty mixture.")

    if args.sft_scope == "full":
        # mirrors upstream chat_sft.py
        train_tasks = [
            SmolTalk(split="train"), # 460K rows of general conversations
            CustomJSON(filepath=identity_filepath), # 1000 rows of synthetic identity conversations
            CustomJSON(filepath=identity_filepath), # 2 epochs of these
            *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
            *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
            SimpleSpelling(size=200000, split="train"), # 200K rows
            SpellingBee(size=80000, split="train"), # 80K rows
        ]
    elif args.sft_scope == "reduced":
        # Smaller public smoke/validation scope. Not a substitute for full SFT.
        train_tasks = [
            SmolTalk(split="train", stop=50000), # 50K (~10% of full)
            CustomJSON(filepath=identity_filepath),
            CustomJSON(filepath=identity_filepath), # 2 epochs
            MMLU(subset="all", split="auxiliary_train", stop=50000), # 50K (~half of full epoch)
            GSM8K(subset="main", split="train"), # full ~7K (small enough)
            SimpleSpelling(size=30000, split="train"), # 30K (15% of full)
            SpellingBee(size=20000, split="train"), # 20K (25% of full)
        ]
    elif args.sft_scope == "min":
        # Mac CPU sanity scope.
        train_tasks = [
            SmolTalk(split="train", stop=5000),
            CustomJSON(filepath=identity_filepath), # 1K
            MMLU(subset="all", split="auxiliary_train", stop=5000), # 5K
        ]
    else:
        raise ValueError(f"Unknown sft_scope: {args.sft_scope}")

    train_dataset = TaskMixture(train_tasks)
    epochs_note = (
        f" (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})"
        if args.sft_scope == "full" else ""
    )
    _print0(f"Training mixture (sft_scope={args.sft_scope}): {len(train_dataset):,} rows"
            f"{epochs_note}")

    # Val dataset: SmolTalk + MMLU + GSM8K test mixture (~29.6K rows)
    val_dataset = TaskMixture([
        SmolTalk(split="test"), # 24K rows
        MMLU(subset="all", split="test", stop=5200), # 5.2K (truncated to match PT ratios)
        GSM8K(subset="main", split="test", stop=420), # 0.42K (truncated)
    ])
    _print0(f"Val mixture: {len(val_dataset):,} rows")

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"Training dataset is empty — identity_conversations.jsonl missing or empty.\n"
            f"Expected at: {identity_filepath}\n"
            f"URL: {IDENTITY_CONVERSATIONS_URL}"
        )

    # ---------------------------------------------------------------------
    # Data generators (mirrors upstream chat_sft.py)
    # ---------------------------------------------------------------------
    train_state = {"last_step": False, "approx_progress": 0.0, "current_epoch": 1}
    train_loader = sft_data_generator_bos_bestfit(
        dataset=train_dataset,
        tokenizer=tokenizer,
        device_batch_size=batch_semantics.per_process_batch_size,
        max_seq_len=args.max_seq_len,
        ddp_rank=rank,
        ddp_world_size=process_count,
        state=train_state,
        num_iterations=(
            args.num_iterations * grad_accum_steps
            if args.num_iterations > 0 else args.num_iterations
        ),
    )

    def build_val_loader():
        val_state = {"last_step": False, "approx_progress": 0.0, "current_epoch": 1}
        return sft_data_generator_bos_bestfit(
            dataset=val_dataset,
            tokenizer=tokenizer,
            device_batch_size=batch_semantics.per_process_batch_size,
            max_seq_len=args.max_seq_len,
            ddp_rank=rank,
            ddp_world_size=process_count,
            state=val_state,
            num_iterations=args.eval_steps + 1, # +1 buffer
        )

    # ---------------------------------------------------------------------
    # Build sharded train_step factory
    # ---------------------------------------------------------------------
    if grad_accum_steps > 1:
        compute_grads_fn, accumulate_grads_fn, apply_update_fn = make_grad_accum_fns(
            mesh, param_groups, optim_state
        )
        train_step_fn = None
        _print0("Built gradient accumulation train functions")
    else:
        compute_grads_fn = accumulate_grads_fn = apply_update_fn = None
        train_step_fn = make_train_step_sharded(mesh, param_groups, optim_state)
        _print0("Built train_step_fn")

    # ---------------------------------------------------------------------
    # Training loop (mirrors upstream chat_sft.py, simplified)
    # ---------------------------------------------------------------------
    progress = 0.0 # monotonic
    min_val_bpb = float("inf")
    smooth_train_loss = 0.0
    ema_beta = 0.9
    total_training_time = 0.0

    step = 0
    val_bpb = None
    while True:
        # ---------------------------------------------------------------------
        # Training step
        # ---------------------------------------------------------------------
        t0 = time.time()
        x, y = next(train_loader)
        x_jax = jnp.asarray(x)
        y_jax = jnp.asarray(y)

        # Schedules
        progress = max(progress, train_state["approx_progress"])
        lrm = jnp.float32(get_lr_multiplier_progress(
            progress,
            warmup_ratio=args.warmup_ratio,
            warmdown_ratio=args.warmdown_ratio,
            final_lr_frac=args.final_lr_frac,
        ))
        muon_momentum = jnp.float32(get_muon_momentum_sft(step))
        muon_wd = jnp.float32(0.0) # SFT default

        if grad_accum_steps <= 1:
            assert train_step_fn is not None
            loss, optim_state = train_step_fn(
                model, optim_state, x_jax, y_jax, lrm, muon_momentum, muon_wd,
            )
        else:
            assert compute_grads_fn is not None
            assert accumulate_grads_fn is not None
            assert apply_update_fn is not None
            total_loss = 0.0
            micro_loss, acc_grads = compute_grads_fn(model, x_jax, y_jax)
            total_loss += float(micro_loss)
            for _ in range(1, grad_accum_steps):
                x_micro, y_micro = next(train_loader)
                micro_loss, acc_grads = accumulate_grads_fn(
                    model, acc_grads, jnp.asarray(x_micro), jnp.asarray(y_micro)
                )
                total_loss += float(micro_loss)
            inv_accum = jnp.float32(1.0 / grad_accum_steps)
            acc_grads = jax.tree.map(lambda g: g * inv_accum, acc_grads)
            loss = jnp.float32(total_loss / grad_accum_steps)
            optim_state = apply_update_fn(
                model, optim_state, acc_grads, lrm, muon_momentum, muon_wd
            )
        train_loss = float(loss)

        t1 = time.time()
        dt = t1 - t0

        # last_step DDP sync after consuming the final micro-batch for this step.
        if process_count > 1:
            last_step_arr = jax.numpy.asarray([1 if train_state["last_step"] else 0], dtype=jnp.int32)
            from jax.experimental import multihost_utils

            gathered = multihost_utils.process_allgather(last_step_arr)
            train_state["last_step"] = bool(int(gathered.max()))

        # State
        step += 1

        # Logging (EMA loss)
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
        debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        if step > 5: # skip warmup
            total_training_time += dt
        if step % args.log_every == 0 or step == 1:
            tok_per_sec = int(args.total_batch_size / max(dt, 1e-6))
            _print0(
                f"step {step:05d} ({100*progress:.2f}%) | loss: {debiased_loss:.6f} | "
                f"lrm: {float(lrm):.3f} | dt: {dt*1000:.1f}ms | tok/sec: {tok_per_sec:,} | "
                f"epoch: {train_state['current_epoch']}"
            )
            wandb_run.log({"step": step, "train/loss": debiased_loss, "train/dt": dt})

        # Eval after the completed optimizer step, including the final step.
        if train_state["last_step"] or (args.eval_every > 0 and step % args.eval_every == 0):
            val_loader = build_val_loader()
            try:
                val_bpb = float(evaluate_bpb(
                    model, val_loader, args.eval_steps, token_bytes
                ))
                _print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
                if val_bpb < min_val_bpb:
                    min_val_bpb = val_bpb
                wandb_run.log({"step": step, "val/bpb": val_bpb})
            except Exception as e:
                _print0(f"WARNING: evaluate_bpb failed ({e}). Skipping eval.")
                val_bpb = None

        # Periodic ChatCORE eval (mirrors upstream chat_sft.py).
        if args.chatcore_every > 0 and (
            train_state["last_step"] or (step > 0 and step % args.chatcore_every == 0)
        ):
            from nanochat_jax.engine import Engine
            from scripts.chat_eval import run_chat_eval

            all_tasks = [
                "ARC-Easy",
                "ARC-Challenge",
                "MMLU",
                "GSM8K",
                "HumanEval",
                "SpellingBee",
            ]
            categorical_tasks = {"ARC-Easy", "ARC-Challenge", "MMLU"}
            baseline_accuracies = {
                "ARC-Easy": 0.25,
                "ARC-Challenge": 0.25,
                "MMLU": 0.25,
                "GSM8K": 0.0,
                "HumanEval": 0.0,
                "SpellingBee": 0.0,
            }
            task_results: dict[str, float] = {}
            attn_state = set_attention_impl_for_chat_eval(model)
            try:
                if any(old_value == "splash" for _, _, old_value in attn_state):
                    _print0("Using XLA attention for in-loop ChatCORE variable-length prompts")
                engine = Engine(model, tokenizer)
                for task_name in all_tasks:
                    limit = (
                        args.chatcore_max_cat
                        if task_name in categorical_tasks
                        else args.chatcore_max_sample
                    )
                    max_problems = None if limit < 0 else limit
                    acc = run_chat_eval(
                        task_name,
                        model,
                        tokenizer,
                        engine,
                        batch_size=batch_semantics.per_process_batch_size,
                        max_problems=max_problems,
                    )
                    task_results[task_name] = acc
                    _print0(f"  {task_name}: {100 * acc:.2f}%")
            finally:
                restore_attention_impl_state(attn_state)

            def centered_mean(task_names: list[str] | set[str]) -> float:
                return sum(
                    (task_results[name] - baseline_accuracies[name])
                    / (1.0 - baseline_accuracies[name])
                    for name in task_names
                ) / len(task_names)

            chatcore = centered_mean(all_tasks)
            chatcore_cat = centered_mean(categorical_tasks)
            _print0(
                f"Step {step:05d} | ChatCORE: {chatcore:.4f} | "
                f"ChatCORE_cat: {chatcore_cat:.4f}"
            )
            wandb_run.log({
                "step": step,
                "chatcore_metric": chatcore,
                "chatcore_cat": chatcore_cat,
                **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
            })

        # save_checkpoint at last_step
        if train_state["last_step"]:
            output_dirname = args.output_model_tag or args.model_tag or f"d{depth}"
            checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
            resolved_user_config = vars(args).copy()
            output_model_config = dict(meta["model_config"])
            output_model_config.update({
                "sequence_len": args.max_seq_len,
                "vocab_size": tokenizer.get_vocab_size(),
                "n_layer": depth,
                "n_head": meta["model_config"]["n_head"],
                "n_kv_head": meta["model_config"]["n_kv_head"],
                "n_embd": meta["model_config"]["n_embd"],
                "window_pattern": meta["model_config"].get("window_pattern", "L"),
            })
            meta_data = {
                "step": step,
                "val_bpb": val_bpb if val_bpb is not None else float("nan"),
                "model_config": output_model_config,
                "sft": {
                    "base_model_tag": base_model_tag,
                    "base_model_step": base_model_step,
                    "output_model_tag": output_dirname,
                    "sft_scope": args.sft_scope,
                    "train_rows": len(train_dataset),
                    "val_rows": len(val_dataset),
                    "mmlu_epochs": args.mmlu_epochs,
                    "gsm8k_epochs": args.gsm8k_epochs,
                    "batch_semantics": asdict(batch_semantics),
                    "grad_accum_impl": "loop",
                },
                "raw_user_config": raw_user_config,
                "user_config": resolved_user_config,
            }
            save_checkpoint(checkpoint_dir, step, model, optim_state, meta_data, rank=rank)
            _print0(f"Saved SFT checkpoint to {checkpoint_dir}/[model|optim|meta]_{step:06d}.*")
            break

        # GC management (PT mirror)
        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    _print0(f"Total training time: {total_training_time:.2f}s")
    _print0(f"Min val_bpb: {min_val_bpb:.4f}")
    _print0(f"Final val_bpb: {val_bpb if val_bpb is not None else 'N/A'}")
    if is_master:
        min_val_bpb_report = None if min_val_bpb == float("inf") else min_val_bpb
        log_report_safe(
            section="Chat SFT",
            data=[
                vars(args),
                {
                    "base_model_tag": base_model_tag,
                    "base_model_step": base_model_step,
                    "output_model_tag": output_dirname,
                    "checkpoint_dir": checkpoint_dir,
                    "sft_scope": args.sft_scope,
                    "train_rows": len(train_dataset),
                    "val_rows": len(val_dataset),
                    "Number of iterations": step,
                    "JAX world size": process_count,
                    "JAX device count": device_count,
                    "grad_accum_steps": grad_accum_steps,
                    "total_batch_size": args.total_batch_size,
                    "max_seq_len": args.max_seq_len,
                    "device_batch_size": args.device_batch_size,
                },
                {
                    "Minimum validation bpb": min_val_bpb_report,
                    "Final validation bpb": val_bpb,
                    "Total training time s": total_training_time,
                },
                {"batch_semantics": asdict(batch_semantics)},
            ],
        )
    wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
