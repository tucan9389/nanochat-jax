"""Unified evaluation script for base models.

Combines ``--eval bpb`` (bits-per-byte on the validation split), ``--eval sample``
(generative samples), and ``--eval core`` (DCLM CORE 22 ICL tasks) under one
entry point.

Default mode is ``bpb,sample`` (CORE is slower and opt-in via
``--eval core,bpb,sample``).

Usage::

    # Mac CPU sanity
    python -m scripts.base_eval --source base --model-tag d12_jax_seed42 \\
        --eval bpb,sample --eval-steps 2 --device-batch-size 2

    # TPU bf16 strict tier
    python -m scripts.base_eval --source base --model-tag d12_jax_seed42 \\
        --eval bpb,sample --eval-steps 20 --device-batch-size 4 --bf16

    # Single-mode quick check
    python -m scripts.base_eval --eval bpb --eval-steps 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import time
import zipfile

# Allow `python scripts/base_eval.py` from the project root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import yaml  # noqa: E402

from nanochat_jax import core_eval as _core_eval  # noqa: E402
from nanochat_jax.checkpoint_manager import load_model  # noqa: E402
from nanochat_jax.common import (  # noqa: E402
    download_file_with_lock,
    get_base_dir,
    setup_distributed_env_vars,
)
from nanochat_jax.engine import Engine  # noqa: E402
from nanochat_jax.loss_eval import evaluate_bpb  # noqa: E402
from nanochat_jax.report import log_report_safe  # noqa: E402

# PT mirror: nanochat/scripts/base_eval.py:92 (S3 bundle URL, Karpathy public)
_EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


# Dtype mapping
_DTYPE_FP32 = jnp.float32
_DTYPE_BF16 = jnp.bfloat16


# PT 1:1 mirror — sample mode prompts (base_eval.py:230-238)
_CONDITIONED_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


def _is_master() -> bool:
    """Master process indicator."""
    return jax.process_index() == 0


def _print0(*args, **kwargs) -> None:
    """Master-only print. PT print0 mirror."""
    if _is_master():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse ArgumentParser for base_eval.

    Exposed as a function (not module-level) so tests can construct + inspect
    without triggering jax.devices() side effects.
    """
    parser = argparse.ArgumentParser(
        description="Base model evaluation "
    )
    # PT mirror: --eval 
    parser.add_argument(
        "--eval",
        type=str,
        default="bpb,sample",
        help=(
            "Comma-separated evaluations: bpb,sample,core (default: bpb,sample). "
            "core uses nanochat_jax.core_eval  — slow on "
            "Mac CPU; --max-per-task=5 for sanity, =100 for TPU strict tier."
        ),
    )
    # PT mirror: --hf-path
    parser.add_argument(
        "--hf-path",
        type=str,
        default=None,
        help=(
            "HuggingFace model path. NOT IMPLEMENTED in JAX-port — "
            "use PT base_eval directly for HF model evaluation."
        ),
    )
    # PT mirror: --source
    parser.add_argument(
        "--source",
        type=str,
        default="base",
        choices=["base", "sft", "rl"],
        help=(
            "Source of the model: base|sft|rl. Default 'base'. "
            "RL loading is reserved for a future port."
        ),
    )
    # PT mirror: --model-tag, --step
    parser.add_argument(
        "--model-tag",
        type=str,
        default=None,
        help="Model tag (default: auto-detect largest depth in checkpoints dir)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step to load (default: largest step in model_tag dir)",
    )
    # PT mirror: --max-per-task (CORE-only)
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=100,
        help=(
            "(CORE-only) Max examples per CORE task. Default 100. "
            "-1 = all examples; full CORE is slow and should run on TPU."
        ),
    )
    # PT mirror: --device-batch-size
    parser.add_argument(
        "--device-batch-size",
        type=int,
        default=4,
        help=(
            "Per-device batch size for BPB evaluation. Default 4 "
            "(Mac CPU sanity; PT default 32 for 8-GPU setup)"
        ),
    )
    # JAX-port: --eval-steps
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=4,
        help=(
            "Number of eval batches per BPB split. Total tokens evaluated per "
            "process = eval_steps × device_batch_size × sequence_len."
        ),
    )
    # PT mirror: --device-type 
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        help="(unused; jax.devices() autodetects)",
    )
    # PT mirror: --seed (sample multinomial RNG)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="JAX PRNGKey seed for sample multinomial (uncond) reproducibility",
    )
    # JAX-port: dtype opt-in 
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help=(
            "Use bf16 compute_dtype. Default fp32 for Mac "
            "CPU sanity, opt-in for TPU."
        ),
    )
    # JAX-port: sample uncond config 
    parser.add_argument(
        "--num-uncond-samples",
        type=int,
        default=2,
        help=(
            "Number of unconditioned samples ("
            "PT default 8 → JAX 2 for ~5x speed)"
        ),
    )
    parser.add_argument(
        "--max-tokens-uncond",
        type=int,
        default=64,
        help=(
            "Max tokens per unconditioned sample ("
            "PT default 128 → JAX 64 for ~2x speed)"
        ),
    )
    parser.add_argument(
        "--max-tokens-cond",
        type=int,
        default=16,
        help="Max tokens per conditioned sample ",
    )
    parser.add_argument(
        "--temperature-uncond",
        type=float,
        default=1.0,
        help="Temperature for unconditioned samples ",
    )
    # JAX-port: distributed setup 
    parser.add_argument(
        "--no-distributed",
        action="store_true",
        default=False,
        help="Skip jax.distributed.initialize() (Mac CPU / dev mode default-on)",
    )
    # PT mirror: output (JSON primary + CSV optional)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output JSON path (default: get_base_dir()/base_eval/"
            "{model_slug}.json). Use - to skip writing."
        ),
    )
    parser.add_argument(
        "--partial-out",
        type=str,
        default=None,
        help=(
            "(CORE-only) Path to per-task partial JSON for preempt safety + "
            "resume. Each task result persisted immediately after completion. "
            "On rerun, completed tasks are skipped (resume from partial_out). "
            "Recommended for TPU spot runs (preemption recovery)."
        ),
    )
    # TPU default fp32 matmul can use bf16 internal accumulation; highest
    # requests a slower, higher-accuracy accumulation path when needed.
    parser.add_argument(
        "--matmul-precision",
        type=str,
        default=None,
        choices=[None, "default", "high", "highest"],
        help=(
            "Override JAX default matmul precision (jax.config update). "
            "TPU default = bf16 internal acc (Fastest). 'highest' = 6-pass bf16 "
            "= near-fp32 acc (~2.5x slower)."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help=(
            "Override checkpoints base dir (default: nanochat-jax cache directory)"
        ),
    )
    return parser


def _validate_eval_modes(eval_arg: str) -> set[str]:
    """Parse and validate the ``--eval`` argument.

    Returns:
        Set of mode strings: subset of {"bpb", "sample", "core"}.

    Raises:
        ValueError: invalid mode.
    """
    eval_modes = {mode.strip() for mode in eval_arg.split(",") if mode.strip()}
    valid_modes = {"bpb", "sample", "core"}
    invalid = eval_modes - valid_modes
    if invalid:
        raise ValueError(
            f"Invalid eval modes: {sorted(invalid)}. Valid: {sorted(valid_modes)}"
        )
    if not eval_modes:
        raise ValueError(
            "--eval must specify at least one mode (e.g. --eval bpb,sample)"
        )
    return eval_modes


def _format_devices() -> str:
    """Compact device list for diagnostics."""
    devices = jax.devices()
    if not devices:
        return "<no devices>"
    kinds: dict[str, int] = {}
    for d in devices:
        kinds[d.platform] = kinds.get(d.platform, 0) + 1
    return ", ".join(f"{k}:{n}" for k, n in kinds.items())


def _resolve_compute_dtype(bf16_flag: bool):
    """Resolve compute_dtype from --bf16 flag (Q-3 (C)).

    bf16 not supported on Mac arm64 CPU (numerical issues on some platforms). Default fp32 for safety.
    """
    return _DTYPE_BF16 if bf16_flag else _DTYPE_FP32


# -----------------------------------------------------------------------------
# CORE evaluation
# -----------------------------------------------------------------------------


def _place_eval_bundle(file_path: str) -> None:
    """Unzip ``eval_bundle.zip`` and place ``eval_bundle/`` in base dir.

PT mirror: ``nanochat/scripts/base_eval.py:95-104`` (substitution 0).

    Used as ``postprocess_fn`` for :func:`download_file_with_lock` — extracts
    the zip into a temp dir, then atomically moves the inner ``eval_bundle/``
    directory to ``<base_dir>/eval_bundle/``.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    _print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def _evaluate_core(
    model,
    tokenizer,
    max_per_task: int = -1,
    *,
    partial_out: str | None = None,
) -> dict:
    """Evaluate base model on the CORE benchmark (22 ICL tasks).

PT mirror: ``nanochat/scripts/base_eval.py:107-173`` (substitution 0
    — yaml/csv/random framework-agnostic + nanochat_jax.core_eval.evaluate_task).

    POST-FIX (2026-05-03 — TPU spot preemption recovery cascade):
    ``partial_out`` parameter adds **resumable** partial JSON dump per task.
    On preemption / restart, completed tasks are skipped (read from
    ``partial_out`` JSON file). Each task result is persisted IMMEDIATELY
    after completion → max loss = 1 task worth of compute on preemption.

    Parameters
    ----------
    partial_out
        Optional path to persist per-task partial results. Format:
        ``{"results": {label: acc}, "centered_results": {label: c}}``.
        If file exists at start, completed tasks are skipped (resume).

    Returns
    -------
    dict
        ``{"results": {label: accuracy}, "centered_results": {label: ...},
        "core_metric": float}``. PT 1:1 schema.
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle if needed
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(
            _EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=_place_eval_bundle
        )

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    # Load random baseline values (eval_meta_data.csv → centered_result formula)
    random_baselines: dict[str, float] = {}
    with open(eval_meta_data, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["Eval Task"]
            random_baseline = row["Random baseline"]
            random_baselines[task_name] = float(random_baseline)

    # Resume from partial_out if exists (POST-FIX — preempt safety net)
    results: dict[str, float] = {}
    centered_results: dict[str, float] = {}
    if partial_out and os.path.exists(partial_out):
        try:
            with open(partial_out, encoding="utf-8") as f:
                partial = json.load(f)
            results = partial.get("results", {})
            centered_results = partial.get("centered_results", {})
            _print0(
                f"[resume] Loaded {len(results)} completed tasks from "
                f"{partial_out}: {sorted(results.keys())}"
            )
        except (OSError, json.JSONDecodeError) as e:
            _print0(f"[warn] Failed to load partial_out, starting fresh: {e}")
            results = {}
            centered_results = {}

    # Evaluate each task (PT 1:1 mirror — 22 ICL tasks loop, with resume)
    for task in tasks:
        label = task["label"]
        # Skip already-completed tasks (resume capability)
        if label in results:
            _print0(
                f"[skip] {label} already done "
                f"(accuracy={results[label]:.4f}, centered={centered_results[label]:.4f})"
            )
            continue

        start_time = time.time()
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        _print0(
            f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, "
            f"type: {task_meta['task_type']})... ",
            end="",
        )

        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        # Shuffle for consistent subsampling when using max_per_task (PT 1:1)
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # Guard against `random.sample(available, num_fewshot)` ValueError in
        # core_eval.py. `available` = len(data) - 1 (current item excluded), so
        # we need at least num_fewshot + 1 samples.
        num_fewshot_req = task_meta["num_fewshot"]
        if max_per_task > 0 and len(data) < num_fewshot_req + 1:
            raise ValueError(
                f"--max-per-task={max_per_task} too small for task '{label}': "
                f"num_fewshot={num_fewshot_req} requires at least "
                f"{num_fewshot_req + 1} samples (current data: {len(data)}). "
                f"Increase --max-per-task to {num_fewshot_req + 1} or higher "
                f"(default 100 recommended for safety)."
            )

        # JAX-port: device kept for PT 1:1 signature (unused, jax handles)
        accuracy = _core_eval.evaluate_task(model, tokenizer, data, None, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        _print0(
            f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | "
            f"time: {elapsed:.2f}s"
        )

        # Persist partial results IMMEDIATELY after task completes (POST-FIX preempt safety)
        if partial_out:
            os.makedirs(os.path.dirname(partial_out) or ".", exist_ok=True)
            tmp_path = partial_out + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"results": results, "centered_results": centered_results},
                    f,
                    indent=2,
                )
            os.replace(tmp_path, partial_out)  # atomic rename

    core_metric = sum(centered_results.values()) / len(centered_results)
    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }


def _run_core_mode(model, tokenizer, args: argparse.Namespace) -> dict | None:
    """Run CORE mode and write CSV output (PT base_eval.py:278-298 mirror).

Returns the ``core_results`` dict (or None if not master).

    POST-FIX (2026-05-03): Forwards ``--partial-out`` to ``_evaluate_core``
    for preempt-safe resume capability.
    """
    if not _is_master():
        # Non-master ranks still call evaluate_core (rank-stride loop), but
        # only master writes CSV. Run evaluate_core for distributed reduce.
        return _evaluate_core(
            model, tokenizer,
            max_per_task=args.max_per_task,
            partial_out=args.partial_out,
        )

    _print0("\n" + "=" * 80)
    _print0("CORE Evaluation")
    _print0("=" * 80)
    core_results = _evaluate_core(
        model, tokenizer,
        max_per_task=args.max_per_task,
        partial_out=args.partial_out,
    )
    _print0(f"\nCORE metric: {core_results['core_metric']:.4f}")
    return core_results


def _write_core_csv(out_path: str, core_results: dict) -> None:
    """Write CORE results to CSV (PT base_eval.py:286-297 mirror).

Format: ``Task,Accuracy,Centered`` rows + final ``CORE,,score`` row.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        for label in core_results["results"]:
            acc = core_results["results"][label]
            centered = core_results["centered_results"][label]
            f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
        f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
    _print0(f"\nResults written to: {out_path}")


def _hf_not_implemented(args: argparse.Namespace) -> None:
    """Raise NotImplementedError for --hf-path .

    HuggingFace model evaluation requires:
    - transformers package
    - HuggingFaceTokenizer (PT-only path)
    - ModelWrapper for nanochat-compatible interface

    JAX-port focus is nanochat-trained models. Use PT base_eval directly for HF.
    """
    raise NotImplementedError(
        "HuggingFace model evaluation (--hf-path) is not implemented in "
        "nanochat-jax. Use PT base_eval directly: "
        "torchrun --nproc_per_node=N -m scripts.base_eval --hf-path <path>."
    )


# -----------------------------------------------------------------------------
# Sample mode (PT base_eval.py:225-256 mirror)
# -----------------------------------------------------------------------------


def _run_sample_mode(
    model,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    """Run sample mode: 7 conditioned (greedy) + N unconditioned (multinomial).


    Args:
        model: NNX GPT instance (load_model output).
        tokenizer: RustBPETokenizer.
        args: argparse Namespace.

    Returns:
        (samples: list[str], unconditioned_samples: list[str]).
        Both empty if not master process (PT mirror: only ddp_rank == 0 generates).
    """
    samples: list[str] = []
    unconditioned_samples: list[str] = []

    if not _is_master():
        return samples, unconditioned_samples

    _print0("\n" + "=" * 80)
    _print0("Model Samples")
    _print0("=" * 80)
    engine = Engine(model, tokenizer)

    # Conditioned (greedy, temperature=0)
    _print0("\nConditioned samples (greedy, max_tokens="
            f"{args.max_tokens_cond}):")
    t_cond_start = time.time()
    for prompt in _CONDITIONED_PROMPTS:
        # PT 1:1 mirror: tokenizer(prompt, prepend="<|bos|>")
        tokens = tokenizer(prompt, prepend="<|bos|>")
        sample, _masks = engine.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=args.max_tokens_cond,
            temperature=0.0,
        )
        sample_str = tokenizer.decode(sample[0])
        _print0("-" * 80)
        _print0(sample_str)
        samples.append(sample_str)
    t_cond = time.time() - t_cond_start

    # Unconditioned (multinomial, temperature=1.0)
    _print0(
        f"\nUnconditioned samples (temperature={args.temperature_uncond}, "
        f"num_samples={args.num_uncond_samples}, "
        f"max_tokens={args.max_tokens_uncond}):"
    )
    t_uncond_start = time.time()
    # PT 1:1 mirror: tokenizer("", prepend="<|bos|>") → just [bos_id]
    tokens = tokenizer("", prepend="<|bos|>")
    uncond, _masks = engine.generate_batch(
        tokens,
        num_samples=args.num_uncond_samples,
        max_tokens=args.max_tokens_uncond,
        temperature=args.temperature_uncond,
        seed=args.seed,
    )
    for sample_tokens in uncond:
        sample_str = tokenizer.decode(sample_tokens)
        _print0("-" * 80)
        _print0(sample_str)
        unconditioned_samples.append(sample_str)
    t_uncond = time.time() - t_uncond_start

    _print0(
        f"\n[time] sample cond={t_cond:.1f}s "
        f"uncond={t_uncond:.1f}s total={t_cond + t_uncond:.1f}s"
    )

    return samples, unconditioned_samples


# -----------------------------------------------------------------------------
# BPB mode
# -----------------------------------------------------------------------------


def _run_bpb_mode(
    model,
    tokenizer,
    sequence_len: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Run BPB mode: train + val splits, weighted by token byte length.

    PT base_eval.py:260-276 mirror. Uses evaluate_bpb (with multi-host
    all-reduce extension via process_allgather, automatic single-process
    short-circuit).

    Args:
        model: NNX GPT instance.
        tokenizer: RustBPETokenizer.
        sequence_len: model sequence length (from meta).
        args: argparse Namespace.

    Returns:
        dict like {"train": <float>, "val": <float>}.
    """
    # Lazy imports — only loaded when bpb mode is used (avoids data-loader deps for
    # sample-only quick checks)
    from nanochat_jax.dataloader import (  # noqa: PLC0415
        tokenizing_distributed_data_loader_bos_bestfit,
    )
    from nanochat_jax.tokenizer import get_token_bytes  # noqa: PLC0415

    bpb_results: dict[str, float] = {}

    _print0("\n" + "=" * 80)
    _print0("BPB Evaluation")
    _print0("=" * 80)

    # PT base_eval.py mirror: each process evaluates ``device_batch_size`` rows
    # per step. This path is not pmap/shard_map based, so local TPU device count
    # must not multiply the dataloader batch size.
    local_device_count = max(jax.local_device_count(), 1)
    process_count = max(jax.process_count(), 1)
    per_process_batch = args.device_batch_size
    tokens_per_step = per_process_batch * sequence_len * process_count
    total_tokens_per_split = args.eval_steps * tokens_per_step

    _print0(
        f"[info] device_batch_size={args.device_batch_size} "
        f"local_device_count={local_device_count} process_count={process_count} "
        f"per_process_batch={per_process_batch} sequence_len={sequence_len} "
        f"tokens_per_step={tokens_per_step:,} eval_steps={args.eval_steps} "
        f"total_tokens_per_split={total_tokens_per_split:,}"
    )

    # token_bytes → JAX array inside)
    token_bytes_np = get_token_bytes()
    token_bytes = jnp.asarray(token_bytes_np)
    assert token_bytes.shape[0] == tokenizer.get_vocab_size(), (
        f"token_bytes vocab mismatch: token_bytes.shape[0]="
        f"{token_bytes.shape[0]} vs tokenizer.get_vocab_size()="
        f"{tokenizer.get_vocab_size()}"
    )

    for split_name in ["train", "val"]:
        t_start = time.time()
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, per_process_batch, sequence_len, split=split_name
        )
        bpb = evaluate_bpb(model, loader, args.eval_steps, token_bytes)
        elapsed = time.time() - t_start
        bpb_results[split_name] = bpb
        _print0(f"{split_name} bpb: {bpb:.6f} (elapsed={elapsed:.1f}s)")

    return bpb_results


# -----------------------------------------------------------------------------
# Output writing (JSON primary)
# -----------------------------------------------------------------------------


def _resolve_out_path(
    args: argparse.Namespace, model_slug: str
) -> str | None:
    """Resolve output JSON path.

    Returns:
        Path string, or None if writing is disabled (--out -).
    """
    if args.out == "-":
        return None
    if args.out:
        return args.out

    # Default: get_base_dir()/base_eval/{model_slug}.json
    base_dir = get_base_dir()
    out_dir = os.path.join(base_dir, "base_eval")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{model_slug}.json")


def _write_results_json(
    out_path: str,
    *,
    model_name: str,
    model_slug: str,
    eval_modes: set[str],
    bpb_results: dict[str, float],
    samples: list[str],
    unconditioned_samples: list[str],
    meta: dict,
    args: argparse.Namespace,
    elapsed: float,
    core_results: dict | None = None,
) -> None:
    """Write results to JSON (JSON primary over PT CSV).

``core_results`` keyword added (None → no CORE block in JSON).
    """
    payload = {
        "model_name": model_name,
        "model_slug": model_slug,
        "eval_modes": sorted(eval_modes),
        "bpb": bpb_results,
        "core": core_results,  # None or {results, centered_results, core_metric}
        "samples": samples,
        "unconditioned_samples": unconditioned_samples,
        "meta_step": meta.get("step"),
        "meta_val_bpb_train_time": meta.get("val_bpb"),
        "model_config": meta.get("model_config", {}),
        "args": {
            "source": args.source,
            "model_tag": args.model_tag,
            "step": args.step,
            "device_batch_size": args.device_batch_size,
            "eval_steps": args.eval_steps,
            "bf16": args.bf16,
            "num_uncond_samples": args.num_uncond_samples,
            "max_tokens_uncond": args.max_tokens_uncond,
            "max_tokens_cond": args.max_tokens_cond,
            "max_per_task": args.max_per_task,
            "seed": args.seed,
        },
        "wall_time_s": elapsed,
        "jax_devices": _format_devices(),
        "process_count": int(jax.process_count()),
        "device_count": int(jax.device_count()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    _print0(f"\n[info] Saved results to: {out_path}")


def prepare_model_for_base_eval(model) -> bool:
    """Use eval-safe attention kernels for base model evaluation.

    Splash Attention is the TPU training kernel and requires query lengths to
    be block-size multiples. Base BPB may use full training sequence length, but
    CORE and sampling evaluate shorter prompts, so switch Splash checkpoints to
    XLA attention at runtime. Weights and checkpoint metadata are unchanged.
    """
    switched = False
    if getattr(model.config, "attn_impl", "xla") == "splash":
        model.config.attn_impl = "xla"
        switched = True
    transformer = getattr(model, "transformer", None)
    for block in getattr(transformer, "h", []):
        attn = getattr(block, "attn", None)
        if getattr(attn, "attn_impl", "xla") == "splash":
            attn.attn_impl = "xla"
            switched = True
    return switched


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Main entry point — base model evaluation (BPB + sample, optional CORE).

    Args:
        argv: optional explicit argv (for testing). None → ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success.
    """
    t_main_start = time.time()
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 1. Validate eval modes
    eval_modes = _validate_eval_modes(args.eval)

    # 2. HF model NotImplementedError
    if args.hf_path is not None:
        _hf_not_implemented(args)

    # 2b. matmul precision override.
    # MUST be applied BEFORE jax.distributed.initialize() / jax.devices() to
    # affect compilation.
    if args.matmul_precision is not None:
        jax.config.update("jax_default_matmul_precision", args.matmul_precision)
        _print0(f"[info] jax_default_matmul_precision = {args.matmul_precision}")

    # 3. Distributed setup 
    if not args.no_distributed:
        try:
            jax.distributed.initialize()
        except RuntimeError as exc:
            # Already-initialized or single-process (Mac CPU dev) — non-fatal
            _print0(
                f"[warn] jax.distributed.initialize() skipped: {exc}",
                file=sys.stderr,
            )
    setup_distributed_env_vars()

    _print0(f"[info] jax.devices() = {jax.devices()}")
    _print0(
        f"[info] process_index={jax.process_index()} "
        f"process_count={jax.process_count()} "
        f"local_device_count={jax.local_device_count()} "
        f"jax.device_count()={jax.device_count()}"
    )

    # 4. Load model + tokenizer + meta
    compute_dtype = _resolve_compute_dtype(args.bf16)
    _print0(
        f"[info] Loading {args.source} model "
        f"(compute_dtype={compute_dtype}, model_tag={args.model_tag}, "
        f"step={args.step}, base_dir={args.base_dir})"
    )
    t_load_start = time.time()
    model, tokenizer, meta = load_model(
        args.source,
        compute_dtype=compute_dtype,
        model_tag=args.model_tag,
        step=args.step,
        base_dir=args.base_dir,
    )
    t_load = time.time() - t_load_start
    if prepare_model_for_base_eval(model):
        _print0("Using XLA attention for base_eval variable-length prompts")

    sequence_len = meta["model_config"]["sequence_len"]
    model_name = f"base_model (step {meta['step']})"
    model_slug = f"base_model_{meta['step']:06d}"
    _print0(
        f"[info] Loaded model: name={model_name} slug={model_slug} "
        f"sequence_len={sequence_len} vocab={tokenizer.get_vocab_size()} "
        f"meta_val_bpb={meta.get('val_bpb', 'N/A')} load_time={t_load:.1f}s"
    )

    _print0(f"[info] Eval modes: {sorted(eval_modes)}")

    # Storage for results
    bpb_results: dict[str, float] = {}
    samples: list[str] = []
    unconditioned_samples: list[str] = []
    core_results: dict | None = None

    # 5. Sample mode (PT base_eval.py:225-256 mirror)
    if "sample" in eval_modes:
        samples, unconditioned_samples = _run_sample_mode(
            model, tokenizer, args
        )

    # 6. BPB mode
    if "bpb" in eval_modes:
        bpb_results = _run_bpb_mode(model, tokenizer, sequence_len, args)

    # 7. CORE mode
    if "core" in eval_modes:
        core_results = _run_core_mode(model, tokenizer, args)

    # 8. Final output
    elapsed = time.time() - t_main_start
    _print0(f"\n[done] Total wall time: {elapsed:.1f}s")
    if bpb_results:
        for split, bpb in bpb_results.items():
            _print0(f"  {split} bpb: {bpb:.6f}")
    if core_results:
        _print0(f"  CORE metric: {core_results['core_metric']:.6f}")

    out_path = _resolve_out_path(args, model_slug)
    if out_path is not None and _is_master():
        _write_results_json(
            out_path,
            model_name=model_name,
            model_slug=model_slug,
            eval_modes=eval_modes,
            bpb_results=bpb_results,
            samples=samples,
            unconditioned_samples=unconditioned_samples,
            meta=meta,
            args=args,
            elapsed=elapsed,
            core_results=core_results,
        )

    # CSV output for CORE (PT 1:1 mirror — base_eval.py:286-297)
    if core_results is not None and _is_master():
        csv_out_path = os.path.join(
            get_base_dir(), "base_eval", f"{model_slug}.csv"
        )
        _write_core_csv(csv_out_path, core_results)

    if _is_master():
        report_summary = {
            "model": model_name,
            "source": args.source,
            "model_tag": args.model_tag,
            "requested_step": args.step,
            "resolved_step": meta.get("step"),
            "wall_time_s": elapsed,
            "load_time_s": t_load,
        }
        if core_results is not None:
            report_summary["CORE metric"] = core_results["core_metric"]
        if bpb_results:
            report_summary["train bpb"] = bpb_results.get("train")
            report_summary["val bpb"] = bpb_results.get("val")
        report_data = [report_summary]
        if core_results is not None:
            report_data.append({"CORE centered results": core_results["centered_results"]})
        if samples:
            report_data.append({f"sample {i}": sample for i, sample in enumerate(samples)})
        if unconditioned_samples:
            report_data.append({
                f"unconditioned sample {i}": sample
                for i, sample in enumerate(unconditioned_samples)
            })
        log_report_safe(section="Base model evaluation", data=report_data)

    # 9. Cleanup
    return 0


if __name__ == "__main__":
    sys.exit(main())
