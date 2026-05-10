"""Evaluate the Chat model on multiple-choice and generative tasks.

Generic loop logic lives here; per-task code is imported from ``nanochat_jax.tasks``.

Example runs::

    python -m scripts.chat_eval --source sft --task-name ARC-Easy
    python -m scripts.chat_eval -i sft  # all six tasks + ChatCORE metric
"""

from __future__ import annotations

import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils

from nanochat_jax.checkpoint_manager import load_model
from nanochat_jax.common import setup_distributed_env_vars
from nanochat_jax.engine import Engine

# -----------------------------------------------------------------------------
# Helpers (master-only printing, distributed sum reduce)
# -----------------------------------------------------------------------------


def _is_master() -> bool:
    return jax.process_index() == 0


def _print0(*args, **kwargs) -> None:
    if _is_master():
        print(*args, **kwargs)


def _all_reduce_sum(value: int) -> int:
    """All-reduce SUM across processes; single-process noop returns value as-is.

    PT pattern: ``torch.tensor([v], dtype=torch.long); dist.all_reduce(..., SUM)``
    JAX-port: ``multihost_utils.process_allgather + sum``.
    """
    if jax.process_count() <= 1:
        return value
    arr = jnp.asarray([value], dtype=jnp.int64)
    gathered = multihost_utils.process_allgather(arr)
    return int(jnp.sum(gathered))


# -----------------------------------------------------------------------------
# Generative evaluation loop (one problem at a time, sample, evaluate)
# -----------------------------------------------------------------------------


def run_generative_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    max_problems: int | None = None,
):
    """Sample completions and evaluate via ``task_object.evaluate``.

    PT mirror: ``nanochat/scripts/chat_eval.py:29-81``. JAX substitution:
    ``torch.tensor + dist.all_reduce`` → ``jnp.asarray + multihost_utils.process_allgather``.
    """
    process_count = max(jax.process_count(), 1)
    process_idx = jax.process_index()

    num_problems = (
        len(task_object) if max_problems is None else min(len(task_object), max_problems)
    )

    num_passed, total = 0, 0
    for i in range(process_idx, num_problems, process_count):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)

        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]

        # Evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        print(
            f"\r\033[KRank {process_idx} | {num_passed}/{total} ({100 * num_passed / total:.2f}%)",
            end="",
            flush=True,
        )

    # Newline before final summary
    print()

    # Aggregate results across all processes
    num_passed = _all_reduce_sum(num_passed)
    total = _all_reduce_sum(total)

    _print0("=" * 50)
    if total == 0:
        _print0("Final: 0/0 (no problems evaluated)")
        return 0.0
    _print0(f"Final: {num_passed}/{total} ({100 * num_passed / total:.2f}%)")
    return num_passed / total


# -----------------------------------------------------------------------------
# Categorical evaluation loop (multi-choice, no sampling — single batched forward)
# -----------------------------------------------------------------------------


def run_categorical_eval(
    task_object,
    tokenizer,
    model,
    batch_size: int,
    max_problems: int | None = None,
):
    """Argmax over the available answer letters in a single forward pass.

    PT mirror: ``nanochat/scripts/chat_eval.py:88-153``. JAX substitution:
    ``torch.tensor + torch.no_grad + dist.all_reduce`` →
    ``jnp.asarray + jnp.argmax + multihost_utils.process_allgather``.
    """
    process_count = max(jax.process_count(), 1)
    process_idx = jax.process_index()
    bos = tokenizer.get_bos_token_id()  # use BOS as pad token (positions ignored)

    num_problems = (
        len(task_object) if max_problems is None else min(len(task_object), max_problems)
    )

    def ceil_div(x: int, y: int) -> int:
        return -(-x // y)

    num_batches = ceil_div(num_problems, batch_size)

    # Cache letter→token id (letters often repeat across problems)
    letter_to_id_cache: dict[str, int] = {}
    num_passed, total = 0, 0

    for i in range(process_idx, num_batches, process_count):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch — pad/collate variable-length prompts
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids_list = [
            tokenizer.render_for_completion(conversation) for conversation in conversations
        ]
        max_length = max(len(ids) for ids in prompt_ids_list)
        # Position of the last token (predicted answer position)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids_list]
        padded_prompt_ids = [
            ids + [bos] * (max_length - len(ids)) for ids in prompt_ids_list
        ]
        prompt_ids = jnp.asarray(padded_prompt_ids, dtype=jnp.int32)

        # Forward pass
        logits = model(prompt_ids)  # (B, T, V)

        # Narrow logits to the available answer letters of each problem
        for idx, conversation in enumerate(conversations):
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, jnp.asarray(letter_ids)]
            argmax_letter_id = int(jnp.argmax(focus_logits, axis=-1))
            predicted_letter = letters[argmax_letter_id]
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all processes
    num_passed = _all_reduce_sum(num_passed)
    total = _all_reduce_sum(total)

    if total == 0:
        _print0("Final: 0/0 (no problems evaluated)")
        return 0.0
    average = num_passed / total
    _print0(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


# -----------------------------------------------------------------------------
# Task dispatch
# -----------------------------------------------------------------------------


def _build_task_module(task_name: str):
    """Lazy import each task to avoid datasets/HF startup cost when not needed."""
    if task_name == "HumanEval":
        from nanochat_jax.tasks.humaneval import HumanEval

        return HumanEval
    if task_name == "MMLU":
        from nanochat_jax.tasks.mmlu import MMLU

        return partial(MMLU, subset="all", split="test")
    if task_name == "ARC-Easy":
        from nanochat_jax.tasks.arc import ARC

        return partial(ARC, subset="ARC-Easy", split="test")
    if task_name == "ARC-Challenge":
        from nanochat_jax.tasks.arc import ARC

        return partial(ARC, subset="ARC-Challenge", split="test")
    if task_name == "GSM8K":
        from nanochat_jax.tasks.gsm8k import GSM8K

        return partial(GSM8K, subset="main", split="test")
    if task_name == "SpellingBee":
        from nanochat_jax.tasks.spellingbee import SpellingBee

        return partial(SpellingBee, size=256, split="test")
    raise ValueError(f"Unknown task name: {task_name}")


def run_chat_eval(
    task_name: str,
    model,
    tokenizer,
    engine,
    *,
    batch_size: int = 1,
    num_samples: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    max_problems: int | None = None,
):
    """Build the task object and dispatch to the right eval loop."""
    task_module = _build_task_module(task_name)
    task_object = task_module()
    if task_object.eval_type == "generative":
        acc = run_generative_eval(
            task_object,
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
        )
    elif task_object.eval_type == "categorical":
        acc = run_categorical_eval(
            task_object, tokenizer, model, batch_size, max_problems=max_problems
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


# PT baseline_accuracies (chat_eval.py:205-212) — random-baseline floor for ChatCORE
ALL_TASKS: list[str] = [
    "ARC-Easy",
    "ARC-Challenge",
    "MMLU",
    "GSM8K",
    "HumanEval",
    "SpellingBee",
]
BASELINE_ACCURACIES: dict[str, float] = {
    "ARC-Easy": 0.25,  # multiple choice 1 of 4 → 25%
    "ARC-Challenge": 0.25,  # multiple choice 1 of 4 → 25%
    "MMLU": 0.25,  # multiple choice 1 of 4 → 25%
    "GSM8K": 0.0,  # open-ended → 0%
    "HumanEval": 0.0,  # open-ended → 0%
    "SpellingBee": 0.0,  # open-ended → 0%
}


def compute_chatcore(
    results: dict[str, float],
    *,
    all_tasks: list[str] | None = None,
    baseline_accuracies: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute the ChatCORE metric (mean centered accuracy).

    PT mirror: ``nanochat/scripts/chat_eval.py:236-244``.

    Returns ``{"ChatCORE metric": <float>}`` if all_tasks were evaluated, else ``{}``.
    """
    if all_tasks is None:
        all_tasks = ALL_TASKS
    if baseline_accuracies is None:
        baseline_accuracies = BASELINE_ACCURACIES

    all_evaluated = all(task_name in results for task_name in all_tasks)
    if not all_evaluated:
        return {}

    centered_mean = 0.0
    for task_name, acc in results.items():
        baseline = baseline_accuracies.get(task_name, 0.0)
        centered = (acc - baseline) / (1.0 - baseline)
        centered_mean += centered
    chatcore = centered_mean / len(results)
    return {"ChatCORE metric": chatcore}


def _build_parser() -> argparse.ArgumentParser:
    """argparse 11 args PT 1:1 mirror (chat_eval.py:182-195)."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Chat model on 6 ICL tasks + ChatCORE metric."
    )
    parser.add_argument(
        "-i",
        "--source",
        type=str,
        required=True,
        help="Source of the model: sft|rl|base",
    )
    parser.add_argument(
        "-a",
        "--task-name",
        type=str,
        default=None,
        help="Task name. Default = all tasks. Use | to split multiple tasks.",
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for categorical evaluation",
    )
    parser.add_argument(
        "-g", "--model-tag", type=str, default=None, help="Model tag to load"
    )
    parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    parser.add_argument(
        "-x",
        "--max-problems",
        type=int,
        default=None,
        help="Max problems to evaluate per task (None = all)",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        choices=["cuda", "cpu", "mps", ""],
        help="Device type (PT compatibility — JAX-port is device-agnostic, ignored)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Setup distributed env vars
    setup_distributed_env_vars()

    # Load the model
    model, tokenizer, _meta = load_model(
        args.source, model_tag=args.model_tag, step=args.step
    )
    engine = Engine(model, tokenizer)

    # Determine the tasks to evaluate
    task_names = ALL_TASKS if args.task_name is None else args.task_name.split("|")

    results: dict[str, float] = {}
    for task_name in task_names:
        _print0("\n" + "=" * 80)
        _print0(f"Evaluating task: {task_name}")
        _print0("=" * 80)
        acc = run_chat_eval(
            task_name,
            model,
            tokenizer,
            engine,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
        )
        results[task_name] = acc
        _print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # ChatCORE metric (PT 1:1 mirror, only when all 6 tasks evaluated)
    chatcore_dict = compute_chatcore(results, all_tasks=ALL_TASKS)

    if _is_master():
        print("\n" + "=" * 80)
        print("Final results")
        print("=" * 80)
        for task_name, acc in results.items():
            print(f"{task_name}: {100 * acc:.2f}%")
        if chatcore_dict:
            print(f"ChatCORE metric: {chatcore_dict['ChatCORE metric']:.4f}")
        else:
            print(
                "ChatCORE metric: not computed (requires all 6 tasks; use --task-name with | "
                "to evaluate multiple)"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
