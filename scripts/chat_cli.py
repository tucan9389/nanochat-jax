"""Interactive chat CLI for nanochat-jax models.

Usage::

    python -m scripts.chat_cli                       # interactive base mode
    python -m scripts.chat_cli --prompt "Hello"      # single prompt + exit
    python -m scripts.chat_cli -t 0.0                # greedy
    python -m scripts.chat_cli --dtype bfloat16      # bf16 opt-in (TPU)
"""

from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp

from nanochat_jax.checkpoint_manager import load_model
from nanochat_jax.engine import Engine


_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse ArgumentParser for chat_cli."""
    parser = argparse.ArgumentParser(
        description="Chat with the nanochat-jax model"
    )
    parser.add_argument(
        "-i",
        "--source",
        type=str,
        default="base",
        choices=["base", "sft", "rl"],
        help=("Source of the model: base|sft|rl. Default 'base'."),
    )
    parser.add_argument(
        "-g",
        "--model-tag",
        type=str,
        default=None,
        help="Model tag to load (default: largest depth in checkpoints dir)",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=None,
        help="Step to load (default: largest step in model_tag dir)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="",
        help="Single prompt -> single response -> exit (no interactive loop)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (0.0 = greedy)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per assistant turn",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="JAX PRNGKey seed for sampling reproducibility",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=list(_DTYPE_MAP.keys()),
        help="Compute dtype for the model. Mac CPU: float32; TPU: bfloat16 opt-in.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Override checkpoints base dir (default: nanochat-jax cache)",
    )
    return parser


def _format_devices() -> str:
    """Compact device list for diagnostics (e.g., 'cpu:1' or 'TpuDevice:8')."""
    devices = jax.devices()
    if not devices:
        return "<no devices>"
    kinds = {}
    for d in devices:
        kinds[d.platform] = kinds.get(d.platform, 0) + 1
    return ", ".join(f"{k}:{n}" for k, n in kinds.items())


def main(argv: list[str] | None = None) -> int:
    """Main entry point: interactive or single-prompt chat with the model."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    compute_dtype = _DTYPE_MAP[args.dtype]
    print(f"Devices: {_format_devices()}", flush=True)
    print(f"Loading model (source={args.source}, dtype={args.dtype})...", flush=True)

    model, tokenizer, meta = load_model(
        args.source,
        compute_dtype=compute_dtype,
        model_tag=args.model_tag,
        step=args.step,
        base_dir=args.base_dir,
    )
    val_bpb = meta.get("val_bpb")
    val_bpb_str = f"{val_bpb:.4f}" if val_bpb is not None else "N/A"
    print(
        f"Loaded {args.source} model (step={meta.get('step', '?')}, "
        f"val_bpb={val_bpb_str}, n_layer={model.config.n_layer}, "
        f"n_embd={model.config.n_embd})",
        flush=True,
    )

    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    engine = Engine(model, tokenizer)

    print("\nnanochat-jax interactive mode", flush=True)
    print("-" * 50, flush=True)
    print("Type 'quit' or 'exit' to end the conversation", flush=True)
    print("Type 'clear' to start a new conversation", flush=True)
    if args.source == "base":
        print(
            "[Note] base models have not seen chat-formatted data; "
            "instruction-following is limited until SFT.",
            flush=True,
        )
    print("-" * 50, flush=True)

    conversation_tokens = [bos]
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k if args.top_k and args.top_k > 0 else None,
        "seed": args.seed,
    }

    while True:
        if args.prompt:
            user_input = args.prompt
        else:
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!", flush=True)
                break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!", flush=True)
            break

        if user_input.lower() == "clear":
            conversation_tokens = [bos]
            print("Conversation cleared.", flush=True)
            continue

        if not user_input:
            continue

        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)

        conversation_tokens.append(assistant_start)
        response_tokens: list[int] = []
        print("\nAssistant: ", end="", flush=True)
        for token_column, _ in engine.generate(
            conversation_tokens, **generate_kwargs
        ):
            token = token_column[0]
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
        print(flush=True)

        if response_tokens and response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)

        if args.prompt:
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
