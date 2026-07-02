#!/usr/bin/env bash
# nanochat-jax CPU smoke: run the full speedrun pipeline on a tiny d2 model, on CPU.
#
#   tokenizer -> base pretrain/eval -> SFT -> ChatEval -> report
#
# This validates WIRING ONLY (stage plumbing, flags, checkpoint continuity) — the
# d2 model is far too small to be quality evidence. It uses its own -smoke base dir
# so it never reuses production checkpoints, and forces the CPU backend. The full
# production d24 recipe lives in runs/speedrun.sh.
#
#   bash runs/runcpu.sh

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
WANDB_RUN="${WANDB_RUN:-dummy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python3"

export NANOCHAT_JAX_BASE_DIR="${NANOCHAT_JAX_BASE_DIR:-$HOME/.cache/nanochat-jax-smoke}"
mkdir -p "$NANOCHAT_JAX_BASE_DIR"

MODEL_TAG="${MODEL_TAG:-speedrun_smoke_d2}"
SFT_MODEL_TAG="${SFT_MODEL_TAG:-${MODEL_TAG}_sft_smoke}"
RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_JAX_BASE_DIR/results/$MODEL_TAG}"
mkdir -p "$RESULTS_DIR"

echo "[runcpu] tiny d2 CPU wiring check | base tag: $MODEL_TAG | results: $RESULTS_DIR"

# -----------------------------------------------------------------------------
# Report reset

"$PYTHON_BIN" -m nanochat_jax.report reset

# -----------------------------------------------------------------------------
# Tokenizer  (1 tiny shard is enough for a wiring check)

"$PYTHON_BIN" -m nanochat_jax.dataset -n 1
"$PYTHON_BIN" -m scripts.tok_train --max-chars 100000 --doc-cap 2000 --vocab-size 8192
"$PYTHON_BIN" -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (tiny d2, synthetic tokens, fp32, xla attention)

"$PYTHON_BIN" -m scripts.base_train \
    --depth=2 \
    --seq-len=256 \
    --vocab-size=8192 \
    --total-batch-size=256 \
    --device-batch-size=1 \
    --grad-accum-steps=1 \
    --grad-accum-impl=loop \
    --num-iterations=2 \
    --no-bf16 \
    --attn-impl=xla \
    --matmul-precision=default \
    --lm-head-precision=highest \
    --ve-grad-impl=onehot \
    --checkpoint-every=1 \
    --keep-last-checkpoints=2 \
    --resume-from-step=-1 \
    --model-tag="$MODEL_TAG" \
    --no-final-eval \
    --no-distributed

# -----------------------------------------------------------------------------
# Base eval  (sample only; CPU-safe)

"$PYTHON_BIN" -m scripts.base_eval \
    --source base \
    --model-tag "$MODEL_TAG" \
    --eval sample \
    --eval-steps 1 \
    --device-batch-size 1 \
    --max-per-task 1 \
    --out "$RESULTS_DIR/base_eval.json" \
    --partial-out "$RESULTS_DIR/base_eval_core_partial.json" \
    --no-distributed

# -----------------------------------------------------------------------------
# SFT  (minimal mixture, 1 step)

"$PYTHON_BIN" -m scripts.chat_sft \
    --model-tag "$MODEL_TAG" \
    --output-model-tag "$SFT_MODEL_TAG" \
    --num-iterations 1 \
    --device-batch-size 1 \
    --max-seq-len 256 \
    --total-batch-size 256 \
    --sft-scope min \
    --eval-every -1 \
    --eval-steps 1 \
    --chatcore-every -1 \
    --chatcore-max-cat -1 \
    --chatcore-max-sample 24 \
    --load-optimizer 1 \
    --log-every 10 \
    --run "$WANDB_RUN" \
    --no-distributed

# -----------------------------------------------------------------------------
# ChatEval  (one problem, short generation)

"$PYTHON_BIN" -m scripts.chat_eval \
    -i sft \
    -g "$SFT_MODEL_TAG" \
    --max-new-tokens 32 \
    --batch-size 1 \
    --dtype float32 \
    --out "$RESULTS_DIR/chat_eval_sft.json" \
    --task-name ARC-Easy \
    --max-problems 1

# -----------------------------------------------------------------------------
# Final report

"$PYTHON_BIN" -m nanochat_jax.report generate
