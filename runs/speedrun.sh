#!/usr/bin/env bash
# nanochat-jax speedrun: the public v6e-8 TPU d24 pipeline, start to end.
#
#   tokenizer -> base pretrain/eval -> SFT -> ChatEval -> report
#
# Trains the published 1.384B (d24) model. Needs a v6e-8 TPU host with TPU deps:
#   pip install -e ".[tpu,dev]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#
# For a tiny CPU wiring check (not quality evidence), use runs/runcpu.sh instead.
#
# The recipe is hardcoded below (like upstream nanochat's speedrun.sh). A few
# operational knobs are env-overridable: MODEL_TAG, NANOCHAT_JAX_BASE_DIR,
# DEVICE_BATCH_SIZE (lower it if you OOM), NUM_ITERATIONS.

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python3"

export NANOCHAT_JAX_BASE_DIR="${NANOCHAT_JAX_BASE_DIR:-$HOME/.cache/nanochat-jax}"
mkdir -p "$NANOCHAT_JAX_BASE_DIR"

MODEL_TAG="${MODEL_TAG:-d24_speedrun}"
SFT_MODEL_TAG="${SFT_MODEL_TAG:-${MODEL_TAG}_sft}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"   # per-device batch; lower to 1 if you OOM
NUM_ITERATIONS="${NUM_ITERATIONS:-16704}"     # d24 training horizon of the published run
RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_JAX_BASE_DIR/results/$MODEL_TAG}"
mkdir -p "$RESULTS_DIR"

echo "[speedrun] base tag: $MODEL_TAG | sft tag: $SFT_MODEL_TAG | results: $RESULTS_DIR"

# -----------------------------------------------------------------------------
# Report reset

"$PYTHON_BIN" -m nanochat_jax.report reset

# -----------------------------------------------------------------------------
# Tokenizer  (download 8 shards now; the 170 train shards download in the
# background so they overlap tokenizer training)

"$PYTHON_BIN" -m nanochat_jax.dataset -n 8
"$PYTHON_BIN" -m nanochat_jax.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

"$PYTHON_BIN" -m scripts.tok_train --max-chars 100000000 --doc-cap 10000 --vocab-size 32768
"$PYTHON_BIN" -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

# d24 recipe. Hyperparameters not shown (LRs, schedule, weight decay, aspect/head-dim)
# are base_train's defaults. Most flags below differ from the defaults or are
# TPU-specific; a few that match today's defaults (--recipe, --seq-len, --vocab-size,
# --bf16, --keep-last-checkpoints) are kept explicit on purpose: they pin this
# run's recipe, so a future default change cannot silently alter it.
"$PYTHON_BIN" -m scripts.base_train \
    --recipe=dc54a1a \
    --depth=24 \
    --seq-len=2048 \
    --vocab-size=32768 \
    --total-batch-size=524288 \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --grad-accum-steps=16 \
    --grad-accum-impl=fused \
    --num-iterations="$NUM_ITERATIONS" \
    --bf16 --cast-embeddings-bf16 \
    --use-real-data \
    --attn-impl=splash --splash-block-q=512 --splash-block-kv=512 --splash-block-kv-compute=256 \
    --matmul-precision=default \
    --lm-head-precision=highest \
    --ve-grad-impl=onehot \
    --checkpoint-every=500 \
    --keep-last-checkpoints=2 \
    --model-tag="$MODEL_TAG" \
    --no-final-eval

# -----------------------------------------------------------------------------
# Base eval  (CORE metric + BPB on train/val; writes JSON for the report)

"$PYTHON_BIN" -m scripts.base_eval \
    --source base \
    --model-tag "$MODEL_TAG" \
    --eval bpb,core \
    --eval-steps 20 \
    --device-batch-size 1 \
    --max-per-task -1 \
    --out "$RESULTS_DIR/base_eval.json" \
    --partial-out "$RESULTS_DIR/base_eval_core_partial.json" \
    --bf16

# -----------------------------------------------------------------------------
# SFT  (teach conversation special tokens, tool use, multiple choice)

"$PYTHON_BIN" -m scripts.chat_sft \
    --model-tag "$MODEL_TAG" \
    --output-model-tag "$SFT_MODEL_TAG" \
    --num-iterations -1 \
    --device-batch-size 1 \
    --max-seq-len 2048 \
    --total-batch-size 524288 \
    --sft-scope full \
    --eval-every 200 \
    --eval-steps 4 \
    --chatcore-every 200 \
    --chatcore-max-cat -1 \
    --chatcore-max-sample 24 \
    --load-optimizer 1 \
    --log-every 10 \
    --bf16

# -----------------------------------------------------------------------------
# ChatEval  (score the SFT model on the validated categorical task set)

"$PYTHON_BIN" -m scripts.chat_eval \
    -i sft \
    -g "$SFT_MODEL_TAG" \
    --max-new-tokens 512 \
    --batch-size 8 \
    --dtype bfloat16 \
    --out "$RESULTS_DIR/chat_eval_sft.json" \
    --task-name "ARC-Easy|ARC-Challenge|MMLU"

# -----------------------------------------------------------------------------
# Final report

"$PYTHON_BIN" -m nanochat_jax.report generate
