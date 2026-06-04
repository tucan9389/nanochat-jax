#!/usr/bin/env bash
# nanochat-jax speedrun: tokenizer + d24 base train + base eval.
#
# Designed for a v6e-8 TPU host with TPU dependencies installed:
#
#     pip install -e ".[tpu,dev]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#
# CORE eval is the default. For a quick smoke run:
#
#     CORE_MAX_PER_TASK=50 NUM_ITERATIONS=250 bash runs/speedrun.sh

set -euo pipefail

export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"

MODEL_TAG="${MODEL_TAG:-d24_speedrun}"
NUM_ITERATIONS="${NUM_ITERATIONS:-16704}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
CORE_MAX_PER_TASK="${CORE_MAX_PER_TASK:--1}"
EVAL_STEPS="${EVAL_STEPS:-20}"
EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-1}"

# 1) Download data for tokenizer training, then the larger train slice.
python -m nanochat_jax.dataset -n 8
python -m nanochat_jax.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!

# 2) Train and evaluate the tokenizer.
python -m scripts.tok_train --max-chars 100000000
python -m scripts.tok_eval

# 3) Pretrain the d24 base model.
echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

python -m scripts.base_train \
    --depth=24 \
    --use-pt-mirror-config \
    --aspect-ratio=64 \
    --head-dim=128 \
    --seq-len=2048 \
    --total-batch-size=524288 \
    --device-batch-size=2 \
    --grad-accum-steps=16 \
    --grad-accum-impl=fused \
    --num-iterations="$NUM_ITERATIONS" \
    --warmup-steps=40 \
    --warmdown-ratio=0.65 \
    --final-lr-frac=0.05 \
    --weight-decay=0.28 \
    --matrix-lr=0.02 \
    --embedding-lr=0.3 \
    --unembedding-lr=0.008 \
    --scalar-lr=0.5 \
    --bf16 \
    --cast-embeddings-bf16 \
    --use-real-data \
    --attn-impl=splash \
    --splash-block-q=512 \
    --splash-block-kv=512 \
    --splash-block-kv-compute=256 \
    --matmul-precision=default \
    --lm-head-precision=highest \
    --ve-grad-impl=onehot \
    --checkpoint-every="$CHECKPOINT_EVERY" \
    --keep-last-checkpoints=2 \
    --model-tag="$MODEL_TAG" \
    --eval-every=0 \
    --no-final-eval \
    --seed=42 \
    --log-every=10

# 4) Evaluate the base model.
python -m scripts.base_eval \
    --source base \
    --model-tag "$MODEL_TAG" \
    --eval bpb,core \
    --eval-steps "$EVAL_STEPS" \
    --device-batch-size "$EVAL_DEVICE_BATCH_SIZE" \
    --max-per-task "$CORE_MAX_PER_TASK"

# 5) Follow-up only: SFT/chat_eval are not part of the default speedrun until
# validated in the JAX publish path.
#
# python -m scripts.chat_sft \
#     --model-tag "$MODEL_TAG" \
#     --num-iterations 100 \
#     --device-batch-size 4 \
#     --max-seq-len 1024 \
#     --bf16 \
#     --sft-scope min
#
# python -m scripts.chat_eval --source sft --model-tag "$MODEL_TAG" --task-name ARC-Easy
