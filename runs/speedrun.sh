#!/usr/bin/env bash
# nanochat-jax speedrun: tokenizer + d24 base train/eval + SFT + ChatEval + report.
#
# Designed for a v6e-8 TPU host with TPU dependencies installed:
#
#     pip install -e ".[tpu,dev]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#
# Full production mode follows the public d24 TPU shape. For wiring validation
# without reusing old d12 checkpoints:
#
#     SPEEDRUN_SMOKE=1 bash runs/speedrun.sh

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
SPEEDRUN_SMOKE="${SPEEDRUN_SMOKE:-0}"
WANDB_RUN="${WANDB_RUN:-dummy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
fi

if [[ -z "${NANOCHAT_JAX_BASE_DIR:-}" ]]; then
    if [[ "$SPEEDRUN_SMOKE" == "1" ]]; then
        export NANOCHAT_JAX_BASE_DIR="$HOME/.cache/nanochat-jax-smoke"
    else
        export NANOCHAT_JAX_BASE_DIR="$HOME/.cache/nanochat-jax"
    fi
else
    export NANOCHAT_JAX_BASE_DIR
fi
mkdir -p "$NANOCHAT_JAX_BASE_DIR"

if [[ "$SPEEDRUN_SMOKE" == "1" ]]; then
    export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
    MODEL_TAG="${MODEL_TAG:-speedrun_smoke_d2}"
    BASE_DEPTH="${BASE_DEPTH:-2}"
    BASE_SEQ_LEN="${BASE_SEQ_LEN:-256}"
    BASE_TOTAL_BATCH_SIZE="${BASE_TOTAL_BATCH_SIZE:-256}"
    BASE_DEVICE_BATCH_SIZE="${BASE_DEVICE_BATCH_SIZE:-1}"
    BASE_GRAD_ACCUM_STEPS="${BASE_GRAD_ACCUM_STEPS:-1}"
    NUM_ITERATIONS="${NUM_ITERATIONS:-2}"
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1}"
    BASE_KEEP_LAST_CHECKPOINTS="${BASE_KEEP_LAST_CHECKPOINTS:-2}"
    BASE_RESUME_FROM_STEP="${BASE_RESUME_FROM_STEP:--1}"
    DATASET_TOKENIZER_SHARDS="${DATASET_TOKENIZER_SHARDS:-1}"
    DATASET_TRAIN_SHARDS="${DATASET_TRAIN_SHARDS:-1}"
    TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-100000}"
    TOKENIZER_DOC_CAP="${TOKENIZER_DOC_CAP:-2000}"
    TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-8192}"
    BASE_VOCAB_SIZE="${BASE_VOCAB_SIZE:-$TOKENIZER_VOCAB_SIZE}"
    BASE_EVAL_MODES="${BASE_EVAL_MODES:-sample}"
    CORE_MAX_PER_TASK="${CORE_MAX_PER_TASK:-1}"
    EVAL_STEPS="${EVAL_STEPS:-1}"
    EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-1}"
    BASE_EVAL_PRECISION_ARGS=()
    SFT_MODEL_TAG="${SFT_MODEL_TAG:-${MODEL_TAG}_sft_smoke}"
    SFT_SCOPE="${SFT_SCOPE:-min}"
    SFT_NUM_ITERATIONS="${SFT_NUM_ITERATIONS:-1}"
    SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-1}"
    SFT_TOTAL_BATCH_SIZE="${SFT_TOTAL_BATCH_SIZE:-256}"
    SFT_MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-256}"
    SFT_EVAL_EVERY="${SFT_EVAL_EVERY:--1}"
    SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-1}"
    SFT_CHATCORE_EVERY="${SFT_CHATCORE_EVERY:--1}"
    SFT_CHATCORE_MAX_CAT="${SFT_CHATCORE_MAX_CAT:--1}"
    SFT_CHATCORE_MAX_SAMPLE="${SFT_CHATCORE_MAX_SAMPLE:-24}"
    CHAT_EVAL_TASK_NAME="${CHAT_EVAL_TASK_NAME:-ARC-Easy}"
    CHAT_EVAL_MAX_PROBLEMS="${CHAT_EVAL_MAX_PROBLEMS:-1}"
    CHAT_EVAL_MAX_NEW_TOKENS="${CHAT_EVAL_MAX_NEW_TOKENS:-32}"
    CHAT_EVAL_BATCH_SIZE="${CHAT_EVAL_BATCH_SIZE:-1}"
    CHAT_EVAL_DTYPE="${CHAT_EVAL_DTYPE:-float32}"
    BASE_DISTRIBUTED_ARGS=(--no-distributed)
    SFT_DISTRIBUTED_ARGS=(--no-distributed)
    BASE_PRECISION_ARGS=(--no-bf16)
    BASE_DATA_ARGS=()
    BASE_ATTN_ARGS=(--attn-impl=xla)
    BASE_GRAD_ACCUM_IMPL="${BASE_GRAD_ACCUM_IMPL:-loop}"
else
    MODEL_TAG="${MODEL_TAG:-d24_speedrun}"
    BASE_DEPTH="${BASE_DEPTH:-24}"
    BASE_SEQ_LEN="${BASE_SEQ_LEN:-2048}"
    BASE_TOTAL_BATCH_SIZE="${BASE_TOTAL_BATCH_SIZE:-524288}"
    BASE_DEVICE_BATCH_SIZE="${BASE_DEVICE_BATCH_SIZE:-2}"
    BASE_GRAD_ACCUM_STEPS="${BASE_GRAD_ACCUM_STEPS:-16}"
    NUM_ITERATIONS="${NUM_ITERATIONS:-16704}"
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
    BASE_KEEP_LAST_CHECKPOINTS="${BASE_KEEP_LAST_CHECKPOINTS:-2}"
    BASE_RESUME_FROM_STEP="${BASE_RESUME_FROM_STEP:--1}"
    DATASET_TOKENIZER_SHARDS="${DATASET_TOKENIZER_SHARDS:-8}"
    DATASET_TRAIN_SHARDS="${DATASET_TRAIN_SHARDS:-170}"
    TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-100000000}"
    TOKENIZER_DOC_CAP="${TOKENIZER_DOC_CAP:-10000}"
    TOKENIZER_VOCAB_SIZE="${TOKENIZER_VOCAB_SIZE:-32768}"
    BASE_VOCAB_SIZE="${BASE_VOCAB_SIZE:-$TOKENIZER_VOCAB_SIZE}"
    BASE_EVAL_MODES="${BASE_EVAL_MODES:-bpb,core}"
    CORE_MAX_PER_TASK="${CORE_MAX_PER_TASK:--1}"
    EVAL_STEPS="${EVAL_STEPS:-20}"
    EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-1}"
    BASE_EVAL_PRECISION_ARGS=(--bf16)
    SFT_MODEL_TAG="${SFT_MODEL_TAG:-${MODEL_TAG}_sft}"
    SFT_SCOPE="${SFT_SCOPE:-full}"
    SFT_NUM_ITERATIONS="${SFT_NUM_ITERATIONS:--1}"
    SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-1}"
    SFT_TOTAL_BATCH_SIZE="${SFT_TOTAL_BATCH_SIZE:-524288}"
    SFT_MAX_SEQ_LEN="${SFT_MAX_SEQ_LEN:-2048}"
    SFT_EVAL_EVERY="${SFT_EVAL_EVERY:-200}"
    SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-4}"
    SFT_CHATCORE_EVERY="${SFT_CHATCORE_EVERY:-200}"
    SFT_CHATCORE_MAX_CAT="${SFT_CHATCORE_MAX_CAT:--1}"
    SFT_CHATCORE_MAX_SAMPLE="${SFT_CHATCORE_MAX_SAMPLE:-24}"
    CHAT_EVAL_TASK_NAME="${CHAT_EVAL_TASK_NAME:-ARC-Easy|ARC-Challenge|MMLU}"
    CHAT_EVAL_MAX_PROBLEMS="${CHAT_EVAL_MAX_PROBLEMS:--1}"
    CHAT_EVAL_MAX_NEW_TOKENS="${CHAT_EVAL_MAX_NEW_TOKENS:-512}"
    CHAT_EVAL_BATCH_SIZE="${CHAT_EVAL_BATCH_SIZE:-8}"
    CHAT_EVAL_DTYPE="${CHAT_EVAL_DTYPE:-bfloat16}"
    BASE_DISTRIBUTED_ARGS=()
    SFT_DISTRIBUTED_ARGS=()
    BASE_PRECISION_ARGS=(--bf16 --cast-embeddings-bf16)
    BASE_DATA_ARGS=(--use-real-data)
    BASE_ATTN_ARGS=(
        --attn-impl=splash
        --splash-block-q=512
        --splash-block-kv=512
        --splash-block-kv-compute=256
    )
    BASE_GRAD_ACCUM_IMPL="${BASE_GRAD_ACCUM_IMPL:-fused}"
fi

RESULTS_DIR="${RESULTS_DIR:-$NANOCHAT_JAX_BASE_DIR/results/$MODEL_TAG}"
mkdir -p "$RESULTS_DIR"

echo "[speedrun] mode=$([[ "$SPEEDRUN_SMOKE" == "1" ]] && echo smoke || echo production)"
echo "[speedrun] base model tag: $MODEL_TAG"
echo "[speedrun] sft model tag:  $SFT_MODEL_TAG"
echo "[speedrun] results dir:    $RESULTS_DIR"

# -----------------------------------------------------------------------------
# Report reset

"$PYTHON_BIN" -m nanochat_jax.report reset

# -----------------------------------------------------------------------------
# Tokenizer

"$PYTHON_BIN" -m nanochat_jax.dataset -n "$DATASET_TOKENIZER_SHARDS"
"$PYTHON_BIN" -m nanochat_jax.dataset -n "$DATASET_TRAIN_SHARDS" &
DATASET_DOWNLOAD_PID=$!

"$PYTHON_BIN" -m scripts.tok_train \
    --max-chars "$TOKENIZER_MAX_CHARS" \
    --doc-cap "$TOKENIZER_DOC_CAP" \
    --vocab-size "$TOKENIZER_VOCAB_SIZE"
"$PYTHON_BIN" -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

"$PYTHON_BIN" -m scripts.base_train \
    --depth="$BASE_DEPTH" \
    --use-pt-mirror-config \
    --aspect-ratio=64 \
    --head-dim=128 \
    --seq-len="$BASE_SEQ_LEN" \
    --vocab-size="$BASE_VOCAB_SIZE" \
    --total-batch-size="$BASE_TOTAL_BATCH_SIZE" \
    --device-batch-size="$BASE_DEVICE_BATCH_SIZE" \
    --grad-accum-steps="$BASE_GRAD_ACCUM_STEPS" \
    --grad-accum-impl="$BASE_GRAD_ACCUM_IMPL" \
    --num-iterations="$NUM_ITERATIONS" \
    --warmup-steps=40 \
    --warmdown-ratio=0.65 \
    --final-lr-frac=0.05 \
    --weight-decay=0.28 \
    --matrix-lr=0.02 \
    --embedding-lr=0.3 \
    --unembedding-lr=0.008 \
    --scalar-lr=0.5 \
    ${BASE_PRECISION_ARGS[@]+"${BASE_PRECISION_ARGS[@]}"} \
    ${BASE_DATA_ARGS[@]+"${BASE_DATA_ARGS[@]}"} \
    ${BASE_ATTN_ARGS[@]+"${BASE_ATTN_ARGS[@]}"} \
    --matmul-precision=default \
    --lm-head-precision=highest \
    --ve-grad-impl=onehot \
    --checkpoint-every="$CHECKPOINT_EVERY" \
    --keep-last-checkpoints="$BASE_KEEP_LAST_CHECKPOINTS" \
    --resume-from-step="$BASE_RESUME_FROM_STEP" \
    --model-tag="$MODEL_TAG" \
    --eval-every=0 \
    --no-final-eval \
    --seed=42 \
    --log-every=10 \
    ${BASE_DISTRIBUTED_ARGS[@]+"${BASE_DISTRIBUTED_ARGS[@]}"}

"$PYTHON_BIN" -m scripts.base_eval \
    --source base \
    --model-tag "$MODEL_TAG" \
    --eval "$BASE_EVAL_MODES" \
    --eval-steps "$EVAL_STEPS" \
    --device-batch-size "$EVAL_DEVICE_BATCH_SIZE" \
    --max-per-task "$CORE_MAX_PER_TASK" \
    --out "$RESULTS_DIR/base_eval.json" \
    --partial-out "$RESULTS_DIR/base_eval_core_partial.json" \
    ${BASE_EVAL_PRECISION_ARGS[@]+"${BASE_EVAL_PRECISION_ARGS[@]}"} \
    ${BASE_DISTRIBUTED_ARGS[@]+"${BASE_DISTRIBUTED_ARGS[@]}"}

# -----------------------------------------------------------------------------
# SFT

SFT_ARGS=(
    --model-tag "$MODEL_TAG"
    --output-model-tag "$SFT_MODEL_TAG"
    --num-iterations "$SFT_NUM_ITERATIONS"
    --device-batch-size "$SFT_DEVICE_BATCH_SIZE"
    --max-seq-len "$SFT_MAX_SEQ_LEN"
    --total-batch-size "$SFT_TOTAL_BATCH_SIZE"
    --sft-scope "$SFT_SCOPE"
    --eval-every "$SFT_EVAL_EVERY"
    --eval-steps "$SFT_EVAL_STEPS"
    --chatcore-every "$SFT_CHATCORE_EVERY"
    --chatcore-max-cat "$SFT_CHATCORE_MAX_CAT"
    --chatcore-max-sample "$SFT_CHATCORE_MAX_SAMPLE"
    --load-optimizer 1
    --log-every 10
    --run "$WANDB_RUN"
)
if ((${#SFT_DISTRIBUTED_ARGS[@]})); then
    SFT_ARGS+=("${SFT_DISTRIBUTED_ARGS[@]}")
fi
if [[ "$SPEEDRUN_SMOKE" != "1" ]]; then
    SFT_ARGS+=(--bf16)
fi
"$PYTHON_BIN" -m scripts.chat_sft "${SFT_ARGS[@]}"

# -----------------------------------------------------------------------------
# ChatEval

CHAT_EVAL_ARGS=(
    -i sft
    -g "$SFT_MODEL_TAG"
    --max-new-tokens "$CHAT_EVAL_MAX_NEW_TOKENS"
    --batch-size "$CHAT_EVAL_BATCH_SIZE"
    --dtype "$CHAT_EVAL_DTYPE"
    --out "$RESULTS_DIR/chat_eval_sft.json"
)
if [[ -n "$CHAT_EVAL_TASK_NAME" ]]; then
    CHAT_EVAL_ARGS+=(--task-name "$CHAT_EVAL_TASK_NAME")
fi
if [[ "$CHAT_EVAL_MAX_PROBLEMS" != "-1" ]]; then
    CHAT_EVAL_ARGS+=(--max-problems "$CHAT_EVAL_MAX_PROBLEMS")
fi
"$PYTHON_BIN" -m scripts.chat_eval "${CHAT_EVAL_ARGS[@]}"

# -----------------------------------------------------------------------------
# Final report

"$PYTHON_BIN" -m nanochat_jax.report generate
