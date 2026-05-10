#!/usr/bin/env bash
# nanochat-jax d24 reference run.
#
# This mirrors Karpathy's d24 spec (depth=24, target_param_data_ratio=12,
# total_batch_size=524288) and is the closest comparison to the upstream
# leaderboard.
#
# Multi-host TPU is required for reasonable wall-clock. v5p-32 spot
# (4 hosts x 4 chips) is the cheapest practical target; expect ~6h wall
# and ~$30-50 (spot, region-dependent). Always launch on every worker
# in parallel via ``--worker=all`` so ``jax.distributed.initialize``
# can synchronize the coordinator.
#
# CRITICAL: enable ``--matmul-precision highest`` for d24. Without it the
# default fp32 matmul on TPU silently uses bf16 internal accumulation, which
# significantly degrades the converged CORE.
#
# After the run, back up the checkpoint to GCS BEFORE deleting the TPU VM
# (see docs/03-tpu-runbook.md / OPS_PLAYBOOK).

set -euo pipefail

# 1) Download the full ClimbMix slice (170 shards is sufficient).
python -m nanochat_jax.dataset -n 170

# 2) Train the tokenizer if needed.
if [ ! -f "$HOME/.cache/nanochat-jax/tokenizer/tokenizer.pkl" ]; then
    python -m scripts.tok_train
fi

# 3) Train the d24 base model.
python -m scripts.base_train \
    --depth 24 \
    --use-pt-mirror-config \
    --aspect-ratio 64 \
    --head-dim 128 \
    --target-param-data-ratio 12 \
    --total-batch-size 524288 \
    --grad-accum-steps 2 \
    --device-batch-size 4 \
    --seq-len 2048 \
    --bf16 \
    --matmul-precision highest \
    --checkpoint-every 2000 \
    --keep-last-checkpoints 2 \
    --weights-out runs/d24.npz

# 4) Evaluate.
python -m scripts.base_eval \
    --source base \
    --eval bpb,sample,core \
    --eval-steps 20 \
    --device-batch-size 4 \
    --bf16 \
    --max-per-task 100
