#!/usr/bin/env bash
# nanochat-jax speedrun: tokenizer + d12 base train + base eval + chat SFT.
#
# Designed for a single TPU host (v6e-8 spot is the cheapest practical target).
# Total cost is roughly $1-3 depending on capacity and preemptions. Run this
# from the project root after activating the venv with TPU support installed::
#
#     pip install -e ".[tpu,dev]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
#
# Steps:
#   1) Download a small ClimbMix slice (-n 8 shards is enough for tokenizer training).
#   2) Train the BPE tokenizer.
#   3) Pretrain a depth-12 base model for ~250 steps.
#   4) Evaluate the base model (BPB + sample + CORE).
#   5) Optional: SFT on a tiny mixture (sft-scope min) and evaluate again.

set -euo pipefail

# 1) Download dataset shards.
python -m nanochat_jax.dataset -n 8

# 2) Train the tokenizer.
python -m scripts.tok_train --max-chars 100000000

# 3) Pretrain a tiny d12 base model.
python -m scripts.base_train \
    --depth 12 \
    --num-iterations 250 \
    --device-batch-size 4 \
    --seq-len 1024 \
    --bf16 \
    --weights-out runs/d12_speedrun.npz

# 4) Evaluate the base model.
python -m scripts.base_eval \
    --source base \
    --eval bpb,sample,core \
    --eval-steps 8 \
    --device-batch-size 4 \
    --bf16 \
    --max-per-task 50

# 5) (Optional) Quick SFT on a tiny mixture and evaluate.
python -m scripts.chat_sft \
    --num-iterations 100 \
    --device-batch-size 4 \
    --max-seq-len 1024 \
    --bf16 \
    --sft-scope min

python -m scripts.chat_eval --source sft --task-name ARC-Easy
