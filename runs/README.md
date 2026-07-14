# runs/

Public reference run scripts.

| Script | Purpose | Hardware |
|---|---|---|
| `speedrun.sh` | tokenizer -> d24 base train/eval -> SFT -> ChatEval -> report | v6e-8 TPU |
| `runcpu.sh` | the same pipeline on a tiny d2 model (wiring check only) | CPU |

`runs/speedrun.sh` runs the production d24 pipeline; `runs/runcpu.sh` runs the same stages on a tiny d2 model for a CPU wiring check. Both mirror the upstream nanochat experience while preserving the validated JAX d24 TPU train shape.

Production:

```bash
bash runs/speedrun.sh
```

CPU smoke (wiring check only):

```bash
bash runs/runcpu.sh
```

`runcpu.sh` creates a fresh tiny d2 checkpoint and validates wiring only. It defaults to `~/.cache/nanochat-jax-smoke`, forces CPU unless `JAX_PLATFORMS` is already set, and is not quality evidence.

Default production stages:

1. `python -m nanochat_jax.report reset`
2. dataset download
3. tokenizer train/eval
4. d24 base training
5. base BPB/CORE eval
6. SFT
7. SFT ChatEval
8. `python -m nanochat_jax.report generate`

The speedrun's ChatEval scores the categorical task set (`ARC-Easy|ARC-Challenge|MMLU`) in one batched pass, then the generative tasks (GSM8K, HumanEval, SpellingBee) one at a time on the jitted decode path (`--jit-gen 1`). The combined ChatCORE lands in the final report; published scores are in `dev/LEADERBOARD.md`. To score a task manually against the SFT checkpoint, e.g.:

```bash
python -m scripts.chat_eval -i sft -g d24_speedrun_r4_sft --task-name "GSM8K" --jit-gen 1
```

The recipe itself (model geometry, batch/horizon, vocab size, task set) is hardcoded in the script, like upstream nanochat's speedrun. (The script trains the upstream Run-4 recipe `324e69c`; `scripts/base_train.py --recipe dc54a1a` selects the row-1 baseline instead — see `nanochat_jax/recipes.py` for every axis.) A few operational knobs are env-overridable:

| Variable | Default | Meaning |
|---|---:|---|
| `MODEL_TAG` | `d24_speedrun_r4` | Base checkpoint tag |
| `SFT_MODEL_TAG` | `${MODEL_TAG}_sft` | SFT checkpoint tag |
| `DEVICE_BATCH_SIZE` | `2` | Per-device batch size (lower to `1` if you OOM) |
| `NUM_ITERATIONS` | `-1` | Base-train horizon; `-1` derives it from `--target-param-data-ratio=9.5` (6,612 steps) |
| `NANOCHAT_JAX_BASE_DIR` | `~/.cache/nanochat-jax` | Root for checkpoints / tokenizer / results |
| `RESULTS_DIR` | `$NANOCHAT_JAX_BASE_DIR/results/$MODEL_TAG` | JSON artifacts |
| `PYTHON_BIN` | `python` | Interpreter used for every stage |

Cost basis for the published results is the 2026-06-04 `us-central1` v6e-8 spot price snapshot: `$4.321096/hr`. Spot pricing and capacity change; recheck before budgeting.

Expected reproduction values (TPU-eval CORE and BPB at the final step) and the measured end-to-end cost are listed in [`dev/LEADERBOARD.md`](../dev/LEADERBOARD.md) under "Reproduction reference".

Operating notes:

- On TPU VMs, preinstall CPU-only torch before `pip install -e ".[tpu,dev]"` to avoid pulling CUDA wheels onto the boot disk: `pip install "torch~=2.11.0" --index-url https://download.pytorch.org/whl/cpu`.
- `speedrun.sh` writes the base checkpoint under `base_checkpoints/$MODEL_TAG` and the SFT checkpoint under `chatsft_checkpoints/$SFT_MODEL_TAG`.
- Base CORE uses `--max-per-task -1` by default in production.
- During SFT, the periodic in-loop ChatCORE is disabled by default (`--chatcore-every -1`): it runs the eager decode path (minutes per problem for d24), so those eval pauses — not the training steps — used to dominate the SFT stage's wall-clock. The speedrun scores all six tasks after SFT on the jitted decode path instead.
- `report.md` is generated both under `$NANOCHAT_JAX_BASE_DIR/report/` and in the repo root for convenience (the repo-root copy is git-ignored, not shipped).
- Copy expensive checkpoints to GCS before deleting a TPU VM.
- Delete the TPU VM when you are done. If you tried creating TPUs in more than one zone, list each of those zones afterwards to catch leftover VMs — an orphaned TPU keeps billing.
