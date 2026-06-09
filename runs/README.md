# runs/

Public reference run scripts.

| Script | Purpose | Hardware |
|---|---|---|
| `speedrun.sh` | tokenizer -> d24 base train/eval -> SFT -> ChatEval -> report | v6e-8 TPU |

`runs/speedrun.sh` is the only public entrypoint. It mirrors the upstream nanochat experience while preserving the validated JAX d24 TPU train shape.

Production:

```bash
bash runs/speedrun.sh
```

Smoke:

```bash
SPEEDRUN_SMOKE=1 bash runs/speedrun.sh
```

The smoke path creates a fresh tiny d2 checkpoint and validates wiring only. It defaults to `~/.cache/nanochat-jax-smoke`, forces CPU unless `JAX_PLATFORMS` is already set, and is not quality evidence.

Default production stages:

1. `python -m nanochat_jax.report reset`
2. dataset download
3. tokenizer train/eval
4. d24 base training
5. base BPB/CORE eval
6. SFT
7. SFT ChatEval
8. `python -m nanochat_jax.report generate`

Default SFT ChatEval runs the validated categorical task set:
`ARC-Easy|ARC-Challenge|MMLU`. Full six-task ChatEval remains available as an
opt-in run:

```bash
CHAT_EVAL_TASK_NAME="ARC-Easy|ARC-Challenge|MMLU|GSM8K|HumanEval|SpellingBee" bash runs/speedrun.sh
```

Key environment overrides:

| Variable | Default | Meaning |
|---|---:|---|
| `MODEL_TAG` | `d24_speedrun` | Base checkpoint tag |
| `SFT_MODEL_TAG` | `${MODEL_TAG}_sft` | SFT checkpoint tag |
| `SFT_SCOPE` | `full` | SFT mixture scope for production |
| `CHAT_EVAL_TASK_NAME` | `ARC-Easy\|ARC-Challenge\|MMLU` | Pipe-separated ChatEval task set |
| `CHAT_EVAL_MAX_PROBLEMS` | `-1` | `-1` means all examples for the selected ChatEval tasks |
| `TOKENIZER_VOCAB_SIZE` | `32768` | Tokenizer vocab size; smoke defaults to `8192` |
| `BASE_VOCAB_SIZE` | `$TOKENIZER_VOCAB_SIZE` | Base model vocab size, kept aligned with tokenizer |
| `RESULTS_DIR` | `$NANOCHAT_JAX_BASE_DIR/results/$MODEL_TAG` | JSON artifacts |

Cost basis for the current result is the 2026-06-04 `us-central1` v6e-8 spot price snapshot: `$4.321096/hr`. Spot pricing and capacity change; recheck before budgeting.

Operating notes:

- On TPU VMs, preinstall CPU-only torch before `pip install -e ".[tpu,dev]"`
  to avoid pulling CUDA wheels onto the boot disk:
  `pip install "torch~=2.11.0" --index-url https://download.pytorch.org/whl/cpu`.
- `speedrun.sh` writes the base checkpoint under `base_checkpoints/$MODEL_TAG` and the SFT checkpoint under `chatsft_checkpoints/$SFT_MODEL_TAG`.
- Base CORE uses `--max-per-task -1` by default in production.
- `report.md` is generated both under `$NANOCHAT_JAX_BASE_DIR/report/` and in the repo root for convenience.
- Copy expensive checkpoints to GCS before deleting a TPU VM.
- Delete TPU VMs and sweep candidate zones after each session.
