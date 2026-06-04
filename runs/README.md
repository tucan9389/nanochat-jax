# runs/

Public reference run scripts.

| Script | Purpose | Hardware |
|---|---|---|
| `speedrun.sh` | d24 base train -> BPB/CORE eval | v6e-8 TPU |

`runs/speedrun.sh` is the only public entrypoint. Experiment commands and TPU smoke commands stay outside `runs/` until they are audited as public paths.

Cost basis for the current result is the 2026-06-04 `us-central1` v6e-8 spot price snapshot: `$4.321096/hr`. Spot pricing and capacity change; recheck before budgeting.

Operating notes:

- `speedrun.sh` writes a base checkpoint under the nanochat-jax cache and reads the same `MODEL_TAG` in `scripts.base_eval`.
- CORE uses `--max-per-task -1` by default. For a quick smoke run, set `CORE_MAX_PER_TASK=50`.
- Copy expensive checkpoints to GCS before deleting a TPU VM.
- Delete TPU VMs and sweep candidate zones after each session.
