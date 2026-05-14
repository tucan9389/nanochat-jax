# runs/

Reproducible end-to-end pipelines.

| Script | Purpose | Hardware | Typical wall | Cost (spot list-price) |
|---|---|---|---|---|
| `speedrun.sh` | Quick demo: tokenizer -> d12 base -> base eval -> tiny SFT | v6e-8 spot (8 chips, ~$8.7/hr) | ~30-60 min | **~$5-7** per full cycle |
| `d24.sh` | Reference: full d24 base + CORE eval, mirrors Karpathy spec | v5p-32 spot (16 chips, ~$25.5/hr us-east5 / ~$20.5/hr europe-west4) | ~8h (train + eval) | **~$165-210** (europe-west4 ~20% cheaper than us-east5) |

Spot pricing fluctuates by region and capacity. For d24, prefer `europe-west4-b` with a region-matched GCS bucket (`gs://...-eu-west4-*`).

## Operating notes

- Always launch multi-host runs on every worker in parallel (`gcloud compute tpus tpu-vm ssh ... --worker=all --command=...`) so `jax.distributed.initialize` can synchronize the coordinator.
- Use `--spot` (preemptible) and check zone capacity before booting (`gcloud compute tpus tpu-vm list --zone=...`).
- For d24 specifically, pass `--matmul-precision highest`. Without it the default fp32 matmul on TPU silently uses bf16 internal accumulation, which measurably degrades converged CORE.
- After any d24 run, copy the checkpoint to GCS **before** deleting the TPU VM. Losing a 5+ GB checkpoint costs hours of training time.
- Always `gcloud compute tpus tpu-vm delete ...` and sweep all candidate zones at the end of a session to confirm no zombie VMs.
