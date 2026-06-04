# Leaderboard

Upstream nanochat measures Time-to-GPT-2: training wall-clock time needed to beat GPT-2 `CORE=0.256525` on an 8xH100 node. The upstream rows below are reference context.

## Upstream nanochat (PyTorch + GPU)

Excerpted from [`karpathy/nanochat/dev/LEADERBOARD.md`](https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md):

| # | time (h) | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|---:|---:|---:|---|---|---|---|
| 0 | 168.0 | -- | 0.2565 | OpenAI GPT-2 1.6B reference | 2019 | -- | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained, +fp8 | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | total_batch=1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | dataset = NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch round 1 | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

Upstream reports training time excluding evaluation and logging time.

## nanochat-jax (JAX + TPU)

This table uses the upstream append style: one row per completed public-result run. It is separate from upstream Time-to-GPT-2 because the hardware, runtime, and eval hosts differ.

| # | time (h) | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|---:|---:|---:|---|---|---|---|
| 0 | 168.0 | -- | 0.2565 | OpenAI GPT-2 1.6B reference | 2019 | -- | OpenAI |
| 1 | 7.33* | 0.71879 | 0.27409 | d24 baseline on v6e-8 spot TPU, final checkpoint | Jun 3 2026 | 956e043 | @tucan9389 |

`*` Row 1 time is the estimated full 16,704-step train-loop time. Eval and logging are excluded to match the upstream timing convention.

CORE means `max_per_task=-1` and 22/22 tasks. BPB and CORE for row 1 were measured with A100 eval workers. Intermediate checkpoints are not separate leaderboard rows.

Row 1 config: `seq_len=2048`, `n_layer=24`, `n_embd=1536`, `n_head=12`, `524288` tokens/step, `16704` total steps, Splash Attention, value embeddings enabled, `ve_grad_impl=onehot`.

Cost calculations use v6e-8 spot as the primary basis. On-demand is listed only as a reference. Price snapshot: 2026-06-04 Google Cloud.

| Platform | Price basis | Hourly |
|---|---|---:|
| v6e-8 spot, `us-central1` | 8 TPU chips x `$0.540137/chip-h` | `$4.321096/hr` |
| 8xH100 spot, GCP Americas | 8 GPUs x `$4.2014/GPU-h`, accelerator only | `~$33.61/hr` |
| v6e-8 on-demand, `us-central1` | 8 TPU chips x `$2.70/chip-h` | `$21.60/hr` |

Recheck [Google Cloud TPU pricing](https://cloud.google.com/tpu/pricing), [GPU pricing](https://cloud.google.com/compute/gpus-pricing), and [Spot VM pricing notes](https://docs.cloud.google.com/compute/docs/instances/spot#pricing) before budgeting.

For row 1, `16704` steps at `1.58s/step` gives `7.33h`. The spot train estimate is about `$31.7`; the on-demand equivalent is about `$158.4`. The measured spot run stayed within the `$100` TPU budget.

Do not treat sampled CORE (`max_per_task=100`) or `seq_len=4096` / `head_dim=256` probes as nanochat-parity rows.
