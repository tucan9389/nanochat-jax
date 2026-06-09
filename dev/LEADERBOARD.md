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
| 1 | 7.33* | 0.71879 | 0.27409 | d24 baseline on v6e-8 spot TPU, final checkpoint | Jun 3 2026 | 828485b | @tucan9389 |

`*` Row 1 time is the estimated full 16,704-step train-loop time. Eval and logging are excluded to match the upstream timing convention.

CORE means `max_per_task=-1` and 22/22 tasks. BPB and CORE for row 1 were measured with A100 eval workers. Intermediate checkpoints are not separate leaderboard rows. Row 1 trained on a pre-publish branch state; `828485b` is the public commit verified to reproduce its training path (step-100 and step-2000 loss exact match, BPB difference below `1e-5`).

Row 1 config: `seq_len=2048`, `n_layer=24`, `n_embd=1536`, `n_head=12`, `524288` tokens/step, `16704` total steps, Splash Attention, value embeddings enabled, `ve_grad_impl=onehot`.

## Reproduction reference

`runs/speedrun.sh` is the reproduction path. A full validation run of this
pipeline on v6e-8 spot (June 8 2026, surviving repeated spot preemptions via
checkpoint resume) measured, at the same final step 16703:

| Metric | Row 1 (A100 eval) | Pipeline validation run (TPU eval, bf16) |
|---|---:|---:|
| CORE (22/22 tasks, `max_per_task=-1`) | 0.27409 | 0.27343 |
| val_bpb | 0.71879 | 0.72258 |

Both clear the GPT-2 reference CORE `0.2565`. The two columns are different
training runs evaluated on different hosts, so treat deltas of this size
(about `0.0007` CORE, `0.004` BPB) as the expected reproduction band rather
than a regression. The validation run's measured steady-state rate was
`1.39-1.44s/step` at about 24% MFU; the `1.58s/step` behind the row 1 time
estimate is a conservative basis that includes checkpointing overhead.

## Post-SFT categorical ChatEval

SFT is validated per pipeline release, not per base leaderboard row: base rows
can be re-run and appended above without re-running SFT, and this section only
changes when a new SFT run completes.

The scores below come from the June 8 2026 pipeline validation run (the
TPU-eval column above). Its final base checkpoint (step 16703) was fine-tuned
on the full SFT mixture for 973 steps, and the saved SFT checkpoint was
evaluated with `scripts.chat_eval` (`bfloat16`, all examples per task):

| Task | Accuracy | Correct / Total |
|---|---:|---:|
| ARC-Easy | 66.12% | 1571 / 2376 |
| ARC-Challenge | 50.68% | 594 / 1172 |
| MMLU | 37.08% | 5207 / 14042 |

The generative ChatEval tasks (GSM8K, HumanEval, SpellingBee) are not part of
this record: the current JAX generative eval path is too slow to complete them
within the run budget, which is why the public default ChatEval is the
categorical set.

Cost calculations use v6e-8 spot as the primary basis. On-demand is listed only as a reference. Price snapshot: 2026-06-04 Google Cloud.

| Platform | Price basis | Hourly |
|---|---|---:|
| v6e-8 spot, `us-central1` | 8 TPU chips x `$0.540137/chip-h` | `$4.321096/hr` |
| 8xH100 spot, GCP Americas | 8 GPUs x `$4.2014/GPU-h`, accelerator only | `~$33.61/hr` |
| v6e-8 on-demand, `us-central1` | 8 TPU chips x `$2.70/chip-h` | `$21.60/hr` |

Recheck [Google Cloud TPU pricing](https://cloud.google.com/tpu/pricing), [GPU pricing](https://cloud.google.com/compute/gpus-pricing), and [Spot VM pricing notes](https://docs.cloud.google.com/compute/docs/instances/spot#pricing) before budgeting.

For row 1, `16704` steps at `1.58s/step` gives `7.33h`. The spot train estimate is about `$31.7`; the on-demand equivalent is about `$158.4`. The June 8 validation run measured the full pipeline end to end — base training through SFT and categorical ChatEval, including all spot preemption restarts — at `17.92` active TPU-hours, about `$77` at the spot price basis, within the `$100` TPU budget.

Do not treat sampled CORE (`max_per_task=100`) or `seq_len=4096` / `head_dim=256` probes as nanochat-parity rows.
