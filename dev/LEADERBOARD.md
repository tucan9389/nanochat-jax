# Leaderboard

Upstream nanochat measures Time-to-GPT-2: training wall-clock time needed to beat GPT-2 `CORE=0.256525` on an 8xH100 node. The upstream rows below are reference context.

## Upstream nanochat (PyTorch + GPU)

Excerpted from upstream nanochat's README summary table (the repo's dev/LEADERBOARD.md holds the fuller history):

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

Same metric as upstream — wall-clock (and cost) to train past the GPT-2 CORE bar `0.256525` — measured on a v6e-8 TPU node instead of 8xH100, hence a separate table. One row per completed public-result run; **row 2 is the current best time-to-GPT-2**.

| # | time (h) | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|---:|---:|---:|---|---|---|---|
| 0 | 168.0 | -- | 0.2565 | OpenAI GPT-2 1.6B reference | 2019 | -- | OpenAI |
| 1 | 7.33* | 0.71879 | 0.27409 | d24 baseline on v6e-8 spot TPU, final checkpoint | Jun 3 2026 | 828485b | @tucan9389 |
| 2 | 5.28** | 0.71655 | 0.25822 | d24, upstream Run 4 recipe (`--recipe 324e69c`), 1M-token batches | Jun 23 2026 | 4d9d88b | @tucan9389 |

`*` Row 1 time is the estimated full 16,704-step train-loop time. Eval and logging are excluded to match the upstream timing convention.
`**` Row 2 time is the measured steady rate (`2.876s/step` at ~24.5% MFU) times its 6,612 steps, same convention.

CORE means `max_per_task=-1` and 22/22 tasks. BPB and CORE for rows 1-2 were measured with A100 eval workers. Intermediate checkpoints are not separate leaderboard rows. Both rows trained on pre-publish branch states; the listed commit is the public commit verified to reproduce the row's training path (row 1: step-100 and step-2000 loss exact match, BPB difference below `1e-5`; row 2: CPU fp32 losses and forward tensors bit-identical to the training code state, and a 30-step real-data v6e-8 rerun wrote a bit-identical checkpoint whose step-250 resume evaluated identically to the original run's step-250 checkpoint).

Row 1 config: `seq_len=2048`, `n_layer=24`, `n_embd=1536`, `n_head=12`, `524288` tokens/step, `16704` total steps, Splash Attention, value embeddings enabled, `ve_grad_impl=onehot`.

Row 2 config: same geometry with the Run 4 recipe axes (every axis is listed in `nanochat_jax/recipes.py`), `1,048,576` tokens/step, `6,612` total steps, bf16 with `lm_head` matmuls at highest precision. The row-2 training command (checkpointing/logging flags omitted; the recipe flag asserts the LR/schedule values below at startup):

```bash
export LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=65536
python -m scripts.base_train \
    --recipe=324e69c \
    --depth=24 --seq-len=2048 --vocab-size=32768 \
    --target-param-data-ratio=9.5 --total-batch-size=1048576 \
    --device-batch-size=2 --grad-accum-steps=32 --grad-accum-impl=fused \
    --warmup-steps=0 --warmdown-ratio=0.5 --final-lr-frac=0.0 \
    --weight-decay=0.2 --matrix-lr=0.02 --embedding-lr=0.3 \
    --unembedding-lr=0.004 --scalar-lr=0.5 \
    --bf16 --cast-embeddings-bf16 --use-real-data \
    --attn-impl=splash --splash-block-q=512 --splash-block-kv=512 --splash-block-kv-compute=256 \
    --matmul-precision=default --lm-head-precision=highest --ve-grad-impl=onehot
```

It matches upstream Run 4 (`0.71854` / `0.2571`) within upstream's own run-to-run band.

Row 2 eval note: this run's checkpoint metadata predates the recipe fields, so evaluation rebuilt the model with the backout scalar active at its untrained `0.2` init (smear is a no-op at zero init). All row 2 numbers use that protocol self-consistently; with backout disabled val_bpb measures `0.00464` lower. Match the metadata-default semantics when re-evaluating.

## Reproduction reference

`runs/speedrun.sh` is the reproduction path. A full validation run of this pipeline on v6e-8 spot (June 8 2026, surviving repeated spot preemptions via checkpoint resume) measured, at the same final step 16703:

| Metric | Row 1 (A100 eval) | Pipeline validation run (TPU eval, bf16) |
|---|---:|---:|
| CORE (22/22 tasks, `max_per_task=-1`) | 0.27409 | 0.27343 |
| val_bpb | 0.71879 | 0.72258 |

Both clear the GPT-2 reference CORE `0.2565`. The two columns are different training runs evaluated on different hosts, so treat deltas of this size (about `0.0007` CORE, `0.004` BPB) as the expected reproduction band rather than a regression. The validation run's measured steady-state rate was `1.39-1.44s/step` at about 24% MFU; the `1.58s/step` behind the row 1 time estimate is a conservative basis that includes checkpointing overhead.

The shipped pipeline was last re-verified end-to-end on real hardware on July 2 2026 (v6e-8 spot, short run at the exact published d24 recipe): parameter count, scaling-param count, and muP weight decay matched this table exactly; training ran at `1.43s/step` / `23.7%` MFU on real data (consistent with the validation run above); checkpoint save/resume, base eval, SFT, ChatEval, and report generation all completed cleanly.

These numbers ride on four small, intentional divergences from upstream nanochat that predate the published runs (decode-window off-by-one, CORE-eval prompt truncation, smear_gate init, SpellingBee SFT templates). "Fixing" any of them changes what this table (or the SFT section below) reproduces; each is documented in `dev/REPRODUCTION-GUARDS.md` and marked with a NOTE comment at its code site.

## Post-SFT categorical ChatEval

SFT is validated per pipeline release, not per base leaderboard row: base rows can be re-run and appended above without re-running SFT, and this section only changes when a new SFT run completes.

The scores below come from the June 8 2026 pipeline validation run (the TPU-eval column above). Its final base checkpoint (step 16703) was fine-tuned on the full SFT mixture for 973 steps, and the saved SFT checkpoint was evaluated with `scripts.chat_eval` (`bfloat16`, all examples per task):

| Task | Accuracy | Correct / Total |
|---|---:|---:|
| ARC-Easy | 66.12% | 1571 / 2376 |
| ARC-Challenge | 50.68% | 594 / 1172 |
| MMLU | 37.08% | 5207 / 14042 |

The generative ChatEval tasks (GSM8K, HumanEval, SpellingBee) are not part of this record: the current JAX generative eval path is too slow to complete them within the run budget, which is why the public default ChatEval is the categorical set.

Cost calculations use v6e-8 spot as the primary basis. On-demand is listed only as a reference. Price snapshot: 2026-06-04 Google Cloud.

| Platform | Price basis | Hourly |
|---|---|---:|
| v6e-8 spot, `us-central1` | 8 TPU chips x `$0.540137/chip-h` | `$4.321096/hr` |
| 8xH100 spot, GCP Americas | 8 GPUs x `$4.2014/GPU-h`, accelerator only | `~$33.61/hr` |
| v6e-8 on-demand, `us-central1` | 8 TPU chips x `$2.70/chip-h` | `$21.60/hr` |

Recheck [Google Cloud TPU pricing](https://cloud.google.com/tpu/pricing), [GPU pricing](https://cloud.google.com/compute/gpus-pricing), and [Spot VM pricing notes](https://docs.cloud.google.com/compute/docs/instances/spot#pricing) before budgeting.

For row 1, `16704` steps at `1.58s/step` gives `7.33h` — about `$31.7` spot (`$158.4` on-demand). For row 2, `6612` steps at the measured `2.876s/step` gives `5.28h` — about `$22.8` spot (`$114` on-demand), the current best cost-to-GPT-2 here. The June 8 validation run measured the full pipeline end to end — base training through SFT and categorical ChatEval, including all spot preemption restarts — at `17.92` active TPU-hours, about `$77` at the spot price basis, within the `$100` TPU budget.

Do not treat sampled CORE (`max_per_task=100`) or `seq_len=4096` / `head_dim=256` probes as nanochat-parity rows.
