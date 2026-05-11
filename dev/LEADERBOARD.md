# Leaderboard

Reference numbers from this repository alongside upstream nanochat for
comparison. The primary metric is **CORE** (DCLM 22-task ICL), as in upstream
nanochat ("time to GPT-2"). `val_bpb` is the secondary metric.

> Cross-framework / cross-device numerical agreement is bounded by the
> bf16-on-TPU mantissa floor (~9% per published cross-device studies and ~6.4%
> within Karpathy's own commit-to-commit variance). Numbers below should be
> read with that floor in mind.

## Upstream nanochat (PyTorch + GPU, reference)

Excerpted from
[`karpathy/nanochat/dev/LEADERBOARD.md`](https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md):

| # | wall (h) | val_bpb | CORE | Description | Date | Commit |
|---|---------|---------|--------|-------------|------|--------|
| 0 | 168.0 | -- | 0.2565 | OpenAI GPT-2 1.6B (reference) | 2019 | -- |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained, +fp8 | Feb 2 2026 | a67eba3 |
| 3 | 2.76 | 0.74645 | 0.2602 | total_batch=1M tokens | Feb 5 2026 | 2c062aa |
| 4 | 2.02 | 0.71854 | 0.2571 | dataset = NVIDIA ClimbMix | Mar 4 2026 | 324e69c |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch round 1 | Mar 9 2026 | 6ed7d1d |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 |

## nanochat-jax (this repository, JAX + TPU)

Two paired d24 runs, identical except for `--matmul-precision`. Both are full
16,704-iteration trains on a multi-host v5p-32 spot pod mirroring Karpathy's
Run 1 spec (batch 524,288, target-param-data-ratio 12, ClimbMix 170 shards,
bf16, xla attention).

| Date | Hardware | Model | Steps | val_bpb | CORE | Notes |
|---|---|---|---|---|---|---|
| 2026-05-04 | v5p-32 spot, ~6 h, ~$20 | d24, bf16, ClimbMix 170 shards, **default precision** | 16704 | 0.832 | 0.1774 (22-task) | Baseline: Karpathy d24 spec mirror, xla attention, no `--matmul-precision` flag. Tier 2 only; 31% CORE gap from the upstream d24 baseline 0.2585. Same config as the next row minus the precision flag. |
| 2026-05-08 | v5p-32 spot, ~6 h, ~$32 | d24, bf16 + **`--matmul-precision highest`**, ClimbMix 170 shards | 16704 | 0.7596 | 0.227 (22-task) | One flag change vs the row above → **+27.6% CORE relative**. val_bpb enters Karpathy's same-config 6.4% variance band (`0.71854 +- 6.4%`); CORE reaches 88.5% of the GPT-2 threshold (0.2565). Without HIGHEST, JAX's default fp32 matmul on TPU silently uses bf16 internal accumulation, materially degrading converged CORE on d24+. |

## Methodology

- `val_bpb` is computed via `python -m scripts.base_eval --eval bpb`. The
  validation split is the last shard of ClimbMix.
- `CORE` is the DCLM 22-task ICL average, computed via
  `python -m scripts.base_eval --eval core --max-per-task 100`.
- "Spot" pricing is the price reported on the GCP console at the time of the
  run; you may pay more or less depending on region and capacity.

## Implementation notes (TPU-specific)

These are JAX/TPU surprises encountered while reproducing the d24 spec on a
v5p-32 pod. They are not in the upstream nanochat repo because they only show
up on TPU. None of them are bugs in the spec itself.

- **`--matmul-precision highest` is required for d24+.** This is the single
  largest knob on TPU. JAX's default fp32 matmul on TPU silently uses bf16
  internal accumulation for the matmul output, which materially degrades
  converged CORE (paired runs above: 0.1774 → 0.227, +27.6%). The cost is a
  modest sec/step slowdown (~5-10% on v5p in practice; the published "2.5×"
  figure does not match what we measured on v5p MXUs). At d12 scale and short
  trains the effect is in the noise; the regime where it matters is roughly
  d24 + full 16,704-step trains.
- **Splash Attention and `--matmul-precision=highest` are not jointly
  supported in jax 0.10.** Splash is integrated (see `scripts/verify_splash.py`)
  and slightly faster than xla on its own, but combining the two raises a
  MosaicError. The d24 reference run uses `--attn-impl xla` so HIGHEST can
  apply globally. Verified numerical agreement between xla and Splash is in
  `scripts/verify_splash.py`.
- **muP weight decay scaling.** Upstream nanochat scales
  `weight_decay` by `sqrt(B/B_REF) * (D_REF/target_tokens)` (see Karpathy
  `scripts/base_train.py` and `dev/LOG.md` 2026-01-10 entry). The JAX port
  initially missed this; for d24 it means the configured `--weight-decay=0.28`
  is automatically rescaled to ~0.042 (×0.151). The scaling is in place; we
  measured it to be CORE-noise-neutral at d24 (within ±0.003) but kept it
  because it matches upstream semantics and may matter at other depths.
- **ClimbMix needs ≥150 shards.** Upstream `dev/LOG.md` 2026-03-04 entry says
  ~150 shards (~7B tokens) is the minimum for d24 GPT-2 capability. We
  initially tried 5 shards and got an 80-epoch overfit (val_bpb 1.00, CORE
  0.13). The 2026-05-04 / 2026-05-08 runs above both use `python -m
  nanochat_jax.dataset -n 170` for safety margin.
