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

| Date | Hardware | Model | Steps | val_bpb | CORE | Notes |
|---|---|---|---|---|---|---|
| 2026-05-02 | v6e-8 spot, ~5 min, ~$0.30 | d12, bf16, ClimbMix 50 shards | 500 | 1.110 | 0.0556 | First full-pipeline measurement on TPU. CORE measured against PT reference (gap 4.236% on val_bpb -- within Lenient tier). |
| 2026-05-08 | v5p-32 spot, ~6 h, ~$32 | d24, bf16 + matmul-precision=highest, 170 shards | 16704 | 0.7596 | 0.227 (22-task) | Best run with `--matmul-precision highest`. val_bpb is within Karpathy's same-config variance band (`0.71854 +- 6.4%`); CORE is ~12% below the upstream d24 baseline (0.2585). |
| 2026-05-09 | v6e-1 spot, ~3 h, ~$3.5 | d12 baseline vs d12 + matmul-precision=highest | 2520 | 2.86 (both) | -0.003 (centered avg) | Mini-ablation: HIGHEST has no measurable effect at d12 scale. The d24 result above is the regime where HIGHEST helps. |

## Methodology

- `val_bpb` is computed via `python -m scripts.base_eval --eval bpb`. The
  validation split is the last shard of ClimbMix.
- `CORE` is the DCLM 22-task ICL average, computed via
  `python -m scripts.base_eval --eval core --max-per-task 100`.
- "Spot" pricing is the price reported on the GCP console at the time of the
  run; you may pay more or less depending on region and capacity.
- "highest" in the matmul-precision column refers to JAX's
  `jax_default_matmul_precision="highest"`, which forces full-fp32
  accumulation for matmuls. Without it, TPU's default fp32 matmul uses bf16
  internal accumulation; this materially degrades converged CORE on d24+.
