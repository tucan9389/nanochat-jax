# nanochat commit-by-commit comparison & nanochat-jax parity

Columns are upstream nanochat leaderboard runs (**R1–R6**), current master (**HEAD**), and our two JAX/TPU ports (**jax₁**, **jax₂**).
**jax₁ mirrors R6/HEAD; jax₂ reproduces R4 (`324e69c`).**

| col | commit | what |
|---|---|---|
| GPT2 | – | OpenAI GPT-2 1.6B (=gpt2-xl) reference |
| init | `3a5e0bc` | d20 "$100 speedrun", pre-leaderboard |
| R1 | `348fbb3` | d24, ratio 12, first GPT-2 beat |
| R2 | `a67eba3` | d26, ratio 8.5 + FP8 |
| R3 | `2c062aa` | d26, ratio 8.25, batch 1M |
| R4 | `324e69c` | ClimbMix, d24, ratio 9.5 |
| R5 | `6ed7d1d` | autoresearch round 1 |
| R6 | `a825e63` | autoresearch round 2 (smear/backout), SOTA |
| HEAD | `dc54a1a` | current master (not a run) |
| jax₁ | `956e043` | nanochat-jax d24 v6e-8 public run |
| jax₂ | `3327d17` | nanochat-jax Run-4 (`324e69c`) reproduction |

Legend: `–` absent · `X` present-but-off · `O` on · `〃` same as left · `?` unknown · `( )` assumed (non-run col).

## T1. Results

| axis | GPT2 | init | R1 | R2 | R3 | R4 | R5 | R6 | HEAD | jax₁ | jax₂ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| depth | – | 20 | 24 | 26 | 26 | 24 | 24 | 24 | (24) | 24 | 24 |
| HW | TPUv3×32 | 8×H100 | 8×H100 | 8×H100 | 8×H100 | 8×H100 | 8×H100 | 8×H100 | (8×H100) | v6e-8 | v6e-8 |
| time (h) | 168 | – | 3.04 | 2.91 | 2.76 | 2.02 | 1.80 | 1.65 | – | 7.33 | 5.28 |
| steps | – | 21,400 | 16,704 | 14,889 | 7,226 | 6,612 | 6,055 | 5,568 | (5,568) | 16,704 | 6,612 |
| CORE | 0.2565 | 0.2219 | 0.25851 | 0.2578 | 0.26024 | 0.25714 | 0.2690 | 0.26263 | – | 0.27409 | 0.25822 |
| val_bpb | – | ~0.81 F | 0.74833 F | 0.74504 F | 0.74645 F | 0.71854 C | 0.71808 C | 0.71800 C | – | 0.71879 C | 0.71655 C |
| runs measured | – | 1 | 1 | 1 | 1 | 7 | 5 | 5 | – | 1 | 1 |
| MFU | – | – | 50% | 58% | 60% | 60% | 60% | 59% | – | ~22% | ~24.5% |

`F` = FineWeb-EDU val, `C` = ClimbMix val — F↔C not comparable. `time (h)` = training iterations only (excludes eval/logging).

## T2. Run config

| axis | init | R1 | R2 | R3 | R4 | R5 | R6 | HEAD | jax₁ | jax₂ |
|---|---|---|---|---|---|---|---|---|---|---|
| ratio | 20 | 12 | 8.5 | 8.25 | 9.5 | 8.7 | 8 | (8) | 12 | 9.5 |
| batch (tok/step) | 524,288 | 524,288 | 524,288 | 1,048,576 | 1,048,576 | 1,048,576 | 1,048,576 | 1,048,576 | 524,288 | 1,048,576 |
| auto-batch | – | – | – | O | O | O | O | O | – | – |
| device_batch | 32 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 2 | 2 |
| grad_accum | 1 | 2 | 2 | 4 | 4 | 4 | 4 | 4 | 16 | 32 |
| seq_len | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 |
| train tokens | 11.22B | 8.76B | 7.81B | 7.58B | 6.93B | 6.35B | 5.84B | (5.84B) | 8.76B | 6.93B |

## T3. Architecture

| axis | init | R1 | R2 | R3 | R4 | R5 | R6 | HEAD | jax₁ | jax₂ |
|---|---|---|---|---|---|---|---|---|---|---|
| n_embd / n_head | 1280/10 | 1536/12 | 1664/13 | 1664/13 | 1536/12 | 1536/12 | 1536/12 | 1536/12 | 1536/12 | 1536/12 |
| head_dim · KV | 128·1:1 | 128·1:1 | 128·1:1 | 128·1:1 | 128·1:1 | 128·1:1 | 128·1:1 | 128·1:1 | 128·1:1 | 128·1:1 |
| vocab | 65,536 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 | 32,768 |
| RoPE θ | 1e4 | 1e4 | 1e4 | 1e4 | 1e4 | 1e5 | 1e5 | 1e5 | 1e5 | 1e4 |
| QK scale | – | – | – | – | – | 1.15 | 1.2 | 1.2 | 1.2 | – |
| logit softcap | 15 | 15 | 15 | 15 | 20 | 15 | 15 | 15 | 15 | 20 |
| window pattern | –(full) | SSSL | SSSL | SSSL | SSSL | SSSL | SSSL | SSSL | SSSL | SSSL |
| short window | – | 1024 | 1024 | 1024 | 1024 | 768 | 512 | 512 | 512 | 1024 |
| value embeds | – | O | O | O | O | O | O | O | O | O |
| VE gate | – | 32ch·2σ | 32ch·2σ | 32ch·2σ | 32ch·2σ | 12ch·3σ | 12ch·3σ | 12ch·3σ | 12ch·3σ | 32ch·2σ |
| smear | – | – | – | – | – | – | O | O | O | X |
| backout | – | – | – | – | – | – | O | O(0.2) | O(0.2) | X |
| resid λ init | – | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.15→1.05 | 〃 | 〃 | 1.0 |
| x0 λ init | – | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.20→0.05 | 〃 | 〃 | 0.1 |
| wte init std | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.8 | 0.8 | 0.8 | 0.8 | 1.0 |
| c_fc init scale | – | 1× | 1× | 1× | 1× | 0.5× | 0.4× | 0.4× | 0.4× | 1× |
| lm_head init | 0 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 |
| attention kernel | SDPA | FA3 | FA3 | FA3 | FA3 | FA3 | FA3 | FA3 | Splash | Splash |
| activation | ReLU²·4× | ReLU²·4× | ReLU²·4× | ReLU²·4× | ReLU²·4× | ReLU²·4× | ReLU²·4× | ReLU²·4× | ReLU²·4× | ReLU²·4× |
| emb tying | untied | untied | untied | untied | untied | untied | untied | untied | untied | untied |

## T4. Optimizer & schedule

| axis | init | R1 | R2 | R3 | R4 | R5 | R6 | HEAD | jax₁ | jax₂ |
|---|---|---|---|---|---|---|---|---|---|---|
| structure | Muon+AdamW split | unified | unified | unified | unified | unified | unified | unified | mirror | mirror |
| matrix_lr | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| embedding_lr | 0.2 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 |
| unembedding_lr | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 | 0.008 | 0.008 | 0.008 | 0.008 | 0.004 |
| scalar_lr | – | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| VE lr scale | – | 1× | 1× | 1× | 1× | 0.5× | 0.5× | 0.5× | 0.5× | 1× |
| AdamW betas | (0.8, 0.95) | (0.8, 0.95) | 〃 | 〃 | 〃 | per-group | 〃 | 〃 | 〃 | (0.8, 0.95) |
| WD value | 0 | 0.2 | 0.2 | 0.2 | 0.2 | 0.28 | 0.28 | 0.28 | 0.28 | 0.2 |
| WD scale | – | (12/d)² | (12/d)² | (12/d)² | Tepoch | Tepoch | Tepoch | Tepoch | Tepoch | Tepoch |
| WD schedule | – | linear→0 | linear→0 | linear→0 | linear→0 | cosine→0 | cosine→0 | cosine→0 | cosine→0 | linear→0 |
| cautious WD | – | O | O | O | O | O | O | O | O | O |
| Muon orthog. | NS5 | PE5 | PE5 | PE5 | PE5 | PE5(1.01) | PE5 | PE5 | PE5 | PE5(1.02) |
| NorMuon β₂ | – | 0.95 | 0.95 | 0.95 | 0.95 | 0.9 | 0.9 | 0.9 | 0.9 | 0.95 |
| Muon momentum | .85→.95@300 | 〃 | 〃 | 〃 | 〃 | →.97@400 | +wd→.90 | 〃 | 〃 | .85→.97@400+wd→.90 |
| LR warmup | 0 | 0 | 0 | 0 | 0 | 40 | 40 | 40 | 40 | 0 |
| LR warmdown | 0.2 | 0.5 | 0.5 | 0.5 | 0.5 | 0.65 | 0.65 | 0.65 | 0.65 | 0.5 |
| final_lr_frac | 0 | 0 | 0 | 0 | 0 | 0.05 | 0.05 | 0.05 | 0.05 | 0 |
| grad_clip | 1.0 | – | – | – | – | – | – | – | – | – |
| dist method | ZeRO-2 | ZeRO-2 | 〃 | 〃 | 〃 | 〃 | 〃 | 〃 | jit+GSPMD | jit+GSPMD |

## T5. Data & tokenizer

| axis | init | R1 | R2 | R3 | R4 | R5 | R6 | HEAD | jax₁ | jax₂ |
|---|---|---|---|---|---|---|---|---|---|---|
| dataset | FW-EDU | FW-EDU | FW-EDU | FW-EDU | ClimbMix | ClimbMix | ClimbMix | ClimbMix | ClimbMix | ClimbMix |
| total shards | 1823 | 1823 | 1823 | 1823 | 6543 | 6543 | 6543 | 6543 | 6543 | 6543 |
| download -n | 240 | ? | ? | ? | 170 | 170 | 170 | 170 | 170 | 170 |
| val split | last shard | last shard | last shard | last shard | last shard | 〃 | 〃 | 〃 | 〃 | 〃 |
| runtime shuffle | X | X | X | X | X | X | X | X | X | X |
| tok regex digits | {1,2} | {1,2} | {1,2} | {1,2} | {1,2} | {1,2} | {1,2} | {1,2} | {1,2} | {1,2} |
| special tokens | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 |
| dataloader | deque-crop | BOS-bestfit | BOS-bestfit | 〃 | 〃 | 〃 | 〃 | 〃 | mirror | mirror |

`download -n` = train shards fetched (+1 val). ClimbMix runs use 170 (R4 reproduce minimum: 150); R1–R3 (FineWeb) not recorded. Tokens above (T2) consume ~150.

## T6. Precision & kernel

| axis | init | R1 | R2 | R3 | R4 | R5 | R6 | HEAD | jax₁ | jax₂ |
|---|---|---|---|---|---|---|---|---|---|---|
| base precision | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 |
| FP8 | – | – | torchao | torchao | own | own | own | own | – | – |
| eval precision | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 | bf16 + lm_head highest | bf16 + lm_head highest |
| compile | – | torch | torch | torch | torch | torch | torch | torch | jax.jit | jax.jit |
| seed | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 | 42 |

## T7. Eval protocol

| axis | init | R1 | R2 | R3 | R4 | R5 | R6 | HEAD | jax₁ | jax₂ |
|---|---|---|---|---|---|---|---|---|---|---|
| CORE tasks | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 22 |
| final full CORE | O | O | O | O | O | O | O | O | O | O |

<sub>Values verified by direct grep of each commit and nanochat `dev/LEADERBOARD.md`. Upstream commits: R1 `348fbb3` · R2 `a67eba3` · R3 `2c062aa` · R4 `324e69c` · R5 `6ed7d1d` · R6 `a825e63` · HEAD `dc54a1a`. Ours: jax₁ `956e043` · jax₂ `3327d17`.</sub>
