# nanochat-jax

```
███╗   ██╗ █████╗ ███╗   ██╗ ██████╗  ██████╗██╗  ██╗ █████╗ ████████╗        ██╗ █████╗ ██╗  ██╗
████╗  ██║██╔══██╗████╗  ██║██╔═══██╗██╔════╝██║  ██║██╔══██╗╚══██╔══╝        ██║██╔══██╗╚██╗██╔╝
██╔██╗ ██║███████║██╔██╗ ██║██║   ██║██║     ███████║███████║   ██║█████╗     ██║███████║ ╚███╔╝ 
██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║██║     ██╔══██║██╔══██║   ██║╚════╝██   ██║██╔══██║ ██╔██╗ 
██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝╚██████╗██║  ██║██║  ██║   ██║      ╚█████╔╝██║  ██║██╔╝ ██╗
╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝       ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
```

> JAX / Flax NNX port of [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat), tuned for Google Cloud TPU.

`nanochat-jax` is the simplest experimental harness for training a GPT-2-class small LLM from scratch on TPU. A single v6e-8 spot host runs one d12 cycle in \~0.5-1h for **\~$5-10**; v5p-32 spot reproduces Karpathy's d24 reference in \~8h for **\~$165-205**, landing in the **CORE 0.227-0.290** band (single-seed; upstream d24 baseline 0.2585; see [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md) for the full numbers). The pipeline mirrors upstream nanochat: tokenization → pretraining → SFT → evaluation → CLI chat.

## Why this repo?

Porting `karpathy/nanochat` to JAX is mechanical, but recovering CORE on TPU required clearing several TPU-specific hurdles that don't exist on PyTorch / GPU: `--matmul-precision=highest` (+27.6% CORE), Muon ZeRO-2 momentum sharding, ClimbMix ≥150 shards, a multi-host resume bug, and a handful of others.

The full trail of trial-and-error -- together with the remaining known issues (single seed, cross-host bf16 noise, Splash + `--matmul-precision=highest` incompatibility) -- is documented in [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md) under *Implementation notes*.

The JAX-specific additions are confined to where TPU matters: a Pallas Splash Attention path, a data-parallel (DP) sharding mesh, and a Muon ZeRO-2 momentum buffer split.

Several JAX ports of nanochat exist in the wild, but as of 2026-05-15 no other one is publicly known to have reached Karpathy's d24 CORE baseline on TPU.

→ Start in 5 minutes: `bash runs/speedrun.sh` (see [Getting started](#getting-started) below).

## Getting started

`nanochat-jax` uses a standard Python venv. JAX is installed with the TPU extra and the libtpu releases index:

```bash
python3.11 -m venv ~/venv
source ~/venv/bin/activate
pip install -U pip
pip install -e ".[tpu,dev]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

For Mac CPU or plain CPU/GPU without a TPU, drop the `[tpu]` extra:

```bash
pip install -e ".[dev]"
```

### Quick demo (TPU spot)

The full pipeline -- tokenizer training, depth-12 base pretraining, base evaluation, optional SFT -- lives in [`runs/speedrun.sh`](runs/speedrun.sh). On a v6e-8 spot host:

```bash
bash runs/speedrun.sh
```

**Cost**: a full speedrun cycle (tokenizer + d12 train + base eval + tiny SFT) on v6e-8 spot takes \~0.5-1h ≈ **\~$5-10** at list-price spot rate ($1.09/chip-h × 8 chips).

### Reproducing the d24 reference (TPU spot)

The d24 reference run (v5p-32 spot, full ClimbMix, `--matmul-precision highest`) lives in [`runs/d24.sh`](runs/d24.sh). It mirrors Karpathy's upstream d24 spec and is the closest comparison to the upstream leaderboard.

```bash
# Launch on every worker in parallel.
gcloud compute tpus tpu-vm ssh nanochat-jax-d24 --worker=all \
    --command="bash runs/d24.sh"
```

**Cost**: full d24 reference run on v5p-32 spot (16 chips, \~8h wall including eval) ≈ **\~$165-205** at list-price spot rate. europe-west4 (\~$20.5/hr) is \~20% cheaper than us-east5 (\~$25.5/hr).

### Chat with the model

```bash
python -m scripts.chat_cli                         # interactive
python -m scripts.chat_cli --prompt "Hello" -t 0   # single-prompt, greedy
```

## File structure

```
.
|-- LICENSE
|-- README.md
|-- pyproject.toml
|-- nanochat_jax/
|   |-- __init__.py
|   |-- gpt.py                    # GPT model: xla + Splash Attention paths
|   |-- optim.py                  # AdamW + Muon (Polar Express + NorMuon)
|   |-- base_train.py             # train_step + sharded factory + schedules
|   |-- engine.py                 # KV cache + sampling + calculator tool
|   |-- execution.py              # sandboxed Python execution for tools
|   |-- checkpoint_manager.py     # save / load + optimizer state
|   |-- common.py                 # cache dir, distributed info, file lock
|   |-- core_eval.py              # DCLM CORE (22 ICL tasks)
|   |-- loss_eval.py              # bits-per-byte
|   |-- dataloader.py             # BOS-aligned best-fit packing
|   |-- dataset.py                # parquet shard download / iteration
|   |-- tokenizer.py              # RustBPE + tiktoken + HF wrapper
|   |-- ops.py                    # rms_norm, RoPE, ReLU squared, softcap
|   |-- layers.py                 # NanochatLinear (fp32 master + dot_general hook)
|   |-- sharding.py               # Mesh + Muon ZeRO-2 spec
|   |-- grad_utils.py             # compute_loss_and_grad + flat-dict helpers
|   |-- weight_converter.py       # PyTorch state_dict <-> JAX pytree
|   |-- golden_loader.py          # PT reference snapshot loader
|   `-- tasks/                    # ARC, MMLU, GSM8K, HumanEval, SmolTalk, ...
|-- scripts/
|   |-- base_train.py             # train: data-parallel, dry-run flag
|   |-- base_eval.py              # eval: BPB + sample + CORE
|   |-- chat_sft.py               # SFT: full / reduced / min / identity-only scopes
|   |-- chat_eval.py              # ChatCORE on six tasks
|   |-- chat_cli.py               # interactive chat
|   |-- tok_train.py              # train BPE tokenizer
|   |-- tok_eval.py               # tokenizer compression vs GPT-2 / GPT-4
|   |-- generate_golden.py        # produce PyTorch reference snapshots
|   |-- compare_weights.py        # diff two .npz weight files
|   `-- verify_splash.py          # Splash Attention TPU verification
|-- runs/
|   |-- speedrun.sh               # quick demo (v6e-8 spot)
|   |-- d24.sh                    # d24 reference (v5p-32 spot, see runs/README.md)
|   `-- README.md
|-- tests/                        # pytest (Mac CPU + selective TPU)
`-- dev/
    `-- LEADERBOARD.md            # this repo's measurements vs upstream
```

## Notes on running on TPU

- Boot a `v6e-1` spot for cheap d12 evaluation, a `v6e-8` spot for d12 training, a `v5p-32` spot for d24 multi-host training, or a `v5p-8` spot for the cheapest single-host d24 path (d24 evaluation OOMs on v6e-1; see Roadmap). Always use `--spot`.
- Use the `--dry-run` flag on `scripts/base_train.py` for cheap end-to-end smoke tests (initializes mesh, model, optimizer, dataloader, then exits).
- `--matmul-precision highest` matters for d24+. Without it the default fp32 matmul on TPU silently uses bf16 internal accumulation, which materially degrades converged CORE.
- Place your GCS bucket in the same region as the TPU (e.g. `asia-northeast1` for `asia-northeast1-b`); cross-region 5+ GB copies time out and waste hours.
- Always copy training checkpoints to a GCS bucket **before** deleting the TPU VM. A 5+ GB checkpoint takes hours to recompute.
- Spot preemption is common on `v5p` during \~8h trains; sweep nearby zones (`us-east5-a/b/c`, `europe-west4-a/b`) when capacity disappears.
- After every session, `gcloud compute tpus tpu-vm delete ...` and sweep all candidate zones to confirm there are no leftover VMs.

## Roadmap

### Done

- [x] Full pipeline: tokenizer → base pretrain → base eval → SFT → chat CLI.
- [x] d24 reference run reproduced on TPU; CORE reaches Karpathy's d24 baseline (see [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md)).
- [x] Multi-host data-parallel training with Muon ZeRO-2 momentum sharding.
- [x] Pallas Splash Attention path, numerically verified against xla (`scripts/verify_splash.py`).
- [x] 22-task DCLM CORE eval matching upstream's metric definition.

### Not yet ported from upstream

- [ ] RL fine-tuning (`chat_rl.py`); the chat pipeline here ends at SFT.
- [ ] FP8 training (`fp8.py`); training is bf16 / fp32 only.
- [ ] Web chat UI (`chat_web.py` + `ui.html`); only the CLI is shipped.

### d24 follow-ups

The single largest win on TPU is `--matmul-precision highest` (see [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md) Implementation notes).

- [ ] Multi-seed validation of the v5p-8 single-host run before promoting its score to the LEADERBOARD (single-seed CORE 0.290; see [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md)).
- [ ] Make Splash Attention compatible with `--matmul-precision=highest` (MosaicError on jax 0.10; the d24 reference falls back to `xla`).
- [ ] Fix multi-host resume from a non-zero step (today aborts; only fresh restarts work).
- [ ] Resolve the d24 CORE-eval OOM on a single v6e-1 host (workaround: eval on v5p).

## Acknowledgements

- [Andrej Karpathy / nanochat](https://github.com/karpathy/nanochat) -- the upstream PyTorch implementation that this project ports. The model architecture, optimizer, data pipeline, and eval here are direct mappings.
- [Keller Jordan / modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) -- origin of the Muon optimizer.
- [`google/jax`](https://github.com/google/jax) and [`google/flax`](https://github.com/google/flax) -- the framework.
- [`AI-Hypercomputer/maxtext`](https://github.com/AI-Hypercomputer/maxtext) -- prior art for Splash Attention integration patterns.

## License

MIT. See [`LICENSE`](LICENSE).

## Disclaimer

This is a personal project. The views, code, and opinions expressed here are my own and do not represent those of my current or past employers.
