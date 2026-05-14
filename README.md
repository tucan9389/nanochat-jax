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

`nanochat-jax` is the simplest experimental harness for training small LLMs on TPU. It is designed to run on a single TPU host (or a multi-host pod), the code is minimal and hackable, and it covers all major LLM stages: tokenization, pretraining, fine-tuning, evaluation, inference, and a CLI chat. The whole pipeline is driven by one complexity dial -- `--depth` -- and every other hyperparameter is derived automatically.

This port follows upstream nanochat closely: the model architecture, the optimizer (AdamW + Muon with Polar Express + NorMuon), the data pipeline (BOS-aligned best-fit packing on ClimbMix), and the eval (BPB + DCLM CORE) are all 1:1 mappings of the PyTorch reference. The JAX-specific additions are confined to where TPU matters: a Pallas Splash Attention path, a data-parallel sharding mesh, and a Muon ZeRO-2 momentum buffer split.

## Time-to-GPT-2 Leaderboard

The reference numbers from upstream nanochat (PyTorch / 8xH100) and this repository's measurements (JAX / TPU) live in [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md).

The primary metric is **CORE** -- the average over the DCLM 22-task ICL benchmark -- as in upstream nanochat ("time to GPT-2"). `val_bpb` is the secondary metric. Cross-framework / cross-device numerical agreement is bounded by the bf16-on-TPU mantissa floor (~9% per published cross-device studies).

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

**Cost**: a single d12 train cycle on v6e-8 spot takes ~0.83h ≈ **~$7** at list-price spot rate ($1.09/chip-h × 8 chips).

### Reproducing the d24 reference (multi-host TPU spot)

The d24 reference run (multi-host pod, full ClimbMix, `--matmul-precision highest`) lives in [`runs/d24.sh`](runs/d24.sh). It mirrors Karpathy's upstream d24 spec and is the closest comparison to the upstream leaderboard.

```bash
# Launch on every worker in parallel.
gcloud compute tpus tpu-vm ssh nanochat-jax-d24 --worker=all \
    --command="bash runs/d24.sh"
```

**Cost**: full d24 reference run on v5p-32 spot (16 chips, ~6-8h wall including eval) ≈ **~$165-210** at list-price spot rate. europe-west4 (~$20.5/hr, ~20% cheaper) is preferred over us-east5 (~$25.5/hr).

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

- Boot a `v6e-1` spot for cheap evaluation, a `v6e-8` spot for d12 training, and a `v5p-32` spot for d24 multi-host training. Always use `--spot`.
- Use the `--dry-run` flag on `scripts/base_train.py` for cheap end-to-end smoke tests (initializes mesh, model, optimizer, dataloader, then exits).
- `--matmul-precision highest` matters for d24+. Without it the default fp32 matmul on TPU silently uses bf16 internal accumulation, which materially degrades converged CORE.
- Always copy training checkpoints to a GCS bucket **before** deleting the TPU VM. A 5+ GB checkpoint takes hours to recompute.
- After every session, `gcloud compute tpus tpu-vm delete ...` and sweep all candidate zones to confirm there are no leftover VMs.

## Roadmap

### Done

- [x] Full pipeline: tokenizer → base pretrain → base eval → SFT → chat CLI.
- [x] d24 reference reproduced on a multi-host v5p-32 spot pod (CORE 0.227, 88% of GPT-2 grade; see [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md)).
- [x] Multi-host data-parallel training with Muon ZeRO-2 momentum sharding.
- [x] Pallas Splash Attention path, numerically verified against xla (`scripts/verify_splash.py`).
- [x] 22-task DCLM CORE eval matching upstream's metric definition.

### Not yet ported from upstream

- [ ] RL fine-tuning (`chat_rl.py`); the chat pipeline here ends at SFT.
- [ ] FP8 training (`fp8.py`); training is bf16 / fp32 only.
- [ ] Web chat UI (`chat_web.py` + `ui.html`); only the CLI is shipped.

### Closing the d24 gap to upstream

The d24 result above (CORE 0.227, 88% of upstream's 0.2585) already includes the biggest single win, `--matmul-precision highest`.

- [ ] Track down the remaining ~12% gap. Wider than upstream's ~6.4% within-config variance, so it isn't pure noise. Untried: multi-seed averaging, longer training horizons, the recipes from later leaderboard rows.
- [ ] Make Splash Attention compatible with `--matmul-precision=highest` (MosaicError on jax 0.10; the d24 reference falls back to `xla`).
- [ ] Fix multi-host resume from a non-zero step (today aborts; only fresh restarts work).
- [ ] Resolve the d24 CORE-eval OOM on a single v6e-1 host (workaround: eval on v5p).
- [ ] Add a `--no-final-eval` opt-out to `scripts/base_train.py`; the last training step forces a validation pass that adds ~1 h on d24.

## Other JAX nanochat ports

This is one of several JAX ports of nanochat in the wild. The ones that share the most code are:

- [`monatis/nanochat-jax`](https://github.com/monatis/nanochat-jax)
- [`vishal/nanochat-jax`](https://github.com/vishalbollu/nanochat-jax)
- [`ainaomotayo/nanochat-jax`](https://github.com/ainaomotayo/nanochat-jax)

Compared to those, this repo is more focused on multi-host TPU pods and the d24 reference run.

## Acknowledgements

- [Andrej Karpathy / nanochat](https://github.com/karpathy/nanochat) -- the upstream PyTorch implementation that this project ports. The model architecture, optimizer, data pipeline, and eval here are direct mappings.
- [Keller Jordan / modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) -- origin of the Muon optimizer.
- [`google/jax`](https://github.com/google/jax) and [`google/flax`](https://github.com/google/flax) -- the framework.
- [`AI-Hypercomputer/maxtext`](https://github.com/AI-Hypercomputer/maxtext) -- prior art for Splash Attention integration patterns.

## License

MIT. See [`LICENSE`](LICENSE).

## Disclaimer

This is a personal project. The views, code, and opinions expressed here are my own and do not represent those of my current or past employers.
