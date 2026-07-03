# nanochat-jax

```
███╗   ██╗ █████╗ ███╗   ██╗ ██████╗  ██████╗██╗  ██╗ █████╗ ████████╗        ██╗ █████╗ ██╗  ██╗
████╗  ██║██╔══██╗████╗  ██║██╔═══██╗██╔════╝██║  ██║██╔══██╗╚══██╔══╝        ██║██╔══██╗╚██╗██╔╝
██╔██╗ ██║███████║██╔██╗ ██║██║   ██║██║     ███████║███████║   ██║█████╗     ██║███████║ ╚███╔╝ 
██║╚██╗██║██╔══██║██║╚██╗██║██║   ██║██║     ██╔══██║██╔══██║   ██║╚════╝██   ██║██╔══██║ ██╔██╗ 
██║ ╚████║██║  ██║██║ ╚████║╚██████╔╝╚██████╗██║  ██║██║  ██║   ██║      ╚█████╔╝██║  ██║██╔╝ ██╗
╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝       ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝
```

> JAX / Flax NNX port of [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat), optimized for Google Cloud TPU.

Like upstream, the number that matters here is **time-to-GPT-2**: the wall-clock time to train past the GPT-2 CORE score of `0.256525`. On a single v6e-8 spot TPU node, training with `--recipe 324e69c` does it in **5.28 hours — about $23** at the June 2026 spot price ([launch command](dev/LEADERBOARD.md)).

## Time-to-GPT-2 on TPU

| # | time (h) | val_bpb | CORE | Description | Date | Commit |
|---|---:|---:|---:|---|---|---|
| 0 | 168.0 | -- | 0.2565 | OpenAI GPT-2 1.6B reference | 2019 | -- |
| 1 | 7.33 | 0.71879 | 0.27409 | d24 baseline (`--recipe dc54a1a`) | Jun 3 2026 | 828485b |
| 2 | **5.28** | 0.71655 | 0.25822 | d24, upstream Run 4 recipe (`--recipe 324e69c`), 1M-token batches | Jun 23 2026 | 4d9d88b |

Times are train-loop wall-clock on v6e-8, excluding eval and logging (the upstream convention). Configs, launch commands, eval protocol, reproduction verification, and cost basis: [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md).

**Matched to upstream:** the d24 model geometry (24 layers / 1536 wide / 12 heads), the 32,768-vocab tokenizer recipe, the 2048-token context, the upstream batch/horizon recipes, the NVIDIA ClimbMix pretraining data, and the full 22-task CORE eval protocol.

**Changed for the TPU stack:** JAX/Flax NNX instead of PyTorch, Pallas Splash Attention kernels, bf16 embedding storage, TPU mesh sharding, and no FP8 path yet.

**Intentionally preserved divergences:** the port differs from upstream in four known, small places (a decode-window off-by-one, CORE-eval prompt truncation, the smear_gate init, and the SpellingBee SFT templates). They predate the published runs, so "fixing" them would break their reproduction — each is documented in [`dev/REPRODUCTION-GUARDS.md`](dev/REPRODUCTION-GUARDS.md) and marked with a NOTE comment at its code site.

## Getting started

`nanochat-jax` uses a standard Python venv with Python 3.11+. Note that TPU VMs ship Python 3.10, so get a newer interpreter first — the verified path is [`uv`](https://docs.astral.sh/uv/) (`uv venv --python 3.12 ~/venv`), or install `python3.11` from your distro. For TPU:

```bash
python3.11 -m venv ~/venv   # or: uv venv --python 3.12 ~/venv
source ~/venv/bin/activate
pip install -U pip
pip install "torch~=2.11.0" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[tpu,dev]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

For Mac CPU or plain CPU/GPU without TPU:

```bash
pip install -e ".[dev]"
```

## Reference run

[`runs/speedrun.sh`](runs/speedrun.sh) is the public run script. It trains the row-1 baseline (`--recipe dc54a1a`): `seq_len=2048`, total batch `524288`, Splash Attention, onehot value-embedding gradients, and model-tag checkpoint/eval continuity. It then trains an SFT checkpoint, runs ChatEval on the SFT checkpoint, and writes a report via `python -m nanochat_jax.report generate`.

SFT ChatEval covers the validated categorical tasks: ARC-Easy, ARC-Challenge, and MMLU. The generative tasks (GSM8K, HumanEval, SpellingBee) are not part of the pipeline — the current JAX generative eval path is too slow to complete them at full scale; they can be scored manually with `scripts/chat_eval.py --task-name`. Measured post-SFT scores are recorded in [`dev/LEADERBOARD.md`](dev/LEADERBOARD.md).

On a v6e-8 TPU host:

```bash
bash runs/speedrun.sh
```

For a wiring-only CPU smoke run that creates a fresh tiny checkpoint:

```bash
bash runs/runcpu.sh
```

The smoke path is not quality evidence. It defaults to `~/.cache/nanochat-jax-smoke`, uses a tiny d2 model, and forces CPU unless `JAX_PLATFORMS` is already set.

## Chat with the model

```bash
python -m scripts.chat_cli -i sft --model-tag d24_speedrun_sft                         # interactive
python -m scripts.chat_cli -i sft --model-tag d24_speedrun_sft --prompt "Hello" -t 0   # single-prompt, greedy
```

Use `-i base --model-tag d24_speedrun` to inspect the pretrained base checkpoint directly. Base models have limited instruction following because they have not seen chat-formatted SFT data.

## File structure

```
.
|-- LICENSE
|-- README.md
|-- pyproject.toml
|-- nanochat_jax/
|   |-- gpt.py                    # GPT model: XLA + Splash Attention paths
|   |-- attention.py              # attention backend helpers (Splash/Pallas wiring)
|   |-- layers.py                 # NanochatLinear
|   |-- ops.py                    # rms_norm, rotary, ReLU^2, softcap
|   |-- optim.py                  # AdamW + Muon
|   |-- grad_utils.py             # loss + gradient computation
|   |-- train_core.py             # train_step, schedules, sharded/fused factories
|   |-- base_train_config.py      # muP sizing math + weight-decay reference
|   |-- sharding.py               # Mesh + Muon ZeRO-2 spec
|   |-- perf.py                   # hardware peak FLOPS table + MFU helpers
|   |-- engine.py                 # KV cache + sampling
|   |-- checkpoint_manager.py     # save / load + optimizer state
|   |-- common.py                 # cache dir, distributed info, file lock
|   |-- report.py                 # markdown report reset/log/generate
|   |-- core_eval.py              # DCLM CORE tasks
|   |-- loss_eval.py              # bits-per-byte
|   |-- dataloader.py             # BOS-aligned best-fit packing
|   |-- dataset.py                # parquet shard download / iteration
|   |-- tokenizer.py              # RustBPE + tiktoken + HF wrapper
|   |-- execution.py              # sandboxed Python execution (tool use, HumanEval)
|   |-- weight_converter.py       # PyTorch state_dict <-> JAX pytree converter
|   `-- tasks/                    # ARC, MMLU, GSM8K, HumanEval, SmolTalk, ...
|-- scripts/
|   |-- base_train.py             # base training
|   |-- base_eval.py              # BPB, samples, CORE
|   |-- chat_sft.py               # SFT
|   |-- chat_eval.py              # ChatCORE
|   |-- chat_cli.py               # interactive chat
|   |-- tok_train.py              # tokenizer train
|   `-- tok_eval.py               # tokenizer eval
|-- runs/
|   |-- speedrun.sh               # public d24 e2e reference run (v6e-8 TPU)
|   |-- runcpu.sh                 # same pipeline on a tiny d2 model (CPU wiring check)
|   `-- README.md
`-- dev/
    |-- LEADERBOARD.md            # published results + reproduction reference
    `-- REPRODUCTION-GUARDS.md    # the 4 intentional upstream divergences
```

## TPU notes

- Use a `v6e-8` TPU for the d24 reference run. Capacity and pricing can vary.
- Use `--dry-run` on `scripts/base_train.py` for cheap preflight checks.
- Keep the GCS bucket in the same region as the TPU.
- Copy expensive checkpoints to GCS before deleting the TPU VM.
- Delete the TPU VM when you are done. If you tried creating TPUs in more than one zone, list each of those zones afterwards to catch leftover VMs — an orphaned TPU keeps billing.

## Status

Done:

- [x] d24 v6e-8 TPU/JAX train result with BPB and CORE artifacts.
- [x] Public speedrun wiring from tokenizer through base/SFT/ChatEval/report.
- [x] Full-mixture SFT checkpoint path and post-SFT categorical ChatEval.
- [x] Pallas Splash Attention path.
- [x] Muon optimizer support.

Not in the default path yet:

- [ ] RL fine-tuning.
- [ ] FP8 training.

## Acknowledgements

- [Andrej Karpathy / nanochat](https://github.com/karpathy/nanochat) - the upstream PyTorch implementation.
- NVIDIA ClimbMix ([`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)) - the pretraining dataset.
- [Keller Jordan / modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) - origin of the Muon optimizer.
- [`google/jax`](https://github.com/google/jax) and [`google/flax`](https://github.com/google/flax) - the framework.
- [`AI-Hypercomputer/maxtext`](https://github.com/AI-Hypercomputer/maxtext) - Splash Attention integration patterns.

## License

MIT. See [`LICENSE`](LICENSE).

## Disclaimer

This is a personal project. The views, code, and opinions expressed here are my own and do not represent those of my current or past employers.
