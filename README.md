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

`nanochat-jax` is a TPU-optimized JAX/NNX port of `karpathy/nanochat`: same d24 baseline model shape, same 2048-token context, same 524K-token batch, same 16,704-step train horizon, and the same CORE eval protocol.

On v6e-8 spot, `nanochat-jax` reached `CORE=0.274` within a $100 TPU budget. The reference path is [`runs/speedrun.sh`](runs/speedrun.sh): d24 base training followed by CORE eval.

## Getting started

`nanochat-jax` uses a standard Python venv. For TPU:

```bash
python3.11 -m venv ~/venv
source ~/venv/bin/activate
pip install -U pip
pip install -e ".[tpu,dev]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

For Mac CPU or plain CPU/GPU without TPU:

```bash
pip install -e ".[dev]"
```

## Reference run

[`runs/speedrun.sh`](runs/speedrun.sh) is the public run script. It uses the d24 TPU train shape used by the current result: `seq_len=2048`, total batch `524288`, Splash Attention, onehot value-embedding gradients, and model-tag checkpoint/eval continuity.

On a v6e-8 TPU host:

```bash
bash runs/speedrun.sh
```

CORE eval is the default. For a quick smoke run:

```bash
CORE_MAX_PER_TASK=50 NUM_ITERATIONS=250 bash runs/speedrun.sh
```

## Cost note

The v6e-8 spot result fits within the $100 TPU budget. Pricing changes; see [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for the result row, spot price basis, and hardware comparison.

## Chat with the model

```bash
python -m scripts.chat_cli --model-tag d24_speedrun                         # interactive
python -m scripts.chat_cli --model-tag d24_speedrun --prompt "Hello" -t 0   # single-prompt, greedy
```

The speedrun checkpoint is a base model. Instruction following is limited until an SFT checkpoint is trained.

## File structure

```
.
|-- LICENSE
|-- README.md
|-- pyproject.toml
|-- nanochat_jax/
|   |-- gpt.py                    # GPT model: XLA + Splash Attention paths
|   |-- optim.py                  # AdamW + Muon
|   |-- base_train.py             # train_step, schedules, sharded factory
|   |-- engine.py                 # KV cache + sampling
|   |-- checkpoint_manager.py     # save / load + optimizer state
|   |-- common.py                 # cache dir, distributed info, file lock
|   |-- core_eval.py              # DCLM CORE tasks
|   |-- loss_eval.py              # bits-per-byte
|   |-- dataloader.py             # BOS-aligned best-fit packing
|   |-- dataset.py                # parquet shard download / iteration
|   |-- tokenizer.py              # RustBPE + tiktoken + HF wrapper
|   |-- layers.py                 # NanochatLinear
|   |-- sharding.py               # Mesh + Muon ZeRO-2 spec
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
|   |-- speedrun.sh               # public d24 reference run
|   `-- README.md
|-- tests/
`-- dev/
    `-- LEADERBOARD.md
```

## TPU notes

- Use a `v6e-8` TPU for the d24 reference run. Capacity and pricing can vary.
- Use `--dry-run` on `scripts/base_train.py` for cheap preflight checks.
- Keep the GCS bucket in the same region as the TPU.
- Copy expensive checkpoints to GCS before deleting the TPU VM.
- Delete TPU VMs and sweep candidate zones after each session.

## Status

Done:

- [x] d24 v6e-8 TPU/JAX train result with BPB and CORE artifacts.
- [x] Public speedrun path from base train to BPB/CORE eval.
- [x] Pallas Splash Attention path.
- [x] Muon optimizer support.

Not in the default path yet:

- [ ] SFT and chat eval publish-path validation.
- [ ] Upstream 8xH100 Time-to-GPT-2 leaderboard claim.
- [ ] v5p rows; retired until a separate parity audit.
- [ ] RL fine-tuning, FP8 training, and web chat UI.

## Acknowledgements

- [Andrej Karpathy / nanochat](https://github.com/karpathy/nanochat) - the upstream PyTorch implementation.
- [Keller Jordan / modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) - origin of the Muon optimizer.
- [`google/jax`](https://github.com/google/jax) and [`google/flax`](https://github.com/google/flax) - the framework.
- [`AI-Hypercomputer/maxtext`](https://github.com/AI-Hypercomputer/maxtext) - Splash Attention integration patterns.

## License

MIT. See [`LICENSE`](LICENSE).

## Disclaimer

This is a personal project. The views, code, and opinions expressed here are my own and do not represent those of my current or past employers.
