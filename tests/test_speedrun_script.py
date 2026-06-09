from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEEDRUN = ROOT / "runs" / "speedrun.sh"


def _assert_in_order(text: str, needles: list[str]) -> None:
    cursor = -1
    for needle in needles:
        next_pos = text.find(needle, cursor + 1)
        assert next_pos != -1, f"missing stage marker: {needle}"
        assert next_pos > cursor, f"stage marker is out of order: {needle}"
        cursor = next_pos


def test_speedrun_shell_syntax_is_valid():
    subprocess.run(["bash", "-n", str(SPEEDRUN)], check=True)


def test_speedrun_production_defaults_are_public_d24_contract():
    text = SPEEDRUN.read_text()

    assert 'MODEL_TAG="${MODEL_TAG:-d24_speedrun}"' in text
    assert 'BASE_DEPTH="${BASE_DEPTH:-24}"' in text
    assert 'BASE_SEQ_LEN="${BASE_SEQ_LEN:-2048}"' in text
    assert 'BASE_TOTAL_BATCH_SIZE="${BASE_TOTAL_BATCH_SIZE:-524288}"' in text
    assert 'BASE_DEVICE_BATCH_SIZE="${BASE_DEVICE_BATCH_SIZE:-2}"' in text
    assert 'BASE_GRAD_ACCUM_STEPS="${BASE_GRAD_ACCUM_STEPS:-16}"' in text
    assert 'NUM_ITERATIONS="${NUM_ITERATIONS:-16704}"' in text
    assert 'BASE_EVAL_MODES="${BASE_EVAL_MODES:-bpb,core}"' in text
    assert 'CORE_MAX_PER_TASK="${CORE_MAX_PER_TASK:--1}"' in text
    assert 'SFT_SCOPE="${SFT_SCOPE:-full}"' in text
    assert 'CHAT_EVAL_TASK_NAME="${CHAT_EVAL_TASK_NAME:-ARC-Easy|ARC-Challenge|MMLU}"' in text
    assert "--attn-impl=splash" in text
    assert "--splash-block-q=512" in text
    assert "--splash-block-kv=512" in text
    assert "--splash-block-kv-compute=256" in text


def test_speedrun_smoke_defaults_are_cpu_safe_and_fresh_checkpointed():
    text = SPEEDRUN.read_text()

    assert 'export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"' in text
    assert 'MODEL_TAG="${MODEL_TAG:-speedrun_smoke_d2}"' in text
    assert 'BASE_DEPTH="${BASE_DEPTH:-2}"' in text
    assert 'BASE_SEQ_LEN="${BASE_SEQ_LEN:-256}"' in text
    assert 'BASE_TOTAL_BATCH_SIZE="${BASE_TOTAL_BATCH_SIZE:-256}"' in text
    assert 'NUM_ITERATIONS="${NUM_ITERATIONS:-2}"' in text
    assert 'BASE_RESUME_FROM_STEP="${BASE_RESUME_FROM_STEP:--1}"' in text
    assert 'BASE_DISTRIBUTED_ARGS=(--no-distributed)' in text
    assert 'BASE_ATTN_ARGS=(--attn-impl=xla)' in text


def test_speedrun_stage_order_matches_public_pipeline():
    text = SPEEDRUN.read_text()

    _assert_in_order(
        text,
        [
            "-m nanochat_jax.report reset",
            "-m nanochat_jax.dataset -n \"$DATASET_TOKENIZER_SHARDS\"",
            "-m scripts.tok_train",
            "-m scripts.tok_eval",
            "-m scripts.base_train",
            "-m scripts.base_eval",
            "-m scripts.chat_sft",
            "-m scripts.chat_eval",
            "-m nanochat_jax.report generate",
        ],
    )
