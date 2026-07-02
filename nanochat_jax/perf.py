"""Performance utilities — hardware peak FLOPS table + MFU helpers.

Cross-verified against:
- NVIDIA H100 / A100 datasheets (dense BF16 with FP32 accumulate)
- Google Cloud TPU docs (per-chip BF16 throughput)
- Levanter ``flop_utils.py`` (Stanford CRFM's JAX trainer)
- JAX scaling book (TPUs / GPUs chapters)
- karpathy/nanochat ``common.py::get_peak_flops``
"""

from __future__ import annotations

import jax

# BF16 dense peak TFLOPS per chip (FP32 accumulate, no sparsity factor).
# Numbers are per-chip; multiply by chip count for total slice peak.
_DEVICE_BF16_PEAK_TFLOPS: dict[str, float] = {
    # NVIDIA H100 — NVIDIA H100 Tensor Core GPU Datasheet
    "h100-sxm":  989.0,
    "h100-pcie": 756.0,
    "h100-nvl":  835.0,
    # NVIDIA A100 — NVIDIA A100 Tensor Core GPU Datasheet
    "a100":      312.0,
    # Google TPU — Google Cloud TPU docs
    "tpu v3":     61.5,   # per TensorCore (v3 chip = 2 cores)
    "tpu v4":    275.0,   # per chip (v4 jax device = 1 chip)
    "tpu v5p":   459.0,   # per chip
    "tpu v5":    459.0,   # alias — jax.Device.device_kind returns "TPU v5"
    "tpu v5 lite": 197.0, # v5e
    "tpu v5e":   197.0,   # alias
    "tpu v6 lite": 918.0, # v6e (Trillium)
    "tpu v6e":   918.0,   # alias
}


def get_peak_bf16_tflops(device: jax.Device | None = None) -> float | None:
    """Return BF16 dense peak TFLOPS per chip for ``device``.

    Returns ``None`` if the device kind is not in the table. The caller can
    fall back to a hand-specified value (e.g. via CLI).
    """
    if device is None:
        try:
            device = jax.devices()[0]
        except Exception:
            return None

    kind = device.device_kind.lower()

    # TPU first — kind looks like "TPU v5" or "TPU v6 lite"
    if kind.startswith("tpu"):
        # Longest-prefix wins so "tpu v5 lite" beats "tpu v5"
        best = None
        for key, val in _DEVICE_BF16_PEAK_TFLOPS.items():
            if key.startswith("tpu") and kind.startswith(key):
                if best is None or len(key) > len(best[0]):
                    best = (key, val)
        if best is not None:
            return best[1]

    # NVIDIA H100 variants
    if "h100" in kind:
        if "sxm" in kind or "hbm3" in kind:
            return _DEVICE_BF16_PEAK_TFLOPS["h100-sxm"]
        if "pcie" in kind:
            return _DEVICE_BF16_PEAK_TFLOPS["h100-pcie"]
        if "nvl" in kind:
            return _DEVICE_BF16_PEAK_TFLOPS["h100-nvl"]
        return _DEVICE_BF16_PEAK_TFLOPS["h100-sxm"]

    if "a100" in kind:
        return _DEVICE_BF16_PEAK_TFLOPS["a100"]

    return None


def compute_mfu(
    *,
    num_flops_per_token: int,
    tokens_per_step: int,
    step_seconds: float,
    peak_tflops_per_chip: float,
    num_chips: int,
) -> float:
    """Model FLOPS Utilization, fraction of hardware peak ∈ [0, 1].

    Definition (PaLM paper, Chowdhery+ 2022):
        MFU = model_flops_per_second / hardware_peak_flops_per_second

    The numerator is the model's *required* FLOPs (not including
    re-materialization). The denominator is peak BF16 across all chips.
    """
    if step_seconds <= 0 or peak_tflops_per_chip <= 0 or num_chips <= 0:
        return 0.0
    achieved_flops_per_sec = num_flops_per_token * tokens_per_step / step_seconds
    peak_flops_per_sec = peak_tflops_per_chip * 1e12 * num_chips
    return achieved_flops_per_sec / peak_flops_per_sec
