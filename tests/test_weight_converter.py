"""Tests for nanochat_jax.weight_converter (PT state_dict <-> JAX pytree)."""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from flax import nnx

from nanochat_jax.layers import NanochatLinear
from nanochat_jax.weight_converter import (
    inject_pt_linear_weight,
    jax_to_pt_state_dict,
    pt_state_dict_to_jax,
)


class _FakeLinear(nn.Linear):
    """Mirrors nanochat ``Linear`` (fp32 master + cast to input dtype in forward)."""

    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class _FakeAttn(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, has_ve_):
        super().__init__()
        head_dim = n_embd // n_head
        self.c_q = _FakeLinear(n_embd, n_head * head_dim, bias=False)
        self.c_k = _FakeLinear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = _FakeLinear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = _FakeLinear(n_embd, n_embd, bias=False)
        if has_ve_:
            self.ve_gate = _FakeLinear(12, n_kv_head, bias=False)


class _FakeMLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = _FakeLinear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = _FakeLinear(4 * n_embd, n_embd, bias=False)


class _FakeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, has_ve_):
        super().__init__()
        self.attn = _FakeAttn(n_embd, n_head, n_kv_head, has_ve_)
        self.mlp = _FakeMLP(n_embd)


class _FakeGPT(nn.Module):
    """torch.nn.Module mirroring nanochat.gpt.GPT state_dict structure."""

    def __init__(self, n_layer, n_embd, n_head, n_kv_head, padded_vocab):
        super().__init__()
        head_dim = n_embd // n_head
        kv_dim = n_kv_head * head_dim
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab, n_embd),
                "h": nn.ModuleList(
                    [
                        _FakeBlock(
                            n_embd,
                            n_head,
                            n_kv_head,
                            has_ve_=(i % 2 == (n_layer - 1) % 2),
                        )
                        for i in range(n_layer)
                    ]
                ),
            }
        )
        self.lm_head = _FakeLinear(n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))
        self.smear_gate = _FakeLinear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab, kv_dim)
                for i in range(n_layer)
                if i % 2 == (n_layer - 1) % 2
            }
        )


def _make_synthetic_gpt(
    *,
    n_layer=1,
    n_embd=64,
    n_head=4,
    n_kv_head=4,
    padded_vocab=128,
    seed=42,
) -> nn.Module:
    torch.manual_seed(seed)
    return _FakeGPT(n_layer, n_embd, n_head, n_kv_head, padded_vocab)


def _linear_keys_with_shape(state_dict):
    for k, v in state_dict.items():
        if not k.endswith(".weight"):
            continue
        if k == "transformer.wte.weight":
            continue
        if k.startswith("value_embeds.") and k.count(".") == 2:
            continue
        out_f, in_f = v.shape
        yield k, in_f, out_f


@pytest.mark.parametrize(
    "n_layer,expected_keys",
    [(1, 15), (4, 35), (12, 91)],
    ids=["d1", "d4_sparse_VE", "d12_full"],
)
def test_round_trip_identity_fp32(n_layer, expected_keys):
    m = _make_synthetic_gpt(n_layer=n_layer)
    sd = m.state_dict()
    assert len(sd) == expected_keys

    jax_pytree = pt_state_dict_to_jax(sd)
    assert len(jax_pytree) == expected_keys

    sd_back = jax_to_pt_state_dict(jax_pytree)
    assert set(sd_back.keys()) == set(sd.keys())
    for k in sd:
        assert sd_back[k].shape == sd[k].shape, f"shape at {k}"
        diff = (sd[k] - sd_back[k]).abs().max().item()
        assert diff == 0.0, f"value diff at {k}: {diff}"


@pytest.mark.parametrize(
    "compute_dtype_jax,compute_dtype_torch,atol",
    [
        (jnp.float32, torch.float32, 1e-6),
        (jnp.bfloat16, torch.bfloat16, 1e-2),
    ],
    ids=["fp32", "bf16"],
)
def test_per_linear_forward_parity(compute_dtype_jax, compute_dtype_torch, atol):
    m = _make_synthetic_gpt(n_layer=1)
    sd = m.state_dict()
    rng = np.random.RandomState(123)

    linear_keys = list(_linear_keys_with_shape(sd))
    assert len(linear_keys) == 9, f"expected 9 Linear keys for d1, got {len(linear_keys)}"

    for pt_key, in_f, out_f in linear_keys:
        pt_w = sd[pt_key]
        nnx_linear = NanochatLinear(in_f, out_f, rngs=nnx.Rngs(0))
        inject_pt_linear_weight(nnx_linear, pt_w)

        np_x = rng.randn(2, in_f).astype(np.float32)
        pt_x = torch.from_numpy(np_x).to(compute_dtype_torch)
        jx_x = jnp.asarray(np_x).astype(compute_dtype_jax)

        pt_y = F.linear(pt_x, pt_w.to(compute_dtype_torch))
        jx_y = nnx_linear(jx_x)

        pt_y_np = pt_y.float().numpy()
        jx_y_np = np.asarray(jx_y).astype(np.float32)
        np.testing.assert_allclose(
            jx_y_np, pt_y_np, atol=atol, err_msg=f"forward mismatch at {pt_key}"
        )


@pytest.mark.parametrize(
    "storage_dtype_torch",
    [torch.float32, torch.bfloat16],
    ids=["fp32", "bf16"],
)
def test_per_embedding_forward_parity(storage_dtype_torch):
    m = _make_synthetic_gpt(n_layer=1)
    sd = m.state_dict()

    sd_cast = {
        k: (
            v.to(storage_dtype_torch)
            if k == "transformer.wte.weight"
            or (k.startswith("value_embeds.") and k.endswith(".weight"))
            else v
        )
        for k, v in sd.items()
    }

    jax_pytree = pt_state_dict_to_jax(sd_cast)

    rng = np.random.RandomState(456)
    ids_np = rng.randint(0, 128, size=(2, 8)).astype(np.int64)

    for pt_key, jax_key in [
        ("transformer.wte.weight", "transformer.wte.embedding"),
        ("value_embeds.0.weight", "value_embeds.0.embedding"),
    ]:
        wte_pt = sd_cast[pt_key]
        wte_jx = jax_pytree[jax_key]
        emb_pt = nn.Embedding.from_pretrained(wte_pt, freeze=True)
        pt_y = emb_pt(torch.from_numpy(ids_np))
        jx_y = jnp.take(wte_jx, jnp.asarray(ids_np), axis=0)

        pt_y_f = pt_y.float().numpy()
        jx_y_f = np.asarray(jx_y).astype(np.float32)
        np.testing.assert_array_equal(
            jx_y_f, pt_y_f, err_msg=f"embedding lookup mismatch at {pt_key}"
        )


def test_scalar_preserved():
    m = _make_synthetic_gpt(n_layer=4)
    sd = m.state_dict()

    jax_pytree = pt_state_dict_to_jax(sd)
    sd_back = jax_to_pt_state_dict(jax_pytree)

    for sk in ("resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda"):
        assert sk in jax_pytree
        assert jax_pytree[sk].shape == sd[sk].shape
        np.testing.assert_array_equal(
            np.asarray(jax_pytree[sk]), sd[sk].detach().numpy()
        )
        assert sd_back[sk].shape == sd[sk].shape
        assert sd_back[sk].dtype == sd[sk].dtype
        diff = (sd[sk] - sd_back[sk]).abs().max().item()
        assert diff == 0.0, f"round-trip value diff at {sk}: {diff}"


def test_mixed_precision_round_trip():
    """Embedding bf16 + Linear/Scalar fp32 round-trip preserves per-key dtype."""
    m = _make_synthetic_gpt(n_layer=4)
    sd = m.state_dict()

    sd_mixed = {}
    for k, v in sd.items():
        is_embedding = k == "transformer.wte.weight" or (
            k.startswith("value_embeds.") and k.endswith(".weight")
        )
        sd_mixed[k] = v.to(torch.bfloat16) if is_embedding else v

    bf16_count = sum(1 for v in sd_mixed.values() if v.dtype == torch.bfloat16)
    fp32_count = sum(1 for v in sd_mixed.values() if v.dtype == torch.float32)
    assert bf16_count == 3, f"expected 3 bf16 keys (1 wte + 2 ve), got {bf16_count}"
    assert fp32_count == 32, f"expected 32 fp32 keys, got {fp32_count}"

    jax_pytree = pt_state_dict_to_jax(sd_mixed)
    sd_back = jax_to_pt_state_dict(jax_pytree)

    for k in sd_mixed:
        assert sd_back[k].dtype == sd_mixed[k].dtype, (
            f"dtype mismatch at {k}: in={sd_mixed[k].dtype}, "
            f"back={sd_back[k].dtype}"
        )
        a, b = sd_mixed[k], sd_back[k]
        if a.dtype == torch.bfloat16:
            diff = (a.float() - b.float()).abs().max().item()
        else:
            diff = (a - b).abs().max().item()
        assert diff == 0.0, f"value diff at {k}: {diff}"


def test_unknown_pt_key_raises():
    fake_sd = {"fake.unknown_param.weight": torch.randn(2, 2)}
    with pytest.raises(ValueError, match="Unknown nanochat state_dict key"):
        pt_state_dict_to_jax(fake_sd)


def test_unknown_jax_kernel_key_raises():
    fake_jax = {"fake.unknown.kernel": jnp.zeros((2, 2))}
    with pytest.raises(ValueError, match="Unknown nanochat state_dict key"):
        jax_to_pt_state_dict(fake_jax)


def test_unknown_jax_scalar_key_raises():
    fake_jax = {"fake_scalar": jnp.zeros((2,))}
    with pytest.raises(ValueError, match="Unknown nanochat state_dict key"):
        jax_to_pt_state_dict(fake_jax)


def test_jax_suffix_mismatch_raises():
    fake_jax = {"lm_head.embedding": jnp.zeros((128, 64))}
    with pytest.raises(ValueError, match="does not match expected category"):
        jax_to_pt_state_dict(fake_jax)


def _load_real_nanochat():
    """Try to import upstream nanochat.gpt from external/. Returns None on failure."""
    here = os.path.dirname(os.path.abspath(__file__))
    upstream_path = os.path.join(here, "..", "external", "nanochat")
    upstream_path = os.path.normpath(upstream_path)
    if not os.path.isdir(upstream_path):
        return None
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)
    try:
        from nanochat.gpt import GPT, GPTConfig
    except Exception:
        return None
    return GPT, GPTConfig


@pytest.mark.parametrize("n_layer", [1, 4], ids=["d1", "d4"])
def test_real_nanochat_sanity(n_layer):
    """When external/nanochat/ is present, verify converter accepts the real
    state_dict and round-trips bit-exactly."""
    real = _load_real_nanochat()
    if real is None:
        pytest.skip("nanochat not available at external/nanochat/")
    GPT, GPTConfig = real

    cfg = GPTConfig(
        sequence_len=128,
        vocab_size=100,
        n_layer=n_layer,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
    )
    torch.manual_seed(7)
    m_real = GPT(cfg, pad_vocab_size_to=64)
    sd_real = m_real.state_dict()

    m_synth = _make_synthetic_gpt(n_layer=n_layer)
    sd_synth = m_synth.state_dict()

    assert set(sd_real.keys()) == set(sd_synth.keys()), (
        f"key mismatch real vs synth: "
        f"real-synth={set(sd_real) - set(sd_synth)}, "
        f"synth-real={set(sd_synth) - set(sd_real)}"
    )
    for k in sd_real:
        assert sd_real[k].shape == sd_synth[k].shape, f"shape at {k}"

    jax_pytree = pt_state_dict_to_jax(sd_real)
    sd_back = jax_to_pt_state_dict(jax_pytree)
    assert set(sd_back.keys()) == set(sd_real.keys())
    for k in sd_real:
        assert sd_back[k].dtype == sd_real[k].dtype
        diff = (sd_real[k] - sd_back[k]).abs().max().item()
        assert diff == 0.0, f"round-trip diff at {k}: {diff}"


def test_inject_helper_shape_mismatch_raises():
    nnx_linear = NanochatLinear(64, 32, rngs=nnx.Rngs(0))
    bad_w = torch.randn(16, 8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        inject_pt_linear_weight(nnx_linear, bad_w)
