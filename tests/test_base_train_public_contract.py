from __future__ import annotations

import jax.numpy as jnp
import pytest

from nanochat_jax import base_train_config
from scripts import base_train


def test_make_config_d24_matches_upstream_shape_contract():
    cfg = base_train.make_config(
        depth=24,
        aspect_ratio=64,
        head_dim=128,
        sequence_len=2048,
        vocab_size=32768,
        window_pattern="SSSL",
        bf16=True,
    )

    assert cfg.sequence_len == 2048
    assert cfg.vocab_size == 32768
    assert cfg.n_layer == 24
    assert cfg.n_embd == 1536
    assert cfg.n_head == 12
    assert cfg.n_kv_head == 12
    assert cfg.window_pattern == "SSSL"
    assert cfg.compute_dtype == jnp.bfloat16


def test_make_config_rounds_model_dim_up_to_head_dim_multiple():
    cfg = base_train.make_config(
        depth=7,
        aspect_ratio=65,
        head_dim=128,
        bf16=False,
    )

    assert cfg.n_embd == 512
    assert cfg.n_head == 4
    assert cfg.n_kv_head == 4
    assert cfg.compute_dtype == jnp.float32


def test_make_d12_config_preserves_legacy_defaults():
    cfg = base_train.make_d12_config(depth=24)

    assert cfg.n_layer == 24
    assert cfg.n_embd == 768
    assert cfg.n_head == 6
    assert cfg.n_kv_head == 6


def test_base_train_script_reexports_config_helpers():
    assert base_train.make_config is base_train_config.make_config
    assert base_train.make_d12_config is base_train_config.make_d12_config


def test_total_batch_size_resolution_matches_d24_release_batch():
    explicit = base_train_config.compute_total_batch_size_tokens(
        total_batch_size=524_288,
        device_batch_size=4,
        seq_len=2048,
        device_count=8,
        grad_accum_steps=16,
    )
    inferred = base_train_config.compute_total_batch_size_tokens(
        total_batch_size=-1,
        device_batch_size=4,
        seq_len=2048,
        device_count=8,
        grad_accum_steps=8,
    )

    assert explicit == 524_288
    assert inferred == 524_288


def test_resolve_num_iterations_matches_d24_ratio12_horizon():
    num_iterations, source = base_train_config.resolve_num_iterations(
        num_iterations=-1,
        target_param_data_ratio=12,
        scaling_params=729_810_624,
        total_batch_size_tokens=524_288,
    )

    assert num_iterations == 16_704
    assert "target_tokens=8,757,727,488" in source
    assert "scaling_params=729,810,624" in source


def test_resolve_num_iterations_explicit_override_wins():
    num_iterations, source = base_train_config.resolve_num_iterations(
        num_iterations=12500,
        target_param_data_ratio=12,
        scaling_params=729_810_624,
        total_batch_size_tokens=524_288,
    )

    assert num_iterations == 12_500
    assert source == "explicit --num-iterations"


def test_d24_weight_decay_scaling_reference_value():
    d12_scaling_params = 110_100_912
    d24_scaling_params = 729_810_624
    batch_lr_scale = base_train_config.compute_batch_lr_scale(524_288)
    weight_decay_scaled = base_train_config.compute_weight_decay_scaled(
        weight_decay=0.28,
        total_batch_size_tokens=524_288,
        d_x_scaling_params=d24_scaling_params,
    )

    assert d12_scaling_params == base_train_config.D12_SCALING_PARAMS
    assert batch_lr_scale == 1.0
    assert weight_decay_scaled == pytest.approx(0.042241445035472655)
