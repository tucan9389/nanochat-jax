"""Smoke tests for nanochat_jax.sharding (single-device CPU only here)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.sharding import NamedSharding

from nanochat_jax.gpt import GPT, GPTConfig
from nanochat_jax.grad_utils import nnx_state_to_flat_dict
from nanochat_jax.optim import (
    MuonAdamWState,
    init_optim_state,
    setup_optimizer_param_groups,
)
from nanochat_jax.sharding import (
    DEFAULT_AXIS_NAME,
    data_parallel_sharding,
    get_process_info,
    make_mesh,
    muon_state_sharding,
    replicated_sharding,
    with_data_parallel_constraint,
)


def test_default_axis_name_is_data():
    assert DEFAULT_AXIS_NAME == "data"


def test_make_mesh_returns_1d_mesh_with_default_axis():
    mesh = make_mesh()
    assert len(mesh.axis_names) == 1
    assert mesh.axis_names[0] == "data"
    assert mesh.devices.shape == (jax.device_count(),)


def test_make_mesh_validates_n_devices():
    with pytest.raises(ValueError, match="exceeds jax.device_count"):
        make_mesh(n_devices=jax.device_count() + 1)
    with pytest.raises(ValueError, match=">= 1"):
        make_mesh(n_devices=0)


def test_data_parallel_sharding_default_batch_axis():
    mesh = make_mesh()
    sharding = data_parallel_sharding(mesh)
    assert isinstance(sharding, NamedSharding)
    spec = sharding.spec
    assert spec[0] == "data"


def test_data_parallel_sharding_other_axis():
    mesh = make_mesh()
    sharding = data_parallel_sharding(mesh, batch_axis=2)
    spec = sharding.spec
    assert spec[2] == "data"


def test_data_parallel_sharding_rejects_negative_axis():
    mesh = make_mesh()
    with pytest.raises(ValueError, match="must be >= 0"):
        data_parallel_sharding(mesh, batch_axis=-1)


def test_replicated_sharding_returns_named_sharding():
    mesh = make_mesh()
    sharding = replicated_sharding(mesh)
    assert isinstance(sharding, NamedSharding)


def test_with_data_parallel_constraint_runs_inside_jit():
    mesh = make_mesh()
    @jax.jit
    def f(x):
        return with_data_parallel_constraint(x, mesh)
    x = jnp.ones((4, 8))
    y = f(x)
    assert y.shape == x.shape


def test_get_process_info_returns_triple():
    info = get_process_info()
    assert len(info) == 3
    assert info[0] == 0
    assert info[1] == 1
    assert info[2] >= 1


def test_muon_state_sharding_returns_pytree_of_namedshardings():
    cfg = GPTConfig(
        sequence_len=16, vocab_size=32, n_layer=2,
        n_head=2, n_kv_head=2, n_embd=32,
    )
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    params = nnx_state_to_flat_dict(nnx.state(m, nnx.Param))
    groups = setup_optimizer_param_groups(params, n_embd=32)
    state = init_optim_state(groups, params)

    mesh = make_mesh()
    spec = muon_state_sharding(mesh, state)
    assert isinstance(spec, MuonAdamWState)
    assert isinstance(spec.step, NamedSharding)
    for inner in spec.adamw.values():
        for s in inner.values():
            assert isinstance(s, NamedSharding)
    for inner in spec.muon.values():
        for s in inner.values():
            assert isinstance(s, NamedSharding)


def test_muon_state_sharding_rejects_non_muon_state():
    mesh = make_mesh()
    with pytest.raises(TypeError, match="expected MuonAdamWState"):
        muon_state_sharding(mesh, "not a state")
