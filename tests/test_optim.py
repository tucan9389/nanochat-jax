"""Smoke tests for nanochat_jax.optim (MuonAdamW)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nanochat_jax.gpt import GPT, GPTConfig
from nanochat_jax.grad_utils import compute_loss_and_grad, nnx_state_to_flat_dict
from nanochat_jax.optim import (
    MuonAdamWState,
    POLAR_EXPRESS_COEFFS,
    adamw_step,
    init_optim_state,
    polar_express_iters,
    setup_optimizer_param_groups,
    step_optim,
)


def _tiny_model() -> tuple[GPT, dict]:
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
    )
    m = GPT(cfg, rngs=nnx.Rngs(0))
    m.init_weights(seed=42)
    params = nnx_state_to_flat_dict(nnx.state(m, nnx.Param))
    return m, params


def test_polar_express_coeffs_have_5_triples_of_3_floats():
    assert len(POLAR_EXPRESS_COEFFS) == 5
    for triple in POLAR_EXPRESS_COEFFS:
        assert len(triple) == 3
        for v in triple:
            assert isinstance(v, float)


def test_polar_express_iters_runs_tall_and_wide():
    rng = np.random.RandomState(0)
    tall = jnp.asarray(rng.randn(8, 4).astype(np.float32))
    wide = jnp.asarray(rng.randn(4, 8).astype(np.float32))
    out_tall = polar_express_iters(tall)
    out_wide = polar_express_iters(wide)
    # Shape preserving and finite output.
    assert out_tall.shape == tall.shape
    assert out_wide.shape == wide.shape
    assert jnp.all(jnp.isfinite(out_tall))
    assert jnp.all(jnp.isfinite(out_wide))
    # Polar Express produces an approximately orthogonal X. The output's
    # singular values should cluster around 1; ||X|| stays bounded.
    assert float(jnp.linalg.norm(out_tall)) < 100.0
    assert float(jnp.linalg.norm(out_wide)) < 100.0


def test_adamw_step_decreases_loss_proxy():
    p = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    g = jnp.array([0.5, -0.5, 0.5], dtype=jnp.float32)
    exp_avg = jnp.zeros_like(p)
    exp_avg_sq = jnp.zeros_like(p)

    p_new, exp_avg_new, exp_avg_sq_new = adamw_step(
        p, g, exp_avg, exp_avg_sq,
        step=jnp.asarray(1.0),
        lr=jnp.asarray(0.01), b1=jnp.asarray(0.9), b2=jnp.asarray(0.95),
        eps=jnp.asarray(1e-8), wd=jnp.asarray(0.0),
    )
    assert p_new.shape == p.shape
    # exp_avg should pick up the gradient sign (positive g -> positive exp_avg).
    assert float(exp_avg_new[0]) > 0
    # Param should move opposite to the gradient direction.
    assert float(p_new[0]) < float(p[0])
    assert float(p_new[1]) > float(p[1])


def test_setup_groups_covers_all_keys():
    _, params = _tiny_model()
    groups = setup_optimizer_param_groups(params, n_embd=32)
    grouped_keys = {k for g in groups for k in g["param_keys"]}
    assert grouped_keys == set(params.keys())
    kinds = {g["kind"] for g in groups}
    assert kinds == {"adamw", "muon"}


def test_setup_groups_rejects_missing_or_extra_keys():
    _, params = _tiny_model()
    bad = dict(params)
    del bad["lm_head.kernel"]
    with pytest.raises(KeyError, match="missing param keys"):
        setup_optimizer_param_groups(bad, n_embd=32)

    bad2 = dict(params)
    bad2["unknown_key"] = jnp.zeros((1,))
    with pytest.raises(KeyError, match="unknown param keys"):
        setup_optimizer_param_groups(bad2, n_embd=32)


def test_init_state_has_buffers_for_all_groups():
    _, params = _tiny_model()
    groups = setup_optimizer_param_groups(params, n_embd=32)
    state = init_optim_state(groups, params)

    assert int(state.step) == 0

    for g in groups:
        if g["kind"] == "adamw":
            for k in g["param_keys"]:
                assert k in state.adamw
                assert state.adamw[k]["exp_avg"].shape == params[k].shape
                assert state.adamw[k]["exp_avg_sq"].shape == params[k].shape
        elif g["kind"] == "muon" and g["param_keys"]:
            first_key = g["param_keys"][0]
            assert first_key in state.muon
            mom = state.muon[first_key]["momentum_buffer"]
            assert mom.shape[0] == len(g["param_keys"])


def test_step_optim_advances_step_and_updates_params():
    m, params = _tiny_model()
    groups = setup_optimizer_param_groups(params, n_embd=32)
    state = init_optim_state(groups, params)

    idx = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 6]])
    _, grads = compute_loss_and_grad(m, idx, targets)

    new_params, new_state = step_optim(state, grads, params, groups)

    assert int(new_state.step) == 1
    # At least one param should have moved (lm_head has non-zero gradient).
    moved = any(
        not np.allclose(np.asarray(new_params[k]), np.asarray(params[k]))
        for k in params
    )
    assert moved


def test_step_optim_is_deterministic():
    m, params = _tiny_model()
    groups = setup_optimizer_param_groups(params, n_embd=32)
    state = init_optim_state(groups, params)

    idx = jnp.array([[1, 2, 3, 4, 5]])
    targets = jnp.array([[2, 3, 4, 5, 6]])
    _, grads = compute_loss_and_grad(m, idx, targets)

    out_a = step_optim(state, grads, params, groups)
    out_b = step_optim(state, grads, params, groups)
    for k in out_a[0]:
        np.testing.assert_array_equal(out_a[0][k], out_b[0][k])
    assert int(out_a[1].step) == int(out_b[1].step)


def test_state_is_pytree_friendly():
    """MuonAdamWState should be a dataclass pytree (basic round-trip)."""
    state = MuonAdamWState(
        step=jnp.asarray(0, dtype=jnp.int32),
        adamw={"a": {"exp_avg": jnp.zeros((2,)), "exp_avg_sq": jnp.zeros((2,))}},
    )
    # Direct attribute access should work.
    assert int(state.step) == 0
    assert "a" in state.adamw
    assert state.muon == {}
