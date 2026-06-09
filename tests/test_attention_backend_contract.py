from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nanochat_jax import gpt


def test_dpa_window_size_maps_full_context_sentinel_to_none():
    assert gpt._dpa_window_size((-1, 0)) is None
    assert gpt._dpa_window_size((512, 0)) == (512, 0)


def test_splash_mesh_set_get_round_trip_none():
    gpt.set_splash_mesh(None)
    assert gpt.get_splash_mesh() is None


def test_xla_attention_backend_matches_jax_dpa_shape_and_finiteness():
    q = jnp.ones((2, 4, 2, 8), dtype=jnp.float32)
    k = jnp.ones((2, 4, 2, 8), dtype=jnp.float32)
    v = jnp.arange(2 * 4 * 2 * 8, dtype=jnp.float32).reshape(2, 4, 2, 8)

    y = jax.nn.dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        local_window_size=gpt._dpa_window_size((-1, 0)),
        implementation="xla",
    )

    assert y.shape == q.shape
    assert np.isfinite(np.asarray(y)).all()
