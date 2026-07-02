"""NanochatLinear: bias-free Linear with fp32 master weights and a dot_general hook."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx


class NanochatLinear(nnx.Module):
    """Linear layer (no bias) that casts master weights to the input dtype on call.

    Mirrors PyTorch ``nn.Linear`` with weight cast in forward, the pattern used
    throughout nanochat for explicit fp32-master / bf16-compute precision control.

    Hooks:
    - ``kernel_axes``: optional sharding metadata for the kernel ``nnx.Param``.
      ``eager_sharding=False`` defers mesh activation until the train step is jit-traced.
    - ``dot_general``: pluggable matmul function with the ``lax.dot_general`` signature.
      Quantization libraries (e.g. Qwix) can swap this without changing the call sites.

    Kernel shape follows JAX/Flax convention ``(in_features, out_features)``. PyTorch
    ``nn.Linear.weight`` is ``(out_features, in_features)``; the weight converter
    transposes when porting checkpoints across frameworks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight_dtype: jnp.dtype = jnp.float32,
        kernel_init: Callable = nnx.initializers.lecun_normal(),
        kernel_axes: tuple[str | None, ...] | None = None,
        precision: jax.lax.PrecisionLike = None,
        dot_general: Callable = jax.lax.dot_general,
        rngs: nnx.Rngs,
    ):
        kernel_shape = (in_features, out_features)
        kernel_value = kernel_init(rngs.params(), kernel_shape, weight_dtype)
        if kernel_axes is not None:
            if len(kernel_axes) != len(kernel_shape):
                raise ValueError(
                    f"kernel_axes length {len(kernel_axes)} does not match "
                    f"kernel rank {len(kernel_shape)} (expected 2 for "
                    f"(in_features, out_features))"
                )
            self.kernel = nnx.Param(
                kernel_value,
                out_sharding=kernel_axes,
                eager_sharding=False,
            )
        else:
            self.kernel = nnx.Param(kernel_value)

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_axes = kernel_axes
        self.precision = precision
        self.dot_general = dot_general

    def __call__(self, x: jax.Array) -> jax.Array:
        w = self.kernel[...].astype(x.dtype)
        contract_dims = (((x.ndim - 1,), (0,)), ((), ()))
        return self.dot_general(x, w, contract_dims, precision=self.precision)
