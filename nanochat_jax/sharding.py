"""``jax.sharding.Mesh`` helpers for multi-TPU training.

A 1-D ``("data",)`` mesh is sufficient for single-host TPU pods like v6e-8.
2-D meshes (``("data", "tensor")``) and tensor-parallel sharding are out of
scope for this port. The Muon optimizer's stacked momentum buffer is sharded
along its first axis, which the GSPMD partitioner translates into the
``reduce_scatter`` / ``all_gather`` collectives that PyTorch's
``DistMuonAdamW`` issues by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.lax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

if TYPE_CHECKING:
    from nanochat_jax.optim import MuonAdamWState

DEFAULT_AXIS_NAME = "data"


def make_mesh(
    n_devices: int | None = None,
    axis_name: str = DEFAULT_AXIS_NAME,
) -> Mesh:
    """Create a 1-D ``Mesh`` over ``n_devices`` along ``axis_name``.

    ``n_devices=None`` uses ``jax.device_count()``. On a 1-device mesh
    (e.g. CPU), ``with_sharding_constraint`` is effectively a no-op because
    XLA simplifies single-device shardings to replicated.
    """
    available = jax.device_count()
    if n_devices is None:
        n_devices = available
    if n_devices > available:
        raise ValueError(
            f"n_devices={n_devices} exceeds jax.device_count()={available}"
        )
    if n_devices < 1:
        raise ValueError(f"n_devices={n_devices} must be >= 1")
    devices_array = mesh_utils.create_device_mesh((n_devices,))
    return Mesh(devices_array, (axis_name,))


def data_parallel_sharding(mesh: Mesh, batch_axis: int = 0) -> NamedSharding:
    """Return a ``NamedSharding`` that shards ``batch_axis`` along the mesh axis."""
    if batch_axis < 0:
        raise ValueError(f"batch_axis={batch_axis} must be >= 0")
    axis_name = mesh.axis_names[0]
    if batch_axis == 0:
        spec = PartitionSpec(axis_name)
    else:
        partitions: list[str | None] = [None] * (batch_axis + 1)
        partitions[batch_axis] = axis_name
        spec = PartitionSpec(*partitions)
    return NamedSharding(mesh, spec)


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Return a ``NamedSharding`` that replicates a tensor across all devices."""
    return NamedSharding(mesh, PartitionSpec())


def with_data_parallel_constraint(x, mesh: Mesh, batch_axis: int = 0):
    """Apply ``jax.lax.with_sharding_constraint`` for data-parallel sharding.

    Must be called inside a ``jax.jit``-compiled function to take effect; outside
    jit, JAX may treat it as a no-op for single-device cases.
    """
    sharding = data_parallel_sharding(mesh, batch_axis=batch_axis)
    return jax.lax.with_sharding_constraint(x, sharding)


def get_process_info() -> tuple[int, int, int]:
    """Return ``(process_index, process_count, local_device_count)``.

    Defaults to ``(0, 1, jax.local_device_count())`` on CPU/single-host TPU.
    Distinct from :func:`nanochat_jax.common.get_dist_info`, which reads the
    PyTorch ``RANK``/``LOCAL_RANK``/``WORLD_SIZE`` env vars.
    """
    return (jax.process_index(), jax.process_count(), jax.local_device_count())


def muon_state_sharding(
    mesh: Mesh,
    optim_state: "MuonAdamWState",
) -> "MuonAdamWState":
    """Build a sharding-spec pytree mirroring :class:`MuonAdamWState`.

    Mapping:
    - ``state.step`` -> replicated.
    - ``state.adamw[k]["exp_avg" | "exp_avg_sq"]`` -> replicated (per-param
      state with no stacking axis to shard; ZeRO-3 sharding of AdamW would
      require sharding the param itself).
    - ``state.muon[k]["momentum_buffer" | "second_momentum_buffer"]`` ->
      sharded along axis 0 (the stacked params). The GSPMD partitioner then
      generates the same ``reduce_scatter`` + ``all_gather`` traffic that
      PyTorch's ``DistMuonAdamW`` Muon path issues by hand.

    If a group's stacked size is not a multiple of the mesh size, that
    group's buffers fall back to replicated (avoiding ``IndivisibleError``).
    The returned pytree has the same structure as the input but every leaf
    is a ``NamedSharding`` instance, ready to pass to ``nnx.jit`` /
    ``jax.jit`` as ``in_shardings`` / ``out_shardings``.
    """
    from nanochat_jax.optim import MuonAdamWState

    if not isinstance(optim_state, MuonAdamWState):
        raise TypeError(
            f"muon_state_sharding: expected MuonAdamWState, got "
            f"{type(optim_state).__name__}"
        )

    replicated = replicated_sharding(mesh)
    muon_stacked = data_parallel_sharding(mesh, batch_axis=0)

    mesh_size = int(mesh.devices.shape[0])

    def _muon_spec(group_state: dict):
        first_axis = int(group_state["momentum_buffer"].shape[0])
        return muon_stacked if first_axis % mesh_size == 0 else replicated

    return MuonAdamWState(
        step=replicated,
        adamw={
            k: {
                "exp_avg": replicated,
                "exp_avg_sq": replicated,
            }
            for k in optim_state.adamw
        },
        muon={
            k: {
                "momentum_buffer": _muon_spec(g),
                "second_momentum_buffer": _muon_spec(g),
            }
            for k, g in optim_state.muon.items()
        },
    )
