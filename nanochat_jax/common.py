"""Common utilities: cache directory, distributed info, lock-protected downloads."""

import os
import urllib.request


def get_base_dir() -> str:
    """Returns the base cache directory for nanochat-jax intermediates.

    Default: ``~/.cache/nanochat-jax/``. Override with ``NANOCHAT_JAX_BASE_DIR``.
    A separate cache from upstream nanochat keeps PyTorch and JAX intermediates
    isolated so both can coexist on the same machine.
    """
    env_dir = os.environ.get("NANOCHAT_JAX_BASE_DIR")
    if env_dir:
        nanochat_dir = env_dir
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat-jax")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def get_dist_info():
    """Returns ``(ddp_active, rank, local_rank, world_size)`` from environment.

    Reads PyTorch-style ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` env vars.
    Returns ``(False, 0, 0, 1)`` for single-rank execution. Used by the data
    loader for parquet row-group sharding across hosts.
    """
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return (
            True,
            int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
        )
    return False, 0, 0, 1


def setup_distributed_env_vars() -> None:
    """Set ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` from JAX state.

    Call after ``jax.distributed.initialize`` (or in single-host contexts where
    ``jax.process_index()`` defaults to 0). Bridges the JAX-native multi-host
    semantic to the PyTorch env-var semantic so :func:`get_dist_info` reads the
    correct rank info.

    On TPU pod slices each host runs a single Python process (one rank), so
    ``LOCAL_RANK="0"`` is always correct (different from PyTorch's per-GPU
    semantic where LOCAL_RANK enumerates GPUs within a host).
    """
    import jax  # local import: common.py is used by CPU-only tests too

    process_idx = jax.process_index()
    process_count = jax.process_count()
    os.environ["RANK"] = str(process_idx)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(process_count)


def download_file_with_lock(url: str, filename: str, postprocess_fn=None) -> str:
    """Download a URL to ``base_dir/filename`` with an exclusive file lock.

    Idempotent across ranks: if the file already exists the function returns
    immediately. Uses ``fcntl.flock`` (Unix only). The optional ``postprocess_fn``
    runs on the downloaded path while the lock is still held, which lets
    callers atomically extract or move the file before other ranks observe it.
    """
    import fcntl  # Unix only

    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)

        if os.path.exists(file_path):
            return file_path

        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()

        with open(file_path, "wb") as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path
