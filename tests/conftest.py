"""Pytest configuration.

Forces JAX onto the CPU backend by default so unit tests do not auto-select
Apple Metal (experimental) or other accelerators that may not support every
operation. Tests that require real TPU/GPU hardware should override the
``JAX_PLATFORMS`` environment variable explicitly or use the ``tpu`` marker.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
