"""Verification tier (ADR-0008): oracle-based numerics checks, `-m slow`, run with
`jax_enable_x64` so that high-order convergence is measurable above the fp32 floor.

    JAX_PLATFORMS=cpu uv run pytest tests/verification -m slow
"""

import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def _enable_x64():
    jax.config.update("jax_enable_x64", True)
    yield
