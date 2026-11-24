"""Named solver configurations and tolerances shared by the test tiers, and the stub models
of the evaluation tests. Every call site uses a configuration or tolerance by its name, and
each entry carries its one-line reason."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


SHARPCLAW_SSP33 = {"time_integrator": "SSP33", "cfl_desired": 0.4, "cfl_max": 0.45}
"""SharpClaw with the third-order SSP Runge–Kutta (SSP coefficient 1; `pyclaw/sharpclaw/
solver.py:137`). Clawpack provides CFL defaults only for SSP104 (2.45 / 2.5 at coefficient 6,
`solver.py:146`), so SSP33 needs them spelled out: the same numbers per unit coefficient."""

PRESSURE_ROUND_TRIP_RTOL = 1.0e-3
"""Relative tolerance of `Euler2D.pressure` on a state from `to_conserved`, in float32: the
pressure is `E` minus the kinetic energy, so the rounding of `E` (about 6e-8 relative) is
amplified by `E / p`, at most about 1200 for the strategy of `tests/unit/test_admissible.py`."""

ROLLOUT_ATOL = 1.0e-4
"""Absolute tolerance of a relative error of `evaluation.rollout` against its closed form,
in float32: the errors are norms over at most 20 frames of 64 cells of order-one values."""


class HoldLastFrame(eqx.Module):
    """A stub `evaluation.Operator` that predicts the last context frame at every step, which
    is the identity baseline."""

    def rollout(
        self,
        u: Float[Array, " time channels *spatial_dims"],
        args: Any,
        num_steps: int,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, " {num_steps} channels *spatial_dims"], None]:
        """Repeat the last frame of `u` for `num_steps` steps.

        **Arguments:**

        - `u`: The context window.
        - `args`, `key`, `inference`: Unused.
        - `num_steps`: The number of frames to predict.

        **Returns:**

        The predicted frames and `None`.
        """
        return jnp.broadcast_to(u[-1], (num_steps, *u.shape[1:])), None


class ShiftByOneCell(eqx.Module):
    """A stub `evaluation.Operator` that moves the last context frame by one cell per step
    along the last axis, periodically. Its error against a wave that translates by one
    cell per frame is zero."""

    def rollout(
        self,
        u: Float[Array, " time channels *spatial_dims"],
        args: Any,
        num_steps: int,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, " {num_steps} channels *spatial_dims"], None]:
        """Shift the last frame of `u` by `n` cells at step `n`, for `n` from 1 to
        `num_steps`.

        **Arguments:**

        - `u`: The context window.
        - `args`, `key`, `inference`: Unused.
        - `num_steps`: The number of frames to predict.

        **Returns:**

        The predicted frames and `None`.
        """
        shifts = jnp.arange(1, num_steps + 1)
        return jax.vmap(lambda shift: jnp.roll(u[-1], shift, axis=-1))(shifts), None
