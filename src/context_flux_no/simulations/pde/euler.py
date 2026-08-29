"""Compressible ideal-gas Euler equations with one physical coefficient.

``gamma``
    Ratio of specific heats.  ``gamma=1.4`` is the usual ideal-air value.

The equation of state is

    p = (gamma - 1) * (E - kinetic_energy).

Consequently, primitive initial data must be converted with

    E = p / (gamma - 1) + kinetic_energy.

The 1-D solver uses an HLL Riemann solver through PyClaw.  The 2-D solver
uses a the 2-D Roe solver with entropy corrections also through PyClaw.
"""

from collections.abc import Callable, Sequence
from typing import ClassVar, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from clawpack import pyclaw, riemann
from jaxtyping import Array, Float

from ..pdesolve import pdesolve_pyclaw, solution_to_dataset
from .base import AbstractHyperbolicConservationLaw


_RHO_FLOOR = 1.0e-12
_PRESSURE_FLOOR = 1.0e-12
_SPEED_GAP_FLOOR = 1.0e-12


def _pressure_flux_1d(
    q: Float[np.ndarray, "3 num_riemanns"],
    gamma: float,
) -> tuple[
    Float[np.ndarray, "num_riemanns"],
    Float[np.ndarray, "3 num_riemanns"],
    Float[np.ndarray, "num_riemanns"],
]:
    """Return pressure, physical flux, and sound speed for 1-D states."""

    rho = np.maximum(q[0], _RHO_FLOOR)
    momentum = q[1]
    energy = q[2]
    velocity = momentum / rho
    kinetic = 0.5 * momentum * velocity
    pressure = (gamma - 1.0) * (energy - kinetic)

    flux = np.stack(
        (
            momentum,
            momentum * velocity + pressure,
            velocity * (energy + pressure),
        ),
        axis=0,
    )
    sound_speed = np.sqrt(gamma * np.maximum(pressure, _PRESSURE_FLOOR) / rho)
    return pressure, flux, sound_speed


def riemann_euler_hll_1D(
    q_l: Float[np.ndarray, "3 num_riemanns"],
    q_r: Float[np.ndarray, "3 num_riemanns"],
    aux_l: Float[np.ndarray, "num_aux num_riemanns"],
    aux_r: Float[np.ndarray, "num_aux num_riemanns"],
    problem_data: dict[str, float],
) -> tuple[
    Float[np.ndarray, "3 2 num_riemanns"],
    Float[np.ndarray, "2 num_riemanns"],
    Float[np.ndarray, "3 num_riemanns"],
    Float[np.ndarray, "3 num_riemanns"],
]:
    """Two-wave HLL Riemann solver for the 1-D Euler equations.

    PyClaw passes left and right states with shape
    ``(num_eqns, num_riemanns)``.  ``aux_l`` and ``aux_r`` are intentionally
    unused because ``gamma`` is spatially constant.
    """

    del aux_l, aux_r
    gamma = problem_data["gamma"]

    _, flux_l, sound_l = _pressure_flux_1d(q_l, gamma)
    _, flux_r, sound_r = _pressure_flux_1d(q_r, gamma)
    velocity_l = q_l[1] / np.maximum(q_l[0], _RHO_FLOOR)
    velocity_r = q_r[1] / np.maximum(q_r[0], _RHO_FLOOR)

    speed_l = np.minimum(velocity_l - sound_l, velocity_r - sound_r)
    speed_r = np.maximum(velocity_l + sound_l, velocity_r + sound_r)
    speed_gap = np.maximum(speed_r - speed_l, _SPEED_GAP_FLOOR)

    dq = q_r - q_l
    dflux = flux_r - flux_l
    wave_l = (speed_r[None, :] * dq - dflux) / speed_gap[None, :]
    wave_r = (dflux - speed_l[None, :] * dq) / speed_gap[None, :]
    wave = np.stack((wave_l, wave_r), axis=1)
    speeds = np.stack((speed_l, speed_r), axis=0)

    amdq = np.sum(np.minimum(speeds, 0.0)[None, :, :] * wave, axis=1)
    apdq = np.sum(np.maximum(speeds, 0.0)[None, :, :] * wave, axis=1)
    return wave, speeds, amdq, apdq


class Euler1D(eqx.Module):
    """One-dimensional compressible Euler equation.

    Conservative state order: ``(rho, rho*u, E)``.
    """

    n_dim: ClassVar[int] = 1
    n_eqns: ClassVar[int] = 3
    gamma: float = eqx.field(static=True)

    def __init__(self, gamma: float = 1.4):
        if gamma <= 1.0:
            raise ValueError("gamma must be greater than 1")
        self.gamma = gamma

    @property
    def coeffs(self) -> dict[str, float]:
        return {"gamma": self.gamma}

    def primitive_to_conservative(
        self,
        rho: Float[Array, "..."],
        velocity: Float[Array, "..."],
        pressure: Float[Array, "..."],
    ) -> Float[Array, "3 ..."]:
        """Convert ``(rho, u, p)`` to ``(rho, rho*u, E)``."""

        rho = jnp.asarray(rho)
        velocity = jnp.asarray(velocity)
        pressure = jnp.asarray(pressure)
        momentum = rho * velocity
        energy = pressure / (self.gamma - 1.0) + 0.5 * rho * velocity**2
        return jnp.stack((rho, momentum, energy), axis=0)

    def solve(
        self,
        ic_factory: Callable[[Float[np.ndarray, "Nx"]], Float[np.ndarray, "3 Nx"]],
        x_span: tuple[float, float],
        Nx: int,
        t_span: tuple[float, float],
        Nt: int,
        bc: Literal["periodic"],
        **pdesolve_kwargs,
    ):
        solver = pyclaw.ClawSolver1D(riemann_euler_hll_1D)
        solver.limiters = pyclaw.limiters.tvd.MC
        solver.num_eqn = self.n_eqns
        solver.num_waves = 2
        solver.kernel_language = "Python"
        solver.cfl_desired = 0.45
        solver.cfl_max = 0.9
        solver.fwave = False

        problem_data = {"gamma": self.gamma}
        q, t, x_grid = pdesolve_pyclaw(
            solver,
            problem_data,
            ic_factory,
            x_span,
            Nx,
            t_span,
            Nt,
            bc,
            **pdesolve_kwargs,
        )
        return solution_to_dataset(q, t, (x_grid,), self.coeffs)


class Euler2D(AbstractHyperbolicConservationLaw):
    n_spatial_dims: ClassVar[int] = 2
    field_rank_names: ClassVar[tuple[tuple[int, str]]] = (
        (0, "rho"),
        (1, "m"),
        (0, "E"),
    )

    gamma: float = eqx.field(static=True)

    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma

    @property
    def parameters(self) -> dict[str, float]:
        return {"gamma": self.gamma}

    def solve(
        self,
        ic_factory: Callable[[Float[np.ndarray, " Nx"]], Float[np.ndarray, " Nx"]],
        x_spans: Sequence[tuple[float, float]],
        Nxs: Sequence[int],
        t_span: tuple[float, float],
        Nt: int,
        bc: Literal["periodic"],  # TODO: extend to other types as well
        use_rho_v_p_ics: bool = False,
        **pdesolve_kwargs,
    ) -> tuple[
        Float[np.ndarray, "time dim x_grid"],
        Float[np.ndarray, " time"],
        list[Float[np.ndarray, " ?x"]],
    ]:
        # Setting from https://www.clawpack.org/gallery/pyclaw/gallery/quadrants.html
        solver = pyclaw.ClawSolver2D(riemann.euler_4wave_2D)
        solver.transverse_waves = 2

        solver.num_eqn = self.n_eqns

        problem_data = self.parameters

        if use_rho_v_p_ics:
            # Transform (rho, u, v, p) to (rho, rho u, rho v, E)
            def ic_factory_new(xs):
                rho, u, v, p = ic_factory(xs)
                E = 0.5 * rho * (u**2 + v**2) + p / (self.gamma - 1)
                return np.stack([rho, rho * u, rho * v, E], axis=0)

        u, t, xs = pdesolve_pyclaw(
            solver,
            problem_data,
            ic_factory if not use_rho_v_p_ics else ic_factory_new,
            x_spans,
            Nxs,
            t_span,
            Nt,
            bc,
            **pdesolve_kwargs,
        )
        return u, t, xs
