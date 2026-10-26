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
import numpy as np
from clawpack import pyclaw, riemann
from jaxtyping import Float

from ..pdesolve import pdesolve_pyclaw
from .base import AbstractHyperbolicConservationLaw


_RHO_FLOOR = 1.0e-12
_PRESSURE_FLOOR = 1.0e-12
_SPEED_GAP_FLOOR = 1.0e-12


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
