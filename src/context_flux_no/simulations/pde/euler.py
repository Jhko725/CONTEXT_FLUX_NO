"""Compressible ideal-gas Euler equations with one physical coefficient.

``gamma``
    Ratio of specific heats.  ``gamma=1.4`` is the usual ideal-air value.

The equation of state is

    p = (gamma - 1) * (E - kinetic_energy).

Consequently, primitive initial data must be converted with

    E = p / (gamma - 1) + kinetic_energy.

The 2-D solver uses Clawpack's compiled 4-wave Roe solver (`riemann.euler_4wave_2D`) with
transverse propagation through PyClaw (the Clawpack quadrants gallery setting).
"""

from collections.abc import Sequence
from typing import Any, ClassVar

import equinox as eqx
import numpy as np
from clawpack import riemann

from context_flux_no.simulations.backend import SolverBackend
from context_flux_no.simulations.pde.base import (
    AbstractHyperbolicConservationLaw,
    ICFactory,
)
from context_flux_no.simulations.pyclaw import PyClawSpec


_RHO_FLOOR = 1.0e-12
_PRESSURE_FLOOR = 1.0e-12
_SPEED_GAP_FLOOR = 1.0e-12


class Euler2D(AbstractHyperbolicConservationLaw):
    n_spatial_dims: ClassVar[int] = 2
    field_rank_names: ClassVar[tuple[tuple[int, str], ...]] = (
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

    @property
    def pyclaw_spec(self) -> PyClawSpec:
        # https://www.clawpack.org/gallery/pyclaw/gallery/quadrants.html
        return PyClawSpec(
            riemann_solver=riemann.euler_4wave_2D,
            num_waves=4,
            kernel_language="Fortran",
            problem_data=self.parameters,
            classic_options={"transverse_waves": 2},
        )

    def to_conserved(self, rho, u, v, p):
        """`(rho, u, v, p)` → `[rho, rho u, rho v, E]` with `E = p/(γ−1) + ½ρ|v|²`."""
        E = 0.5 * rho * (u**2 + v**2) + p / (self.gamma - 1)
        return np.stack([rho, rho * u, rho * v, E], axis=0)

    def solve(
        self,
        ic_factory: ICFactory,
        x_spans: Sequence[Sequence[float]],
        Nxs: Sequence[int],
        t_span: tuple[float, float],
        Nt: int,
        bc: str | Sequence[Any] = "periodic",
        *,
        backend: SolverBackend | None = None,
        verbose: bool = False,
        use_rho_v_p_ics: bool = False,
        **kwargs: Any,
    ):
        """As the base class; with `use_rho_v_p_ics` the initial condition returns
        primitive `(rho, u, v, p)` and is converted with `to_conserved`."""
        if use_rho_v_p_ics:
            primitive = ic_factory

            def ic_factory(xs):  # noqa: F811
                return self.to_conserved(*primitive(xs))

        return super().solve(
            ic_factory,
            x_spans,
            Nxs,
            t_span,
            Nt,
            bc,
            backend=backend,
            verbose=verbose,
            **kwargs,
        )
