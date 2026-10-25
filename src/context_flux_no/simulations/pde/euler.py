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

from typing import ClassVar

import equinox as eqx
import numpy as np
from clawpack import riemann

from context_flux_no.simulations.backends.pyclaw import PyClawSpec
from context_flux_no.simulations.backends.trixi import TrixiSpec
from context_flux_no.simulations.pde.base import AbstractPDEFamily


_RHO_FLOOR = 1.0e-12
_PRESSURE_FLOOR = 1.0e-12
_SPEED_GAP_FLOOR = 1.0e-12


class Euler2D(AbstractPDEFamily):
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

    @property
    def trixi_spec(self) -> TrixiSpec:
        return TrixiSpec("CompressibleEulerEquations2D", self.parameters)  # Trixi's own

    def to_conserved(self, rho, u, v, p):
        """`(rho, u, v, p)` → `[rho, rho u, rho v, E]` with `E = p/(γ−1) + ½ρ|v|²`; wrap a
        primitive initial condition in `initial_conditions.Conserved(ic, pde)`."""
        E = 0.5 * rho * (u**2 + v**2) + p / (self.gamma - 1)
        return np.stack([rho, rho * u, rho * v, E], axis=0)
