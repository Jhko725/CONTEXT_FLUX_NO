from typing import ClassVar

import equinox as eqx
import numpy as np

from context_flux_no.simulations.pde.base import AbstractHyperbolicConservationLaw
from context_flux_no.simulations.pyclaw import PyClawSpec


def riemann_sine_1D(q_l, q_r, aux_l, aux_r, problem_data):
    a = problem_data["a"]
    b = problem_data["b"]

    # flux: f(u) = a sin(bu)
    def flux(u):
        return a * np.sin(b * u)

    def dflux(u):
        return a * b * np.cos(b * u)

    # Instead of Roe speed, approximate the characteristic speed at the average position
    u_hat = 0.5 * (q_l[0, :] + q_r[0, :])
    s = dflux(u_hat)[None, :]

    wave = (q_r - q_l)[None, :, :]
    amdq = np.minimum(s, 0.0) * (q_r - q_l)
    apdq = np.maximum(s, 0.0) * (q_r - q_l)

    return wave, s, amdq, apdq


class SineFlux1D(AbstractHyperbolicConservationLaw):
    """`u_t + (a sin(b u))_x = 0`; Classic PyClaw with the characteristic-speed Python
    Riemann solver above and the MC limiter (the ICLR `sine_1d` setting)."""

    n_spatial_dims: ClassVar[int] = 1
    field_rank_names: ClassVar[tuple[tuple[int, str], ...]] = ((0, "u"),)
    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)

    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a = a
        self.b = b

    @property
    def parameters(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b}

    def flux(self, u):
        return self.a * np.sin(self.b * u)

    @property
    def pyclaw_spec(self) -> PyClawSpec:
        return PyClawSpec(
            riemann_solver=riemann_sine_1D,
            num_waves=1,
            kernel_language="Python",
            problem_data=self.parameters,
            classic_options={"limiters": "MC", "cfl_desired": 0.5, "cfl_max": 0.9},
        )
