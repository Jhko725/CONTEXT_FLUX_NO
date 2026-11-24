"""Laws of `AbstractPDEFamily.positive_quantities` and `admissible`, and of `Euler2D.pressure`
(rule R11: properties and hand-built states, no second copy of a formula)."""

import jax.numpy as jnp
import numpy as np
from context_flux_no.simulations.pde import (
    CubicFlux1D,
    Euler2D,
    ParametrizedShallowWater1D,
)
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from jaxtyping import Float

from tests.helpers import PRESSURE_ROUND_TRIP_RTOL


_N_CELLS = 8


def _fields(
    low: float, high: float, shape: tuple[int, ...] = (_N_CELLS,)
) -> st.SearchStrategy[Float[np.ndarray, " *shape"]]:
    """Float64 fields of the given shape with every value in `[low, high]`."""
    return arrays(np.float64, shape, elements=st.floats(low, high))


@given(
    depth=_fields(0.1, 10.0),
    momentum=_fields(-10.0, 10.0),
    cell=st.integers(0, _N_CELLS - 1),
)
def test_shallow_water_one_negative_depth_is_not_admissible(
    depth: Float[np.ndarray, " cells"],
    momentum: Float[np.ndarray, " cells"],
    cell: int,
) -> None:
    pde = ParametrizedShallowWater1D()
    u = jnp.asarray(np.stack([depth, momentum]))
    assert bool(pde.admissible(u))
    assert bool(~pde.admissible(u.at[0, cell].set(-depth[cell])))


@given(
    rho=_fields(0.1, 10.0, (4, 3)),
    velocity=_fields(-3.0, 3.0, (2, 4, 3)),
    p=_fields(0.1, 10.0, (4, 3)),
    gamma=st.floats(1.1, 5.0 / 3.0),
)
def test_euler_pressure_inverts_to_conserved(
    rho: Float[np.ndarray, "x y"],
    velocity: Float[np.ndarray, "2 x y"],
    p: Float[np.ndarray, "x y"],
    gamma: float,
) -> None:
    pde = Euler2D(gamma)
    u = jnp.asarray(pde.to_conserved(rho, velocity[0], velocity[1], p))
    np.testing.assert_allclose(pde.pressure(u), p, rtol=PRESSURE_ROUND_TRIP_RTOL)
    assert bool(pde.admissible(u))


def test_euler_negative_density_or_pressure_is_not_admissible() -> None:
    """At rest, `E = p / (gamma - 1)` by hand, so `E = -1` is the pressure `-0.4`."""
    pde = Euler2D(1.4)
    at_rest = jnp.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [2.5, 2.5]])[..., None]
    assert bool(pde.admissible(at_rest))
    assert bool(~pde.admissible(at_rest.at[0, 1].set(-1.0)))
    assert bool(~pde.admissible(at_rest.at[3, 1].set(-1.0)))


@given(u=_fields(-1.0e6, 1.0e6, (1, _N_CELLS)))
def test_family_without_positive_quantities_admits_every_state(
    u: Float[np.ndarray, "1 cells"],
) -> None:
    pde = CubicFlux1D(1.0, 0.0, 0.0)
    assert pde.positive_quantities(jnp.asarray(u)).shape == (0, _N_CELLS)
    assert bool(pde.admissible(jnp.asarray(u)))
