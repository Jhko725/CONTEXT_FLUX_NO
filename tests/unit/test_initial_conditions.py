"""`InitialCondition` laws: a sampled condition is a function of position — the same object
evaluated on two grids agrees where the grids coincide; the samplers' ranges hold; the
wrappers compose. (ADR-0011 §Initial conditions.)"""

import gstools as gs
import jax
import numpy as np
import pytest
from context_flux_no.simulations import Grid
from context_flux_no.simulations.initial_conditions import (
    Conserved,
    FromFunction,
    GaussianRandomField,
    Multichannel,
    PeriodicRandomStepFunction1D,
    PeriodicRandomStepFunction2D,
    RandomTruncatedFourier1D,
    Stacked,
    TruncatedFourier1D,
)
from context_flux_no.simulations.pde import Euler2D


SPANS_1D = [(0.0, 1.0)]
SPANS_2D = [(0.0, 1.0), (0.0, 1.0)]


def _samplers():
    return {
        "grf1d": GaussianRandomField([gs.Gaussian(dim=1, var=1.0, len_scale=0.2)]),
        "grf2d": GaussianRandomField(
            [gs.Gaussian(dim=2, len_scale=0.2)] * 2, means=[1.0, 0.0], scales=[0.2, 0.6]
        ),
        "step1d": PeriodicRandomStepFunction1D(2, 5, 0.5, 2.5),
        "step2d": PeriodicRandomStepFunction2D(
            3, 2, 6, [0.6, -0.8, 0.6], [1.4, 0.8, 1.4]
        ),
        "multi": Multichannel(
            [PeriodicRandomStepFunction1D(1, 3), PeriodicRandomStepFunction1D(1, 3)]
        ),
        "fourier": RandomTruncatedFourier1D(4, offset_range=(-0.5, 0.5)),
    }


@pytest.mark.parametrize("name", sorted(_samplers()))
@pytest.mark.parametrize("seed", [0, 7])
def test_sampled_condition_is_a_function_of_position(name, seed):
    sampler = _samplers()[name]
    spans = SPANS_1D if sampler.n_spatial_dims == 1 else SPANS_2D
    ic = sampler.sample(jax.random.key(seed), spans)
    assert ic.n_eqn == sampler.n_eqn and ic.n_spatial_dims == sampler.n_spatial_dims
    coarse, fine = Grid(spans, [8] * len(spans)), Grid(spans, [16] * len(spans))
    u_coarse = ic.at_points(coarse.meshgrid())
    u_fine = ic.at_points(fine.meshgrid())
    assert u_coarse.shape == (ic.n_eqn, *coarse.shape)
    # the coarse centers are the odd fine centers: lo + (2i + 1 + ½) dx_fine ... i.e. every
    # coarse center lies between two fine centers, so compare at the fine centers directly
    sub = ic.at_points(
        [c[::2] if c.ndim == 1 else c[::2, ::2] for c in fine.meshgrid()]
    )
    np.testing.assert_array_equal(
        sub, u_fine[(slice(None),) + (slice(None, None, 2),) * len(spans)]
    )
    # determinism: same key, same function
    again = sampler.sample(jax.random.key(seed), spans).at_points(coarse.meshgrid())
    np.testing.assert_array_equal(again, u_coarse)
    assert np.isfinite(u_fine).all()


def test_step_1d_values_and_periodicity():
    ic = PeriodicRandomStepFunction1D(2, 4, 0.5, 2.5).sample(
        jax.random.key(1), SPANS_1D
    )
    x = np.linspace(-1, 2, 301)
    u = ic.at_points([x])[0]
    assert set(np.unique(u)) <= set(ic.values) and (0.5 <= u).all() and (u <= 2.5).all()
    np.testing.assert_array_equal(ic.at_points([x])[0], ic.at_points([x + 1.0])[0])
    assert len(np.unique(u)) == len(
        ic.values
    )  # every interval is visited on a fine line


def test_step_2d_is_periodic_and_shares_geometry_across_channels():
    ic = PeriodicRandomStepFunction2D(2, 3, 3, [0.0, 10.0], [1.0, 11.0]).sample(
        jax.random.key(2), SPANS_2D
    )
    g = Grid(SPANS_2D, [12, 12])
    u = ic.at_points(g.meshgrid())
    shifted = ic.at_points([c + 1.0 for c in g.meshgrid()])
    np.testing.assert_array_equal(u, shifted)
    # same geometry on both channels: channel 1 is a function of channel 0
    assert len(np.unique(u[0])) == 3
    mapping = {a: b for a, b in zip(u[0].ravel(), u[1].ravel())}
    assert len(mapping) == 3
    assert all(mapping[a] == b for a, b in zip(u[0].ravel(), u[1].ravel()))


def test_grf_matches_gstools_structured_evaluation():
    sampler = _samplers()["grf2d"]
    ic = sampler.sample(jax.random.key(3), SPANS_2D)
    g = Grid(SPANS_2D, [8, 6])
    u = ic.at_points(g.meshgrid())
    for k, (srf, seed) in enumerate(zip(ic.srfs, ic.seeds)):
        structured = srf(tuple(g.centers), seed=seed, mesh_type="structured")
        np.testing.assert_allclose(
            u[k], structured * sampler.scales[k] + sampler.means[k]
        )


def test_grf_rejects_a_mismatched_period():
    with pytest.raises(ValueError):
        _samplers()["grf1d"].sample(jax.random.key(0), [(0.0, 2.0)])


def test_stacked_conserved_and_from_function():
    f = FromFunction(lambda c: np.stack([c[0], c[1]]), n_eqn=2, n_spatial_dims=2)
    g = FromFunction(lambda c: (c[0] * 0 + 1.0)[None], n_eqn=1, n_spatial_dims=2)
    grid = Grid(SPANS_2D, [4, 3])
    s = Stacked([f, g])
    assert s.n_eqn == 3
    np.testing.assert_array_equal(s.at_points(grid.meshgrid())[2], 1.0)
    pde = Euler2D(1.4)
    prim = FromFunction(
        lambda c: np.stack([1 + 0 * c[0], 0.1 + 0 * c[0], 0 * c[0], 1 + 0 * c[0]]),
        n_eqn=4,
        n_spatial_dims=2,
    )
    cons = Conserved(prim, pde).at_points(grid.meshgrid())
    np.testing.assert_allclose(cons[1], 0.1)  # rho u
    np.testing.assert_allclose(cons[3], 1 / 0.4 + 0.5 * 0.01)  # E
    with pytest.raises(ValueError):
        f.at_points([grid.centers[0]])  # wrong number of axes
    with pytest.raises(ValueError):
        FromFunction(lambda c: c[0], n_eqn=1, n_spatial_dims=1).at_points([np.zeros(3)])


def test_truncated_fourier_is_periodic_with_the_domain_length():
    ic = RandomTruncatedFourier1D(4).sample(jax.random.key(5), [(-1.0, 2.0)])
    assert isinstance(ic, TruncatedFourier1D) and ic.L == 3.0
    x = np.linspace(-1, 2, 50, endpoint=False)
    np.testing.assert_allclose(ic.at_points([x]), ic.at_points([x + 3.0]), atol=1e-12)
    assert not np.allclose(ic.at_points([x]), ic.at_points([x + 1.0]))
