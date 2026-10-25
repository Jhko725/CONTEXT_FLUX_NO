"""`TrixiBackend` without Julia: the specs exist, the configuration error is clear, and the
checks that need no Julia run before Julia is started."""

import numpy as np
import pytest
from context_flux_no.simulations import Grid, normalize_bcs, trixi
from context_flux_no.simulations.initial_conditions import FromFunction
from context_flux_no.simulations.pde import (
    Burgers1D,
    CubicFlux1D,
    CubicFlux2D,
    Euler2D,
    ParametrizedShallowWater1D,
    SineFlux1D,
    ViscousBurgers1D,
)


def test_six_families_have_a_trixi_spec():
    families = [
        CubicFlux1D(),
        CubicFlux2D(),
        SineFlux1D(),
        ParametrizedShallowWater1D(),
        Burgers1D(),
        Euler2D(),
    ]
    names = {f.trixi_spec.equations for f in families}
    assert len(names) == 6
    assert (
        ParametrizedShallowWater1D(h_floor=1e-6).trixi_spec.parameters["h_floor"]
        == 1e-6
    )
    with pytest.raises(NotImplementedError):
        ViscousBurgers1D().trixi_spec


def test_backend_options_and_checks_before_julia():
    with pytest.raises(ValueError):
        trixi.TrixiBackend(surface_flux="roe")
    with pytest.raises(ValueError):
        trixi.TrixiBackend(polydeg=0)
    backend = trixi.TrixiBackend()
    pde = CubicFlux1D()
    ic = FromFunction(lambda c: np.sin(c[0])[None], n_eqn=1, n_spatial_dims=1)
    grid = Grid([(0, 1)], [8])
    with pytest.raises(ValueError, match="periodic"):
        backend.solve(pde, ic, grid, np.array([0.0, 0.1]), normalize_bcs("extrap", 1))
    with pytest.raises(ValueError):
        backend.solve(
            pde,
            FromFunction(lambda c: np.stack([c[0], c[0]]), n_eqn=2, n_spatial_dims=1),
            grid,
            np.array([0.0, 0.1]),
            normalize_bcs("periodic", 1),
        )


@pytest.mark.skipif(
    not trixi.available(), reason="Julia/Trixi not set up (setup_julia.py)"
)
def test_describe_records_the_julia_side_versions():
    """`DatasetRecord.backend` pins the oracle: Trixi, Julia, threads and the package whose
    version carries the numerical constants (D-PROV)."""
    described = trixi.TrixiBackend().describe()
    assert {
        "trixi_version",
        "julia_version",
        "julia_threads",
        "julia_package_version",
    } <= described.keys()
    assert described["julia_package_version"].count(".") == 2


@pytest.mark.skipif(trixi.available(), reason="Julia is set up here")
def test_unavailable_is_a_clear_configuration_error():
    backend = trixi.TrixiBackend()
    ic = FromFunction(lambda c: np.sin(c[0])[None], n_eqn=1, n_spatial_dims=1)
    with pytest.raises(trixi.TrixiUnavailable, match="setup_julia|Julia project"):
        backend.solve(
            CubicFlux1D(),
            ic,
            Grid([(0, 1)], [8]),
            np.array([0.0, 0.1]),
            normalize_bcs("periodic", 1),
        )
    with pytest.raises(trixi.TrixiUnavailable):
        backend.describe()
