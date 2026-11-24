"""`DatasetRecord` round trip, `WellWriter` layout (incl. The Well's `boundary_conditions`
group), `store_channel_order` against a store written and read back, and `generate_dataset`
end to end: a tiny cubic-1D store through Classic PyClaw, read back with
`ZarrWellDatasetSource`, whose `dataset_id` then comes from the record."""

import copy
import dataclasses
import json

import numpy as np
import pytest
import zarr
from context_flux_no.data import ZarrWellDatasetSource
from context_flux_no.simulations import (
    DatasetRecord,
    generate_dataset,
    Grid,
    normalize_bcs,
    WellWriter,
)
from context_flux_no.simulations.initial_conditions import (
    FromFunction,
    InitialConditionSampler,
)
from context_flux_no.simulations.pde import AbstractPDEFamily, CubicFlux1D, Euler2D
from context_flux_no.simulations.writer import sidecar_path, store_channel_order
from context_flux_no.utils.serialization import dumps
from hypothesis import given, settings, strategies as st
from jaxtyping import Float


_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(2**40), 2**40),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=12),
)
_flat = st.dictionaries(st.text(min_size=1, max_size=8), _scalar, max_size=5)


def _record(**over):
    base = dict(
        family="CubicFlux1D",
        dataset_name="cubic_1d",
        split="train",
        n_trajectories=4,
        grid=Grid(((0.0, 1.0),), (8,)).to_dict(),
        time={"t_span": [0.0, 0.1], "Nt": 2, "dt": 0.05},
        bcs=normalize_bcs("periodic", 1),
        field_rank_names=((0, "u"),),
        params={
            "names": ["a"],
            "ranges": {"a": [0.0, 1.0]},
            "sampling": "uniform",
            "n_coeffs": 2,
        },
        ic={"name": "grf"},
        backend={"name": "pyclaw", "scheme": "classic"},
        seed=0,
    )
    base.update(over)
    return DatasetRecord.create(**base)


@settings(max_examples=30, deadline=None)
@given(extra=_flat, n_failed=st.integers(0, 100))
def test_record_round_trip(extra, n_failed):
    rec = copy.replace(_record(extra=extra), n_failed=n_failed, status="complete")
    assert DatasetRecord.from_dict(json.loads(dumps(dataclasses.asdict(rec)))) == rec
    assert rec.id.startswith("cubic_1d-train-")


def test_record_sidecar_path(tmp_path):
    assert sidecar_path(tmp_path / "a_seed=0.zarr").name == "a_seed=0.dataset.json"
    assert sidecar_path(tmp_path / "a.zarr.zip").name == "a.dataset.json"


def test_writer_layout_and_boundary_conditions(tmp_path):
    pde = CubicFlux1D(1.0, 0.0, 0.0)
    grid = Grid(((0.0, 1.0),), (8,))
    t_out = np.linspace(0, 0.1, 3)
    for bc, groups in [
        ("periodic", {"x_periodic": ("periodic", [True] + [False] * 6 + [True])}),
        (
            [("extrap", "wall")],
            {
                "x_extrap_lower": ("open", [True] + [False] * 7),
                "x_wall_upper": ("wall", [False] * 7 + [True]),
            },
        ),
    ]:
        bcs = normalize_bcs(bc, 1)
        path = tmp_path / f"{'p' if bc == 'periodic' else 'ew'}.zarr"
        w = WellWriter.create(
            path, pde=pde, grid=grid, t_out=t_out, bcs=bcs, record=_record(bcs=bcs)
        )
        root = zarr.open_group(path, mode="r")
        assert set(root["boundary_conditions"].keys()) == set(groups)
        for name, (bc_type, mask) in groups.items():
            g = root["boundary_conditions"][name]
            assert g.attrs["bc_type"] == bc_type and g.attrs["associated_dims"] == ["x"]
            assert list(g["mask"][:]) == mask
        assert root.attrs["simulation_parameters"] == ["a", "b", "c"]
        assert root["t0_fields"]["u"].shape == (4, 3, 8)
        assert sidecar_path(path).exists()
        w.write_trajectory(0, np.zeros((3, 1, 8)), pde.parameters)
        w.finalize(1, n_failed=3)
        root = zarr.open_group(path, mode="r")
        rec = DatasetRecord.read(sidecar_path(path))
        assert root.attrs["n_trajectories"] == 1 and root["t0_fields"]["u"].shape == (
            1,
            3,
            8,
        )
        assert (
            rec.status == "complete" and rec.n_failed == 3 and rec.n_trajectories == 1
        )
        assert DatasetRecord.from_dict(dict(root.attrs["dataset_record"])) == rec


def _write_and_read_back(
    tmp_path, pde: AbstractPDEFamily, grid: Grid
) -> tuple[
    Float[np.ndarray, "2 channels *shape"], Float[np.ndarray, "2 channels *shape"]
]:
    """Write one two-step trajectory whose channel `c` is the constant `c + 1` with
    `WellWriter`, and read it back through `ZarrWellDatasetSource`.

    **Returns:**

    The trajectory as written, in the family's order, and as read, in the store's order.
    """
    t_out = np.linspace(0.0, 0.1, 2)
    bcs = normalize_bcs("periodic", grid.n_spatial_dims)
    record = _record(
        family=type(pde).__name__,
        dataset_name="order",
        n_trajectories=1,
        grid=grid.to_dict(),
        bcs=bcs,
        field_rank_names=pde.field_rank_names,
    )
    path = tmp_path / "order" / "data" / "train" / "order.zarr"
    writer = WellWriter.create(
        path, pde=pde, grid=grid, t_out=t_out, bcs=bcs, record=record
    )
    channels = np.arange(1.0, pde.n_eqns + 1.0).reshape(1, -1, *[1] * len(grid.shape))
    u_family = np.broadcast_to(channels, (2, pde.n_eqns, *grid.shape))
    writer.write_trajectory(0, u_family, pde.parameters)
    writer.finalize(1)
    source = ZarrWellDatasetSource(
        well_base_path=tmp_path, well_dataset_name="order", window_size=2
    )
    return u_family, np.asarray(source[0])


def test_store_channel_order_of_euler_2d_matches_the_store(tmp_path):
    pde = Euler2D()
    u_family, u_store = _write_and_read_back(
        tmp_path, pde, Grid(((0, 1), (0, 1)), (4, 3))
    )
    # Measured: the store holds (rho, E, m_x, m_y) against the family's (rho, m_x, m_y, E).
    assert list(u_store[0, :, 0, 0]) == [1.0, 4.0, 2.0, 3.0]
    order = store_channel_order(pde.field_rank_names, pde.n_spatial_dims)
    assert order == (0, 3, 1, 2)
    np.testing.assert_array_equal(u_store, u_family[:, list(order)])
    np.testing.assert_array_equal(u_store[:, np.argsort(order)], u_family)


def test_store_channel_order_of_a_scalar_family_is_the_identity(tmp_path):
    pde = CubicFlux1D(1.0, 0.0, 0.0)
    u_family, u_store = _write_and_read_back(tmp_path, pde, Grid(((0, 1),), (8,)))
    assert store_channel_order(pde.field_rank_names, pde.n_spatial_dims) == (0,)
    np.testing.assert_array_equal(u_store, u_family)


def test_generate_dataset_end_to_end(tmp_path):
    pytest.importorskip("clawpack")

    class SinePhase(InitialConditionSampler):
        def __init__(self):
            super().__init__(n_eqn=1, n_spatial_dims=1)

        def sample(self, key, x_spans):
            import jax

            phase = float(jax.random.uniform(key, ()))
            return FromFunction(
                lambda c: np.sin(2 * np.pi * c[0] + phase)[None],
                n_eqn=1,
                n_spatial_dims=1,
            )

    writer = generate_dataset(
        dataset_name="cubic_1d",
        n_coeffs=2,
        n_ics_per_coeff=2,
        pde_factory=CubicFlux1D,
        initial_condition=SinePhase(),
        coeff_range_dict={"a": (-1.0, 1.0), "b": (-1.0, 1.0), "c": (-1.0, 1.0)},
        x_spans=[(0, 1)],
        Nxs=[32],
        t_span=(0, 0.1),
        Nt=4,
        dataset_type="train",
        savedir=tmp_path,
        seed=3,
        ic_description={"name": "sine-phase"},
    )
    rec = writer.record
    assert rec.status == "complete" and rec.n_trajectories == 4 and rec.n_failed == 0
    assert rec.backend["scheme"] == "classic" and rec.family == "CubicFlux1D"
    assert rec.params["ranges"]["a"] == [-1.0, 1.0]
    assert rec.ic == {"name": "sine-phase", "primitive": False}
    assert (
        tmp_path / "cubic_1d" / "data" / "train" / "cubic_1d_seed=3.dataset.json"
    ).exists()

    source = ZarrWellDatasetSource(
        well_base_path=tmp_path,
        well_dataset_name="cubic_1d",
        well_split_name="train",
        window_size=3,
    )
    assert source.dataset_record_ids == [rec.id]
    assert source.dataset_id.startswith("cubic_1d:train:")
    item = source[0]
    assert tuple(item.shape) == (3, 1, 32)  # time, channel, x
    assert source.metadata_common["n_spatial_dims"] == 1
