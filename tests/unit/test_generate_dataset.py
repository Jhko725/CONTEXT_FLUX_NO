"""`DatasetRecord` round trip, `WellWriter` layout (incl. The Well's `boundary_conditions`
group), and `generate_dataset` end to end: a tiny cubic-1D store through Classic PyClaw,
read back with `ZarrWellDatasetSource`, whose `dataset_id` then comes from the record."""

import json

import numpy as np
import pytest
import zarr
from context_flux_no.simulations import (
    DatasetRecord,
    generate_dataset,
    Grid,
    normalize_bcs,
    WellWriter,
)
from context_flux_no.simulations.pde import CubicFlux1D
from hypothesis import given, settings, strategies as st


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
    rec = _record(extra=extra).replace(n_failed=n_failed, status="complete")
    assert DatasetRecord.from_json(rec.to_json()) == rec
    assert rec.id.startswith("cubic_1d-train-")


def test_record_sidecar_path(tmp_path):
    assert (
        DatasetRecord.sidecar_path(tmp_path / "a_seed=0.zarr").name
        == "a_seed=0.dataset.json"
    )
    assert DatasetRecord.sidecar_path(tmp_path / "a.zarr.zip").name == "a.dataset.json"


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
        assert DatasetRecord.sidecar_path(path).exists()
        w.write_trajectory(0, pde.solution_to_well_schema(np.zeros((3, 1, 8))))
        w.finalize(1, n_failed=3)
        reopened = WellWriter.open(path)
        assert len(reopened) == 1 and reopened.record.status == "complete"
        assert reopened.record.n_failed == 3 and reopened.record.n_trajectories == 1
        assert reopened[0]["t0_fields"]["u"].shape == (3, 8)
        assert (
            json.loads(DatasetRecord.sidecar_path(path).read_text())["n_trajectories"]
            == 1
        )


def test_generate_dataset_end_to_end(tmp_path):
    pytest.importorskip("clawpack")
    from context_flux_no.data import ZarrWellDatasetSource

    def ic(centers, key):
        import jax

        phase = jax.random.uniform(key, ())
        return np.sin(2 * np.pi * centers[0] + float(phase))[None]

    writer = generate_dataset(
        dataset_name="cubic_1d",
        n_coeffs=2,
        n_ics_per_coeff=2,
        pde_factory=CubicFlux1D,
        initial_condition_fn=ic,
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
    assert rec.params["ranges"]["a"] == [-1.0, 1.0] and rec.ic == {"name": "sine-phase"}
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
