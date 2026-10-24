"""`WellWriter`: a Well-schema zarr store written trajectory by trajectory (roadmap 4.3).

Backend-agnostic: it takes `[time, n_eqn, *N]` arrays from any `SolverBackend` and a PDE
class's `field_rank_names`/`solution_to_well_schema` for the field layout. Follows The Well's
HDF5 specification [cited: the_well/docs/data_format.md] — `dimensions/`, `scalars/`,
`t{0,1,2}_fields/` and `boundary_conditions/` (one subgroup per axis-side with `bc_type`,
`associated_dims`, `associated_fields`, `sample_varying`, `time_varying` and a boolean `mask`
over the associated dimension; both sides share one `*_periodic` group) — so that
`ZarrWellDatasetSource` and The Well's own reader agree on it. The store's root attrs carry
`dataset_record` (a `DatasetRecord`), which is also written as a sidecar
`<store>.dataset.json`.

Replaces `simulations/dataset.py::ZarrWellDataset` (same layout for everything it wrote; the
`boundary_conditions` group and the record are new)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr

from context_flux_no.simulations.backend import BCs
from context_flux_no.simulations.grid import Grid
from context_flux_no.simulations.record import DatasetRecord


WELL_BC_TYPE = {"periodic": "periodic", "wall": "wall", "extrap": "open"}


class WellWriter:
    """Create with `WellWriter.create(...)`, fill with `write_trajectory(idx, u, scalars)`,
    close with `finalize(n_written)`; `WellWriter.open(path)` reopens a store read-only for
    inspection (`record`, `__getitem__`, `__len__`)."""

    def __init__(self, root: zarr.Group, path: Path, record: DatasetRecord | None):
        self.root = root
        self.path = path
        self.record = record

    # ------------------------------------------------------------ construction ---
    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        pde: Any,
        grid: Grid,
        t_out: np.ndarray,
        bcs: BCs,
        record: DatasetRecord,
        dtype: Literal["float32", "float64"] = "float32",
        codec: Any | None = "default",
    ) -> WellWriter:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"a store already exists at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if codec == "default":
            codec = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")
        root = zarr.group(path)
        n_traj = record.n_trajectories
        n_dims = grid.n_spatial_dims
        Nt1 = len(t_out)

        root.attrs.update(
            {
                "dataset_name": record.dataset_name,
                "n_trajectories": n_traj,
                "grid_type": "cartesian",
                "n_spatial_dims": n_dims,
                "simulation_parameters": list(pde.parameters.keys()),
                "dataset_record": record.to_dict(),
            }
        )

        dims = root.create_group("dimensions")
        dims.attrs["spatial_dims"] = list(grid.axis_names)
        time_arr = dims.create_array(name="time", shape=(Nt1,), dtype=dtype)
        time_arr[:] = np.asarray(t_out)
        time_arr.attrs["sample_varying"] = False
        for name, centers in zip(grid.axis_names, grid.centers):
            arr = dims.create_array(name=name, shape=(len(centers),), dtype=dtype)
            arr[:] = centers
            arr.attrs["sample_varying"] = False
            arr.attrs["time_varying"] = False

        bc_group = root.create_group("boundary_conditions")
        for axis, (name, (lo, hi)) in enumerate(zip(grid.axis_names, bcs)):
            n = grid.Nxs[axis]
            sides = (
                [("periodic", (0, n - 1))]
                if lo == "periodic"
                else [
                    (f"{lo}_lower", (0,)),
                    (f"{hi}_upper", (n - 1,)),
                ]
            )
            for label, idx in sides:
                g = bc_group.create_group(f"{name}_{label}")
                g.attrs["associated_dims"] = [name]
                g.attrs["associated_fields"] = []
                g.attrs["bc_type"] = WELL_BC_TYPE[label.split("_")[0]]
                g.attrs["sample_varying"] = False
                g.attrs["time_varying"] = False
                mask = np.zeros(n, dtype=bool)
                mask[list(idx)] = True
                m = g.create_array(name="mask", shape=(n,), dtype="bool")
                m[:] = mask

        scalars = root.create_group("scalars")
        scalars.attrs["field_names"] = list(pde.parameters.keys())
        for s_name in pde.parameters:
            s = scalars.create_array(name=s_name, shape=(n_traj,), dtype=dtype)
            s.attrs["sample_varying"] = True
            s.attrs["time_varying"] = False

        by_rank: dict[int, list[str]] = {0: [], 1: [], 2: []}
        for r, f_name in pde.field_rank_names:
            by_rank[r].append(f_name)
        for r, names in by_rank.items():
            fields = root.create_group(f"t{r}_fields")
            fields.attrs["field_names"] = names
            shape = (n_traj, Nt1, *grid.shape) + ((n_dims**r,) if r else ())
            for f_name in names:
                arr = fields.create_array(
                    name=f_name,
                    shape=shape,
                    dtype=dtype,
                    chunks=(1,) + shape[1:],
                    shards="auto",
                    compressors=codec,
                )
                arr.attrs["dim_varying"] = [True] * n_dims
                arr.attrs["sample_varying"] = True
                arr.attrs["time_varying"] = True

        writer = cls(root, path, record)
        record.write(DatasetRecord.sidecar_path(path))
        return writer

    @classmethod
    def open(cls, path: str | Path, mode: str = "r") -> WellWriter:
        path = Path(path)
        root = zarr.open_group(path, mode=mode)
        rec = root.attrs.get("dataset_record")
        return cls(root, path, DatasetRecord.from_dict(dict(rec)) if rec else None)

    # ------------------------------------------------------------ data -----------
    @property
    def n_trajectories(self) -> int:
        return int(self.root.attrs["n_trajectories"])

    def __len__(self) -> int:
        return self.n_trajectories

    def write_trajectory(
        self, idx: int, well_schema: dict[str, dict[str, Any]]
    ) -> None:
        """`well_schema` as `pde.solution_to_well_schema(u)` returns it."""
        for grp_name in ("scalars", "t0_fields", "t1_fields", "t2_fields"):
            grp = self.root.get_group(grp_name)
            for f_name, value in well_schema.get(grp_name, {}).items():
                grp[f_name][idx] = value

    def __getitem__(self, idx: int) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {}
        for grp_name in ("scalars", "t0_fields", "t1_fields", "t2_fields"):
            grp = self.root.get_group(grp_name)
            out[grp_name] = {n: grp.get_array(n)[idx] for n in grp.attrs["field_names"]}
        return out

    def finalize(
        self, n_written: int, *, n_failed: int = 0, status: str = "complete"
    ) -> None:
        """Shrink the sample axis to what was written and close the record."""
        for grp_name in ("scalars", "t0_fields", "t1_fields", "t2_fields"):
            grp = self.root.get_group(grp_name)
            for f_name in grp.attrs["field_names"]:
                arr = grp.get_array(f_name)
                if arr.attrs["sample_varying"]:
                    arr.resize((n_written,) + arr.shape[1:])
        self.root.attrs["n_trajectories"] = int(n_written)
        if self.record is not None:
            self.record = self.record.replace(
                n_trajectories=int(n_written), n_failed=int(n_failed), status=status
            )
            self.root.attrs["dataset_record"] = self.record.to_dict()
            self.record.write(DatasetRecord.sidecar_path(self.path))
