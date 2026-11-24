"""`WellWriter`: a Well-schema zarr store written trajectory by trajectory (roadmap 4.3).

Backend-agnostic: it takes `[time, n_eqn, *N]` arrays from any `SolverBackend` and splits them
into the Well's rank groups with the `field_rank_names` the store was created with. Follows The Well's
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

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from itertools import accumulate
from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
import zarr.codecs
from jaxtyping import Float
from zarr.core.common import JSON

from context_flux_no.simulations.backends.base import BCs
from context_flux_no.simulations.grid import Grid
from context_flux_no.simulations.record import DatasetRecord, Status


def sidecar_path(store_path: str | Path) -> Path:
    """`foo.zarr` (or `foo.zarr.zip`) → `foo.dataset.json`, the `DatasetRecord` beside the store."""
    p = Path(store_path)
    name = p.name
    for suffix in (".zarr.zip", ".zarr"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return p.with_name(name + ".dataset.json")


def store_channel_order(
    field_rank_names: tuple[tuple[int, str], ...], n_spatial_dims: int
) -> tuple[int, ...]:
    """Give the family channel that each channel of a store holds.

    `WellWriter.write_trajectory` writes the rank-`r` field of each `(rank, name)` pair to
    `t{r}_fields/<name>`, and `data.WellDatasetSourceBase` packs the groups back rank 0
    first, keeping the order of the names within each group. A family whose fields are not
    sorted by rank therefore reads back in another order: `Euler2D`'s `(rho, m_x, m_y, E)`
    becomes `(rho, E, m_x, m_y)`.

    **Arguments:**

    - `field_rank_names`: The `(rank, name)` pairs in the family's state order, as
        `AbstractPDEFamily.field_rank_names` gives them.
    - `n_spatial_dims`: The number of spatial dimensions; a rank-`r` field has
        `n_spatial_dims**r` components.

    **Returns:**

    The permutation `p` with `u_store[i] == u_family[p[i]]`, so `u_family[p]` is the
    state in the store's order and `u_store[argsort(p)]` the state in the family's order.
    """
    starts = list(
        accumulate((n_spatial_dims**rank for rank, _ in field_rank_names), initial=0)
    )
    blocks = sorted(
        zip(field_rank_names, starts),
        key=lambda pair_and_start: pair_and_start[0][0],
    )
    return tuple(
        start + component
        for (rank, _), start in blocks
        for component in range(n_spatial_dims**rank)
    )


WELL_BC_TYPE = {"periodic": "periodic", "wall": "wall", "extrap": "open"}


def _write_dimensions(
    root: zarr.Group,
    grid: Grid,
    t_out: Float[np.ndarray, " time"],
    dtype: Literal["float32", "float64"],
) -> None:
    """Write the `dimensions/` group: the output times and the cell centers of each axis."""
    dims = root.create_group("dimensions")
    dims.attrs["spatial_dims"] = list(grid.axis_names)
    time_arr = dims.create_array(name="time", shape=(len(t_out),), dtype=dtype)
    time_arr[:] = np.asarray(t_out)
    time_arr.attrs["sample_varying"] = False
    for name, centers in zip(grid.axis_names, grid.centers):
        arr = dims.create_array(name=name, shape=(len(centers),), dtype=dtype)
        arr[:] = centers
        arr.attrs["sample_varying"] = False
        arr.attrs["time_varying"] = False


def _write_boundary_conditions(root: zarr.Group, grid: Grid, bcs: BCs) -> None:
    """Write the `boundary_conditions/` group: one subgroup per axis side, or one
    `*_periodic` subgroup for both sides of a periodic axis, each with its boundary mask."""
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


def _create_field_arrays(
    root: zarr.Group,
    pde: Any,
    grid: Grid,
    n_traj: int,
    Nt1: int,
    dtype: Literal["float32", "float64"],
    codec: Any | None,
) -> None:
    """Create the empty `t0_fields/`, `t1_fields/` and `t2_fields/` arrays, one chunk per
    trajectory, for the fields of each rank of `pde.field_rank_names`."""
    n_dims = grid.n_spatial_dims
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


def _int_attr(attrs: Mapping[str, JSON], key: str) -> int:
    """Read the integer attribute `key`; raise `TypeError` when the store holds another
    kind of value there."""
    value = attrs[key]
    if not isinstance(value, int):
        raise TypeError(f"store attribute {key!r} is {value!r}, not an integer")
    return value


def _sequence_attr(attrs: Mapping[str, JSON], key: str) -> Sequence[JSON]:
    """Read the list attribute `key`; raise `TypeError` when the store holds another
    kind of value there."""
    value = attrs[key]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"store attribute {key!r} is {value!r}, not a list")
    return value


def _names_attr(attrs: Mapping[str, JSON], key: str) -> list[str]:
    """Read the attribute `key` as a list of names; raise `TypeError` when an entry is
    not a string."""
    entries = _sequence_attr(attrs, key)
    names = [entry for entry in entries if isinstance(entry, str)]
    if len(names) != len(entries):
        raise TypeError(f"store attribute {key!r} is {entries!r}, not a list of names")
    return names


def _field_rank_names_attr(attrs: Mapping[str, JSON]) -> tuple[tuple[int, str], ...]:
    """Read the `field_rank_names` attribute as `(rank, name)` pairs; raise `TypeError`
    when an entry is not a pair of an integer and a string."""
    pairs = []
    for entry in _sequence_attr(attrs, "field_rank_names"):
        match entry:
            case [int() as rank, str() as name]:
                pairs.append((rank, name))
            case _:
                raise TypeError(
                    f"store attribute 'field_rank_names' has the entry {entry!r},"
                    " not a (rank, name) pair"
                )
    return tuple(pairs)


class WellWriter:
    """Create with `WellWriter.create(...)`, fill with `write_trajectory(idx, u, scalars)`,
    close with `finalize(n_written)`. Reading a store back is `data.ZarrWellDatasetSource`'s
    job; the record is read from the sidecar (`DatasetRecord.read(sidecar_path(store))`)."""

    def __init__(self, root: zarr.Group, path: Path, record: DatasetRecord):
        self.root = root
        self.path = path
        self.record = record
        self.field_rank_names: tuple[tuple[int, str], ...] = _field_rank_names_attr(
            root.attrs
        )

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
        """Create an empty store at `path` for `record.n_trajectories` trajectories of
        `pde` on `grid`, and write the record's sidecar.

        **Arguments:**

        - `path`: Where the store goes; raises `FileExistsError` if it exists.
        - `pde`: The family, which gives the field layout and the parameter names.
        - `grid`, `t_out`, `bcs`: The grid, the output times and the boundary conditions of every trajectory.
        - `record`: The dataset's record, stored in the root attributes and the sidecar.
        - `dtype`: The dtype of every stored array.
        - `codec`: The compressor of the field arrays; `"default"` is Blosc with zstd at level 3 and bit shuffling, and `None` stores them uncompressed.

        **Returns:**

        The writer of the new store.
        """
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
                "field_rank_names": [[int(r), str(n)] for r, n in pde.field_rank_names],
                "dataset_record": dataclasses.asdict(record),
            }
        )

        _write_dimensions(root, grid, t_out, dtype)
        _write_boundary_conditions(root, grid, bcs)

        scalars = root.create_group("scalars")
        scalars.attrs["field_names"] = list(pde.parameters.keys())
        for s_name in pde.parameters:
            s = scalars.create_array(name=s_name, shape=(n_traj,), dtype=dtype)
            s.attrs["sample_varying"] = True
            s.attrs["time_varying"] = False

        _create_field_arrays(root, pde, grid, n_traj, Nt1, dtype, codec)

        writer = cls(root, path, record)
        record.write(sidecar_path(path))
        return writer

    # ------------------------------------------------------------ data -----------
    @property
    def n_trajectories(self) -> int:
        """The number of trajectories the store was created for."""
        return _int_attr(self.root.attrs, "n_trajectories")

    def write_trajectory(
        self, idx: int, u: np.ndarray, scalars: Mapping[str, float]
    ) -> None:
        """`u[time, n_eqn, *N]` in state order and the family's `parameters` for one sample.
        Component block `r` of rank `r` goes to `t{r}_fields/<name>` as `[time, *N]` (rank 0)
        or `[time, *N, n_spatial_dims**r]`."""
        n_dims = _int_attr(self.root.attrs, "n_spatial_dims")
        start = 0
        for rank, name in self.field_rank_names:
            stop = start + n_dims**rank
            block = np.moveaxis(u[:, start:stop], 1, -1)  # [time, *N, components]
            if rank == 0:
                block = block[..., 0]
            self.root.get_group(f"t{rank}_fields").get_array(name)[idx] = block
            start = stop
        if stop != u.shape[1]:
            raise ValueError(
                f"u has {u.shape[1]} components but the layout {self.field_rank_names} "
                f"covers {stop}"
            )
        scalars_grp = self.root.get_group("scalars")
        for name in _names_attr(scalars_grp.attrs, "field_names"):
            scalars_grp.get_array(name)[idx] = scalars[name]

    def finalize(
        self, n_written: int, *, n_failed: int = 0, status: Status = "complete"
    ) -> None:
        """Shrink the sample axis to what was written and close the record."""
        for grp_name in ("scalars", "t0_fields", "t1_fields", "t2_fields"):
            grp = self.root.get_group(grp_name)
            for f_name in _names_attr(grp.attrs, "field_names"):
                arr = grp.get_array(f_name)
                if arr.attrs["sample_varying"]:
                    arr.resize((n_written,) + arr.shape[1:])
        self.root.attrs["n_trajectories"] = int(n_written)
        self.record = copy.replace(
            self.record,
            n_trajectories=int(n_written),
            n_failed=int(n_failed),
            status=status,
        )
        self.root.attrs["dataset_record"] = dataclasses.asdict(self.record)
        self.record.write(sidecar_path(self.path))
