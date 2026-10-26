"""`DatasetRecord`: the provenance of one generated store (roadmap 4.3, ADR-0011).

The dataset analog of `runs.RunRecord`: written as `<store>.dataset.json` beside the zarr
store and into the store's root attributes, so a dataset can be identified from either. The
`id` is generated locally at creation; `WellDatasetSourceBase.dataset_id` derives the id it
stamps into `RunRecord.index` from the records of the files it reads."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from context_flux_no.utils.provenance import git_state, package_version
from context_flux_no.utils.serialization import dumps


SCHEMA_VERSION = 1
Status = Literal["writing", "complete", "failed", "legacy"]


def new_dataset_id(family: str, split: str) -> str:
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{family}-{split}-{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class DatasetRecord:
    id: str
    family: str  # the PDE class name
    dataset_name: str  # the Well dataset name (directory under the base path)
    split: str
    created_at: str
    status: Status
    n_trajectories: int  # intended at creation; written count after `finalize`
    grid: dict[str, Any]  # Grid.to_dict()
    time: dict[str, Any]  # {"t_span": [t0, t1], "Nt": int, "dt": float}
    bcs: list[list[str]]  # per axis [lower, upper]
    field_rank_names: list[list[Any]]  # [[rank, name], ...] in state order
    params: dict[
        str, Any
    ]  # {"names": [...], "ranges": {...}, "sampling": str, "n_coeffs": int}
    ic: dict[str, Any]  # {"name": str, "config": {...}}
    backend: dict[str, Any]  # SolverBackend.describe()
    seed: int
    n_failed: int = 0
    git: dict[str, Any] = field(default_factory=dict)
    package_version: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        family: str,
        dataset_name: str,
        split: str,
        n_trajectories: int,
        grid: dict[str, Any],
        time: dict[str, Any],
        bcs: Any,
        field_rank_names: Any,
        params: dict[str, Any],
        ic: dict[str, Any],
        backend: dict[str, Any],
        seed: int,
        extra: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> DatasetRecord:
        sha, dirty = git_state()
        return cls(
            id=id or new_dataset_id(dataset_name, split),
            family=family,
            dataset_name=dataset_name,
            split=split,
            created_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            status="writing",
            n_trajectories=int(n_trajectories),
            grid=dict(grid),
            time=dict(time),
            bcs=[list(pair) for pair in bcs],
            field_rank_names=[[int(r), str(n)] for r, n in field_rank_names],
            params=dict(params),
            ic=dict(ic),
            backend=dict(backend),
            seed=int(seed),
            git={"sha": sha, "dirty": dirty},
            package_version=package_version(),
            extra=dict(extra or {}),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetRecord:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known}
        if unknown:
            kwargs["extra"] = {**kwargs.get("extra", {}), "_unknown": unknown}
        return cls(**kwargs)

    def write(self, path: Path) -> None:
        path.write_text(dumps(dataclasses.asdict(self)) + "\n")

    @classmethod
    def read(cls, path: Path) -> DatasetRecord:
        return cls.from_dict(json.loads(path.read_text()))
