"""`EvaluationRecord`: the record of one evaluation, written beside its result file.

`scripts/evaluate.py` writes one HDF5 file of results and this record as JSON next to
it, so that a result names the runs, the test set and the code that produced it. The
record follows `runs.RunRecord`: a frozen dataclass with `create`, `write` and `read`.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from context_flux_no.utils.provenance import git_state, package_version
from context_flux_no.utils.serialization import dumps


SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class EvaluationRecord:
    """The runs, the test set, the arguments and the code of one evaluation.

    **Attributes:**

    - `run_ids`: The `RunRecord.run_id` of every evaluated run, in the order of the
        result file's groups.
    - `dataset_id`: The `dataset_id` of the source that the trajectories were read from.
    - `split`: The split of that source, such as `test`.
    - `arguments`: The arguments of the evaluation, as JSON values.
    - `created_at`: The UTC time of the evaluation, in ISO format.
    - `git`: `{"sha": str, "dirty": bool}` of the checkout the package was imported
        from, as `utils.provenance.git_state` gives it.
    - `package_version`: The installed version of the package.
    - `schema_version`: The version of this record's layout.
    """

    run_ids: list[str]
    dataset_id: str
    split: str
    arguments: dict[str, Any]
    created_at: str
    git: dict[str, Any]
    package_version: str
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        run_ids: list[str],
        dataset_id: str,
        split: str,
        arguments: dict[str, Any],
    ) -> EvaluationRecord:
        """Record an evaluation that runs now, with the provenance of this checkout.

        **Arguments:**

        - `run_ids`: The ids of the evaluated runs.
        - `dataset_id`: The `dataset_id` of the evaluated source.
        - `split`: Its split.
        - `arguments`: The arguments of the evaluation, as JSON values.

        **Returns:**

        The record, with `created_at`, `git` and `package_version` filled in.
        """
        sha, dirty = git_state()
        return cls(
            run_ids=list(run_ids),
            dataset_id=dataset_id,
            split=split,
            arguments=dict(arguments),
            created_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            git={"sha": sha, "dirty": dirty},
            package_version=package_version(),
        )

    def write(self, path: Path) -> None:
        """Write the record to `path` as JSON."""
        path.write_text(dumps(dataclasses.asdict(self)) + "\n")

    @classmethod
    def read(cls, path: Path) -> EvaluationRecord:
        """Read a record that `write` wrote.

        **Raises:**

        `TypeError` if the file holds a field that the record does not have, or lacks
        one that it needs.
        """
        return cls(**json.loads(path.read_text()))
