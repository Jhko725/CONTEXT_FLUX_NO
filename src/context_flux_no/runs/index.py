"""Finding and loading trained runs (ADR-0010; pain point P10).

`index(root)` walks a runs root and returns one `pandas.DataFrame` row per run — the
`RunRecord.row()` columns, from `metadata.json` only (no checkpoint is opened), so it is
fast and works offline. `select` filters it with keyword equalities; `load_run` gives back
the trained model and its record for one run directory. Legacy ICLR checkpoint directories
(backfilled by `scripts/backfill_run_records.py`, `status == "legacy"`) load through
`training.io.load_model`.

`from_wandb` builds the same table from a wandb project, for runs trained elsewhere; the
local `metadata.json` is the primary source and wandb a mirror, never the other way round.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, TYPE_CHECKING

import equinox as eqx

from context_flux_no.runs.record import RunRecord
from context_flux_no.runs.run import DEFAULT_LAYOUT, Run, RunLayout
from context_flux_no.training.trainer import TrainerState


if TYPE_CHECKING:
    import pandas as pd


def iter_runs(
    root: str | Path, *, layout: RunLayout = DEFAULT_LAYOUT
) -> Iterator[Path]:
    """Every directory under `root` (inclusive) holding a run's metadata file, sorted."""
    root = Path(root)
    if (root / layout.metadata_file).exists():
        yield root
    for meta in sorted(root.rglob(layout.metadata_file)):
        if meta.parent != root:
            yield meta.parent


def index(root: str | Path, *, layout: RunLayout = DEFAULT_LAYOUT) -> pd.DataFrame:
    """One row per run under `root`; the `run_dir` column points back at the directory."""
    import pandas as pd

    rows = []
    for run_dir in iter_runs(root, layout=layout):
        try:
            record = RunRecord.read(run_dir / layout.metadata_file)
        except (
            ValueError,
            TypeError,
            KeyError,
        ) as e:  # unreadable record: still listed
            rows.append(
                {"run_id": None, "run_dir": str(run_dir), "status": f"unreadable: {e}"}
            )
            continue
        rows.append({**record.row(), "run_dir": str(run_dir)})
    df = pd.DataFrame(rows)
    if not df.empty and "created_at" in df:
        df = df.sort_values("created_at", na_position="first").reset_index(drop=True)
    return df


def select(table: pd.DataFrame, **where: Any) -> pd.DataFrame:
    """Rows where every `column == value` (a list/tuple/set value means `isin`)."""
    mask = None
    for col, val in where.items():
        m = (
            table[col].isin(list(val))
            if isinstance(val, (list, tuple, set, frozenset))
            else table[col] == val
        )
        mask = m if mask is None else (mask & m)
    return table if mask is None else table[mask]


def load_state(
    run_dir: str | Path,
    step: int | None = None,
    *,
    template: TrainerState | None = None,
) -> tuple[TrainerState, Run]:
    """The checkpointed `TrainerState` at `step` (latest if `None`) and its `Run`. The
    structure to restore into is `template`, or `experiment.state_template(run.config)`
    (the run's own config rebuilds model and optimizer) when not given."""
    run = Run.open(run_dir)
    if run.record.status == "legacy":
        raise ValueError(
            f"{run_dir} is a legacy ICLR checkpoint: it holds model weights only, "
            "use load_run (or training.io.load_model)."
        )
    if template is None:
        from context_flux_no.experiment import state_template  # config → objects

        template = state_template(run.config)
    return run.restore(template, step), run


def load_run(
    run_dir: str | Path, step: int | None = None
) -> tuple[eqx.Module, RunRecord]:
    """The trained model at `step` (latest if `None`) and the run's record. Legacy ICLR
    checkpoints (model weights only, model config in orbax `custom_metadata`) are loaded
    with `training.io.load_model`."""
    run = Run.open(run_dir)
    if run.record.status == "legacy":
        from context_flux_no.training.io import load_model

        ckpt_dir = run.directory / run.record.extra.get("checkpoint_dir", ".")
        return load_model(ckpt_dir, step), run.record
    state, run = load_state(run_dir, step)
    return state.model, run.record


def from_wandb(
    entity: str, project: str, *, filters: dict[str, Any] | None = None
) -> pd.DataFrame:
    """The index columns as logged to wandb (`run.config`) plus wandb id/url/state, one
    row per run of the project. Needs `wandb` and network access."""
    import pandas as pd
    import wandb

    api = wandb.Api()
    rows = []
    for r in api.runs(f"{entity}/{project}", filters=filters):
        cfg = {k: v for k, v in r.config.items() if not k.startswith("_")}
        rows.append(
            {
                **cfg,
                "wandb_id": r.id,
                "wandb_url": r.url,
                "wandb_state": r.state,
                "best_step": r.summary.get("best_step"),
                "best_metric": r.summary.get("best_metric"),
                "run_dir": r.summary.get("run_dir"),
            }
        )
    return pd.DataFrame(rows)
