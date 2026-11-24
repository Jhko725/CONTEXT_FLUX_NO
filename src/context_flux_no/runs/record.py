"""`RunRecord`: the typed `metadata.json` of a run directory (ADR-0010).

One object, written by the experiment script to three places — `metadata.json` beside the
checkpoints, orbax `custom_metadata`, and wandb (`config` = `index` only, the rest as a
saved file) — so that a run can be found from any of them. The **full resolved config** is
not duplicated here: it lives in the run directory's `config.yaml`, which is the latest
truth (a resume may change it). `index` is the flat set of columns that distinguish runs in
`training.runs.index()` and in the wandb table; keep it small on purpose (P10)."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, Literal

from context_flux_no.utils.provenance import git_state, package_version
from context_flux_no.utils.serialization import dumps


SCHEMA_VERSION = 1
Status = Literal["running", "finished", "failed"]

INDEX_COLUMNS = (
    "model",
    "encoder",
    "target_network",
    "conditioning",
    "data",
    "dataset_id",
    "seed",
    "loss",
    "context_length",
    "batch_size",
    "peak_lr",
    "max_steps",
)


def new_run_id() -> str:
    """A locally generated id, so a run has an identity even with wandb off. Sortable by
    creation time (UTC timestamp prefix) and unique (uuid4 suffix)."""
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _tail(target: str | None) -> str | None:
    """The last component of the dotted name `target`, or `None` for `None`."""
    return None if target is None else str(target).rsplit(".", 1)[-1]


def _config_value(cfg: Any, *path: str) -> Any:
    """The node at `path` in a nested config, or `None` when a key along it is missing
    or a node along it is `None`."""
    node: Any = cfg
    for key in path:
        try:
            node = node[key] if not hasattr(node, "get") else node.get(key)
        except (KeyError, TypeError, AttributeError):
            return None
        if node is None:
            return None
    return node


def _peak_learning_rate(opt: Any) -> Any:
    """`learning_rate` of an optimizer node — a number, a schedule's `peak_value`, or
    the first one found inside an `optax.chain`'s `_args_`."""
    if opt is None or not hasattr(opt, "get"):
        return None
    lr = opt.get("learning_rate")
    if lr is not None:
        return lr.get("peak_value") if hasattr(lr, "get") else lr
    for part in opt.get("_args_") or []:
        found = _peak_learning_rate(part)
        if found is not None:
            return found
    return None


def index_from_config(cfg: Any, *, dataset_id: str | None = None) -> dict[str, Any]:
    """The index columns read off a resolved config (a plain dict or an OmegaConf
    `DictConfig`). Missing pieces become `None` rather than raising, because older
    configs lack some of them."""
    peak_lr = _peak_learning_rate(_config_value(cfg, "training", "optimizer"))
    return {
        "model": _tail(_config_value(cfg, "model", "_target_")),
        "encoder": _config_value(cfg, "model", "encoder_type"),
        "target_network": _config_value(cfg, "model", "target_network_type"),
        # Phase 3 makes conditioning a config field; until then it is implied by the class
        "conditioning": {
            "HyperFluxFNOLocal": "hypernet",
            "HyperNeuralOperator": "hypernet",
            "HyperNeuralOperatorv2": "hypernet",
            "ContextAppendedFluxNO": "context-append",
        }.get(_tail(_config_value(cfg, "model", "_target_"))),
        "data": _config_value(cfg, "data", "name"),
        "dataset_id": dataset_id,
        "seed": _config_value(cfg, "model", "key", "seed"),
        "loss": _tail(_config_value(cfg, "loss_fn", "_target_")),
        "context_length": _config_value(cfg, "training", "context_length"),
        "batch_size": _config_value(cfg, "training", "batch_size"),
        "peak_lr": peak_lr,
        "max_steps": _config_value(cfg, "training", "max_steps"),
    }


@dataclass(frozen=True)
class RunRecord:
    """See the module docstring. Construct with `RunRecord.create(...)` for a new run."""

    run_id: str
    created_at: str
    status: Status
    index: dict[str, Any]
    model_config: dict[str, Any]
    git: dict[str, Any] = field(default_factory=dict)  # {"sha": str, "dirty": bool}
    package_version: str = ""
    x64: bool = False
    wandb: dict[str, Any] | None = None  # {"id", "url", "entity", "project", "mode"}
    metric_name: str | None = None
    best_step: int | None = None
    best_metric: float | None = None
    final_step: int | None = None
    resumed: dict[str, Any] | None = (
        None  # in-place continuation: from_step, config_changes
    )
    resumed_from: dict[str, Any] | None = None  # warm start: run dir, step
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------ construction ---
    @classmethod
    def create(
        cls,
        *,
        index: dict[str, Any],
        model_config: dict[str, Any],
        x64: bool = False,
        metric_name: str | None = "valid_loss",
        extra: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunRecord:
        """A record of a new run: a new `run_id` unless one is given, the creation time,
        status `"running"`, and the current git state and package version."""
        sha, dirty = git_state()
        return cls(
            run_id=run_id or new_run_id(),
            created_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            status="running",
            index=dict(index),
            model_config=dict(model_config),
            git={"sha": sha, "dirty": dirty},
            package_version=package_version(),
            x64=x64,
            metric_name=metric_name,
            extra=dict(extra or {}),
        )

    # ------------------------------------------------------------ (de)serialize --
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        """The record of a dictionary as `write` stores it. Keys that are not fields are
        kept under `extra["_unknown"]`, so a newer record still reads; a status
        outside `Status` raises `ValueError`."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known}
        if kwargs.get("status") not in get_args(Status):
            raise ValueError(
                f"unknown run status {kwargs.get('status')!r}; "
                f"expected one of {get_args(Status)}."
            )
        if unknown:  # forward compatibility: keep what we do not understand
            kwargs["extra"] = {**kwargs.get("extra", {}), "_unknown": unknown}
        return cls(**kwargs)

    def write(self, path: Path) -> None:
        """Write the record as JSON to `path`."""
        path.write_text(dumps(dataclasses.asdict(self)) + "\n")

    @classmethod
    def read(cls, path: Path) -> RunRecord:
        """Read a record that `write` wrote to `path`."""
        return cls.from_dict(json.loads(path.read_text()))

    # ------------------------------------------------------------ views ----------
    def row(self) -> dict[str, Any]:
        """One flat row for `runs.index()`: the index columns plus status and outcome."""
        return {
            "run_id": self.run_id,
            **{k: self.index.get(k) for k in INDEX_COLUMNS},
            "status": self.status,
            "best_step": self.best_step,
            "best_metric": self.best_metric,
            "final_step": self.final_step,
            "created_at": self.created_at,
            "wandb_id": (self.wandb or {}).get("id"),
            "git_sha": (self.git or {}).get("sha", "")[:8],
            "git_dirty": (self.git or {}).get("dirty"),
        }
