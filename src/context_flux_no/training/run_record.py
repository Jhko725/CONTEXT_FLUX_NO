"""`RunRecord`: the typed `metadata.json` of a run directory (ADR-0010).

One object, written by the experiment script to three places — `metadata.json` beside the
checkpoints, orbax `custom_metadata`, and wandb (`config` = `index` only, the rest as a
saved file) — so that a run can be found from any of them. The **full resolved config** is
not duplicated here: it lives in the run directory's `config.yaml`, which is the latest
truth (a resume may change it). `index` is the flat set of columns that distinguish runs in
`training.runs.index()` and in the wandb table; keep it small on purpose (P10).

Legacy ICLR checkpoints get a record with `status="legacy"` from
`scripts/backfill_run_records.py`."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from context_flux_no.provenance import git_state, package_version


SCHEMA_VERSION = 1
Status = Literal["running", "finished", "failed", "legacy"]

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
    return None if target is None else str(target).rsplit(".", 1)[-1]


def index_from_config(cfg: Any, *, dataset_id: str | None = None) -> dict[str, Any]:
    """The index columns read off a resolved config (a plain dict or an OmegaConf
    `DictConfig`). Missing pieces become `None` rather than raising, because legacy
    configs lack some of them."""

    def get(*path: str, default: Any = None) -> Any:
        node: Any = cfg
        for key in path:
            try:
                node = node[key] if not hasattr(node, "get") else node.get(key)
            except (KeyError, TypeError, AttributeError):
                return default
            if node is None:
                return default
        return node

    def peak_lr_of(opt: Any) -> Any:
        """`learning_rate` of an optimizer node — a number, a schedule's `peak_value`, or
        the first one found inside an `optax.chain`'s `_args_`."""
        if opt is None or not hasattr(opt, "get"):
            return None
        lr = opt.get("learning_rate")
        if lr is not None:
            return lr.get("peak_value") if hasattr(lr, "get") else lr
        for part in opt.get("_args_") or []:
            found = peak_lr_of(part)
            if found is not None:
                return found
        return None

    peak_lr = peak_lr_of(get("training", "optimizer"))
    return {
        "model": _tail(get("model", "_target_")),
        "encoder": get("model", "encoder_type"),
        "target_network": get("model", "target_network_type"),
        # Phase 3 makes conditioning a config field; until then it is implied by the class
        "conditioning": {
            "HyperFluxFNOLocal": "hypernet",
            "HyperNeuralOperator": "hypernet",
            "HyperNeuralOperatorv2": "hypernet",
            "ContextAppendedFluxNO": "context-append",
        }.get(_tail(get("model", "_target_"))),
        "data": get("data", "name"),
        "dataset_id": dataset_id,
        "seed": get("model", "key", "seed"),
        "loss": _tail(get("loss_fn", "_target_")),
        "context_length": get("training", "context_length"),
        "batch_size": get("training", "batch_size"),
        "peak_lr": peak_lr,
        "max_steps": get("training", "max_steps"),
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

    def replace(self, **changes: Any) -> RunRecord:
        return dataclasses.replace(self, **changes)

    # ------------------------------------------------------------ (de)serialise --
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunRecord:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known}
        if unknown:  # forward compatibility: keep what we do not understand
            kwargs["extra"] = {**kwargs.get("extra", {}), "_unknown": unknown}
        return cls(**kwargs)

    @classmethod
    def from_json(cls, text: str) -> RunRecord:
        return cls.from_dict(json.loads(text))

    def write(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n")

    @classmethod
    def read(cls, path: Path) -> RunRecord:
        return cls.from_json(path.read_text())

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
