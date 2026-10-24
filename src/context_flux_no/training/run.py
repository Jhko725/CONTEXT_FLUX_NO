"""`Run`: a training run as a directory (ADR-0010).

    <run_dir>/
        config.yaml      the resolved Hydra config — the latest truth (a resume rewrites it)
        metadata.json    a `RunRecord`: identity, provenance, index columns, outcome
        checkpoints/     an `OrbaxCheckpointer` directory of whole `TrainerState`s

The directory is Hydra's output directory (`hydra.run.dir`, `runs/<date>/<time>` by
default; a sweep adds `<job-num>/`), so Hydra's own `.hydra/` and the run live together and
nothing has to be joined later. `Run.from_config` **refuses** a directory that already holds
a run unless `resume=True`, in which case the record notes the step resumed from and the
config keys that changed. A warm start from *another* run is a new run whose record carries
`resumed_from`; the script handles that (`train_multiphysics.py`, `cfg.resume`).

Resume caveat (ledger, confirmed): restoring the `TrainerState` restores model, optimizer,
step and PRNG key exactly, but the **data stream restarts** — grain's
`.shuffle().repeat()` iterator is reseeded from the step and not restored. Exact
continuation through grain's iterator state is a follow-up (roadmap Phase 5, open).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from context_flux_no.training.checkpoint import OrbaxCheckpointer
from context_flux_no.training.run_record import index_from_config, RunRecord


CONFIG_FILE = "config.yaml"
METADATA_FILE = "metadata.json"
CHECKPOINT_DIR = "checkpoints"


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def config_changes(old: DictConfig, new: DictConfig) -> dict[str, dict[str, Any]]:
    """`{dotted.key: {"old": ..., "new": ...}}` for every leaf that differs; added or
    removed keys show the missing side as `None`."""
    a = _flatten(OmegaConf.to_container(old, resolve=True))
    b = _flatten(OmegaConf.to_container(new, resolve=True))
    return {
        k: {"old": a.get(k), "new": b.get(k)}
        for k in sorted(set(a) | set(b))
        if a.get(k) != b.get(k)
    }


class Run:
    """See the module docstring. Construct with `from_config` (new or resumed run) or
    `open` (an existing run, for analysis)."""

    def __init__(self, directory: str | Path, record: RunRecord, config: DictConfig):
        self.directory = Path(directory).resolve()
        self.record = record
        self.config = config

    # ------------------------------------------------------------ paths ----------
    @property
    def config_path(self) -> Path:
        return self.directory / CONFIG_FILE

    @property
    def metadata_path(self) -> Path:
        return self.directory / METADATA_FILE

    @property
    def checkpoint_dir(self) -> Path:
        return self.directory / CHECKPOINT_DIR

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / METADATA_FILE).exists()

    # ------------------------------------------------------------ construction ---
    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        directory: str | Path,
        *,
        dataset_id: str | None = None,
        x64: bool = False,
        metric_name: str | None = "valid_loss",
        resume: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> Run:
        """Start a run in `directory`, writing `config.yaml` and `metadata.json`. An
        existing run there raises `FileExistsError` unless `resume=True`, which keeps the
        record (and `run_id`), notes `resumed={"from_step", "config_changes"}` and rewrites
        `config.yaml` with `cfg`."""
        directory = Path(directory)
        if cls.exists(directory):
            if not resume:
                raise FileExistsError(
                    f"{directory} already holds a run (metadata.json); pass resume=True "
                    "to continue it or choose another directory."
                )
            previous = cls.open(directory)
            changes = config_changes(previous.config, cfg)
            record = previous.record.replace(
                status="running",
                resumed={
                    "from_step": previous.latest_step,
                    "config_changes": changes,
                    "previous_resumes": (previous.record.resumed or {}).get(
                        "previous_resumes", []
                    )
                    + ([previous.record.resumed] if previous.record.resumed else []),
                },
            )
        else:
            directory.mkdir(parents=True, exist_ok=True)
            model_config = OmegaConf.to_container(cfg.model, resolve=True)
            record = RunRecord.create(
                index=index_from_config(cfg, dataset_id=dataset_id),
                model_config=model_config,  # ty: ignore[invalid-argument-type]
                x64=x64,
                metric_name=metric_name,
                extra=extra,
            )
        run = cls(directory, record, cfg)
        OmegaConf.save(cfg, run.config_path, resolve=True)
        record.write(run.metadata_path)
        return run

    @classmethod
    def open(cls, directory: str | Path) -> Run:
        directory = Path(directory)
        if not cls.exists(directory):
            raise FileNotFoundError(f"{directory} holds no run (no {METADATA_FILE}).")
        record = RunRecord.read(directory / METADATA_FILE)
        config_path = directory / CONFIG_FILE
        config = (
            OmegaConf.load(config_path) if config_path.exists() else OmegaConf.create()
        )
        return cls(directory, record, config)  # ty: ignore[invalid-argument-type]

    # ------------------------------------------------------------ record ---------
    def update(self, **changes: Any) -> RunRecord:
        """Replace fields of the record and rewrite `metadata.json`."""
        self.record = self.record.replace(**changes)
        self.record.write(self.metadata_path)
        return self.record

    def finish(self, *, status: str = "finished") -> RunRecord:
        """Close the record: status, final step and the best checkpoint by the record's
        metric. Opens its own view of `checkpoints/`, so it works after the training
        checkpointer was closed."""
        changes: dict[str, Any] = {"status": status}
        if self.checkpoint_dir.exists():
            ckpt = OrbaxCheckpointer(
                self.checkpoint_dir, metric=self.record.metric_name
            )
            try:
                changes["final_step"] = ckpt.latest_step
                best = ckpt.best
            finally:
                ckpt.close()
            if best is not None:
                changes["best_step"], changes["best_metric"] = best
        return self.update(**changes)

    # ------------------------------------------------------------ checkpoints ----
    def checkpointer(self, **kwargs: Any) -> OrbaxCheckpointer:
        """An `OrbaxCheckpointer` on `checkpoints/` whose `custom_metadata` carries the
        record and the model config, so a checkpoint directory alone identifies its run
        (`runs.load_run` reads it back). `kwargs` go to `OrbaxCheckpointer`."""
        kwargs.setdefault("metric", self.record.metric_name)
        kwargs.setdefault(
            "custom_metadata",
            {
                "run_record": self.record.to_dict(),
                "model_config": self.record.model_config,
            },
        )
        return OrbaxCheckpointer(self.checkpoint_dir, **kwargs)

    @property
    def latest_step(self) -> int | None:
        if not self.checkpoint_dir.exists():
            return None
        ckpt = OrbaxCheckpointer(self.checkpoint_dir, metric=None)
        try:
            return ckpt.latest_step
        finally:
            ckpt.close()

    def restore(self, template: Any, step: int | None = None) -> Any:
        """The `TrainerState` at `step` (latest if `None`), shaped like `template`."""
        ckpt = OrbaxCheckpointer(self.checkpoint_dir, metric=None)
        try:
            return ckpt.restore(template, step)
        finally:
            ckpt.close()

    def __repr__(self) -> str:
        return f"Run({self.directory}, run_id={self.record.run_id!r}, status={self.record.status!r})"
