"""The wandb side of a run (ADR-0010; pain point P10).

wandb is a **mirror** of the run directory, never its source of truth: the local
`metadata.json` carries wandb's id/url, and wandb's `config` is only `RunRecord.index` —
the dozen columns that distinguish runs — so the wandb table stays readable. The full
resolved config is uploaded as the file `config.yaml`, not as config keys. Grouping follows
the experiment: `cfg.experiment` if set, else the Hydra sweep directory for a multirun, else
the dataset name; `job_type` is the model class, `tags` the index values that are strings.

`cfg.wandb` comes from `scripts/training/configs/wandb/{disabled,offline,online}.yaml`;
`disabled` is the default, so a run leaves nothing but its directory unless asked."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from context_flux_no.runs.run import Run


INDEX_TAGS = ("model", "encoder", "target_network", "conditioning", "data", "loss")


def wandb_group(cfg: DictConfig) -> str | None:
    experiment = cfg.get("experiment")
    if experiment:
        return str(experiment)
    try:
        from hydra.core.hydra_config import HydraConfig

        hc = HydraConfig.get()
        if hc.mode is not None and hc.mode.name == "MULTIRUN":
            return Path(hc.sweep.dir).name
    except (ValueError, ImportError):  # not under @hydra.main
        pass
    data = cfg.get("data")
    return None if data is None else data.get("name")


def build_wandb_run(cfg: DictConfig, run: Run) -> Any | None:
    """Start the wandb run for `run` and stamp its id/url into the record; `None` when
    `cfg.wandb.mode` is `disabled` (or `cfg.wandb` is absent). The wandb files live under
    the run directory (`dir=`), so a run directory is self-contained."""
    wcfg = cfg.get("wandb")
    mode = "disabled" if wcfg is None else str(wcfg.get("mode", "disabled"))
    if mode == "disabled":
        return None
    import wandb

    record = run.record
    index = record.index
    tags = [str(index[k]) for k in INDEX_TAGS if index.get(k)]
    tags += [str(t) for t in (wcfg.get("tags") or [])]
    wrun = wandb.init(
        mode=mode,
        entity=wcfg.get("entity"),
        project=wcfg.get("project"),
        name=wcfg.get("name") or record.run_id,
        group=wcfg.get("group") or wandb_group(cfg),
        job_type=index.get("model"),
        tags=tags,
        dir=str(run.directory),
        config={**index, "run_id": record.run_id, "run_dir": str(run.directory)},
        resume="allow" if record.wandb else None,
        id=(record.wandb or {}).get("id"),
    )
    wrun.save(str(run.config_path), base_path=str(run.directory), policy="now")
    run.update(
        wandb={
            "id": wrun.id,
            "url": wrun.url,
            "entity": wrun.entity,
            "project": wrun.project,
            "mode": mode,
        }
    )
    return wrun


def finish_wandb_run(wrun: Any | None, run: Run, *, exit_code: int = 0) -> None:
    """Write the outcome to the wandb summary and finish. Safe on `None`."""
    if wrun is None:
        return
    record = run.record
    wrun.summary.update(
        {
            "best_step": record.best_step,
            "best_metric": record.best_metric,
            "final_step": record.final_step,
            "status": record.status,
            "run_dir": str(run.directory),
        }
    )
    wrun.save(str(run.metadata_path), base_path=str(run.directory), policy="now")
    wrun.finish(exit_code=exit_code)
