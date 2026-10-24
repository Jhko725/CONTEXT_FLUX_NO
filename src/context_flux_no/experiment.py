"""What the training scripts (`scripts/training/`) share: assembling a `Run`, its loggers and checkpointer around
the `Trainer`, and the resume / warm-start handling of `cfg.resume` / `cfg.warm_start`
(ADR-0010). The scripts keep only what differs between them — the data source and the
loaders."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import equinox as eqx
import hydra
import jax
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from context_flux_no.runs import Run
from context_flux_no.runs.wandb_mirror import build_wandb_run, finish_wandb_run
from context_flux_no.training import (
    DelayedLogger,
    MultiLogger,
    PrintLogger,
    Trainer,
    TrainerState,
    WandbLogger,
)
from context_flux_no.training.loss import PushforwardOneStepLoss


log = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> eqx.Module:
    return hydra.utils.instantiate(cfg.model)


def build_optimizer(cfg: DictConfig):
    return hydra.utils.instantiate(cfg.training.optimizer)


def state_template(cfg: DictConfig) -> TrainerState:
    """A `TrainerState` with the structure a run checkpointed, built from its config: the
    model and optimizer from `cfg.model` / `cfg.training.optimizer` (the optimizer **must**
    be fully described by the config — chains included — since the opt-state structure has
    to match). The PRNG key is a placeholder overwritten on restore. Used by
    `runs.load_state` / `runs.load_run`."""
    return TrainerState.init(
        build_model(cfg), build_optimizer(cfg), key=jax.random.key(0)
    )


def select_device(cfg: DictConfig) -> None:
    if cfg.gpu_id != "auto":
        jax.config.update("jax_default_device", jax.devices("gpu")[cfg.gpu_id])


def segment_length(cfg: DictConfig, loss_fn: Any) -> int:
    """Window length the loaders must produce: context + target (+1 for pushforward)."""
    extra = 2 if isinstance(loss_fn, PushforwardOneStepLoss) else 1
    return cfg.training.context_length + extra


def run_directory(cfg: DictConfig) -> Path:
    """Hydra's output directory, or `cfg.resume` when continuing a run in place."""
    if cfg.get("resume"):
        return Path(cfg.resume)
    return Path(HydraConfig.get().runtime.output_dir)


def start_run(cfg: DictConfig, *, dataset_id: str | None) -> Run:
    return Run.from_config(
        cfg,
        run_directory(cfg),
        dataset_id=dataset_id,
        x64=bool(jax.config.read("jax_enable_x64")),
        resume=bool(cfg.get("resume")),
    )


def training_key(cfg: DictConfig) -> jax.Array:
    """The loop's PRNG key (loss noise, dropout): derived from the model seed but distinct
    from the model-initialisation key."""
    return jax.random.fold_in(jax.random.key(int(cfg.model.key.seed)), 1)


def initial_state(
    cfg: DictConfig, run: Run, trainer: Trainer, model: eqx.Module
) -> eqx.Module | TrainerState:
    """The model to start from, or a restored `TrainerState` when resuming in place or
    warm-starting from another run (`cfg.warm_start`, recorded in `resumed_from`)."""
    template = trainer.init(model, key=jax.random.key(0))
    if cfg.get("resume"):
        step = run.latest_step
        if step is None:
            log.warning(
                "resume=%s has no checkpoint; starting from scratch", run.directory
            )
            return model
        log.info("resuming %s from step %d", run.directory, step)
        return run.restore(template)
    if cfg.get("warm_start"):
        source = Run.open(cfg.warm_start)
        step = source.latest_step
        if step is None:
            raise FileNotFoundError(f"warm_start={cfg.warm_start} has no checkpoint")
        run.update(
            resumed_from={
                "run_dir": str(source.directory),
                "run_id": source.record.run_id,
                "step": step,
            }
        )
        log.info("warm start from %s step %d", source.directory, step)
        return source.restore(template)
    return model


def train(
    cfg: DictConfig,
    *,
    model: eqx.Module,
    loss_fn: Any,
    loss_args: Any,
    train_loader: Iterable,
    valid_loader: Iterable,
    dataset_id: str | None,
    optimizer: Any | None = None,
) -> TrainerState:
    """Assemble the run and train. `optimizer` defaults to `cfg.training.optimizer`;
    anything else (a clipping chain, say) must also be what `experiment.state_template` can
    rebuild from the config, so put it in the config rather than passing it here."""
    run = start_run(cfg, dataset_id=dataset_id)
    optimizer = optimizer or build_optimizer(cfg)
    trainer = Trainer(optimizer, loss_fn, loss_args=loss_args)
    start = initial_state(cfg, run, trainer, model)

    wrun = build_wandb_run(cfg, run)
    loggers = [PrintLogger(every=max(1, cfg.training.get("eval_every", 1) * 10))]
    if wrun is not None:
        loggers.append(WandbLogger(wrun, every=cfg.training.get("log_every", 1)))
    logger = DelayedLogger(MultiLogger(*loggers))
    ckpt = run.checkpointer(save_every=cfg.training.get("save_every", 100))

    status, exit_code = "finished", 0
    try:
        state = trainer.train(
            start,
            train_loader,
            key=training_key(cfg) if not isinstance(start, TrainerState) else None,
            num_steps=cfg.training.max_steps,
            logger=logger,
            checkpointer=ckpt,
            evaluate=trainer.evaluator(valid_loader),
            eval_every=cfg.training.get("eval_every", 1),
        )
    except BaseException:
        status, exit_code = "failed", 1
        raise
    finally:
        run.finish(status=status)
        finish_wandb_run(wrun, run, exit_code=exit_code)
    return state
