"""`training.experiment.train`: the script-level assembly (run dir, resume, warm start,
wandb disabled) on a tiny MLP, driven through Hydra's compose API with a fake output dir."""

import jax
import jax.numpy as jnp
import pytest
from context_flux_no import experiment, runs
from context_flux_no.runs import Run
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from tests.unit.test_run import _cfg, _loader, _Loss


@pytest.fixture
def hydra_output_dir(tmp_path, monkeypatch):
    """Make `HydraConfig.get().runtime.output_dir` point at a temp dir."""
    out = tmp_path / "runs" / "2026-10-24" / "12-00-00"
    hc = OmegaConf.create({"runtime": {"output_dir": str(out)}, "mode": None})
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: hc))
    return out


def _full_cfg(**training):
    cfg = _cfg()
    with open_dict(cfg):
        cfg.wandb = {"mode": "disabled"}
        cfg.gpu_id = "auto"
        cfg.resume = None
        cfg.warm_start = None
        cfg.experiment = None
        cfg.training.update({"eval_every": 2, "save_every": 2, "log_every": 1})
        cfg.training.update(training)
    return cfg


def _train(cfg, num_steps):
    cfg.training.max_steps = num_steps
    return experiment.train(
        cfg,
        model=experiment.build_model(cfg),
        loss_fn=_Loss(),
        loss_args=None,
        train_loader=_loader(100, seed=3),
        valid_loader=_loader(100, seed=4),
        dataset_id="toy:train:0000",
    )


def test_train_writes_run_in_hydra_output_dir(hydra_output_dir):
    state = _train(_full_cfg(), 4)
    run = Run.open(hydra_output_dir)
    assert run.record.status == "finished" and run.record.final_step == 4
    assert run.record.index["dataset_id"] == "toy:train:0000"
    assert run.record.wandb is None
    model, _ = runs.load_run(hydra_output_dir)
    assert jnp.allclose(
        jax.vmap(model)(jnp.ones((2, 2))), jax.vmap(state.model)(jnp.ones((2, 2)))
    )


def test_resume_in_place_and_warm_start(hydra_output_dir):
    _train(_full_cfg(), 4)
    cfg = _full_cfg()
    cfg.resume = str(hydra_output_dir)
    state = _train(cfg, 2)
    assert int(state.step) == 6
    run = Run.open(hydra_output_dir)
    assert run.record.resumed["from_step"] == 4 and run.record.final_step == 6

    other = hydra_output_dir.parent / "warm"
    cfg = _full_cfg()
    cfg.warm_start = str(hydra_output_dir)
    hc = OmegaConf.create({"runtime": {"output_dir": str(other)}, "mode": None})
    HydraConfig.get = staticmethod(lambda: hc)  # type: ignore[method-assign]
    state = _train(cfg, 1)
    assert int(state.step) == 7
    warm = Run.open(other)
    assert warm.record.run_id != run.record.run_id
    assert warm.record.resumed_from["step"] == 6
    assert warm.record.resumed_from["run_id"] == run.record.run_id


def test_failure_marks_record_failed(hydra_output_dir):
    class Boom(_Loss):
        def __call__(self, model, batch, args, key):
            loss, aux = super().__call__(model, batch, args, key)
            return loss * jnp.nan, aux

    cfg = _full_cfg()
    cfg.training.max_steps = 4
    with pytest.raises(FloatingPointError):
        experiment.train(
            cfg,
            model=experiment.build_model(cfg),
            loss_fn=Boom(),
            loss_args=None,
            train_loader=_loader(10),
            valid_loader=_loader(10),
            dataset_id=None,
        )
    assert Run.open(hydra_output_dir).record.status == "failed"
