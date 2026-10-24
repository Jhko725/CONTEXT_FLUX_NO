"""`Run` directories and the `runs` index/loader, end to end on a tiny MLP (CPU)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from context_flux_no.training import ListLogger, Run, RunRecord, runs, Trainer
from context_flux_no.training.run import config_changes
from omegaconf import OmegaConf


class _Loss(eqx.Module):
    def __call__(self, model, batch, args, key):
        x, y = batch
        pred = jax.vmap(model)(x)
        return jnp.mean((pred - y) ** 2), {}


def _loader(n, seed=0, batch=16):
    key = jax.random.key(seed)
    for _ in range(n):
        key, k = jax.random.split(key)
        x = jax.random.normal(k, (batch, 2))
        yield x, x[:, :1] * 2.0 + 1.0


def _cfg(seed=0, lr=1e-2):
    return OmegaConf.create(
        {
            "model": {
                "_target_": "equinox.nn.MLP",
                "in_size": 2,
                "out_size": 1,
                "width_size": 8,
                "depth": 1,
                "key": {"_target_": "jax.random.key", "seed": seed},
            },
            "data": {"name": "toy"},
            "loss_fn": {"_target_": "tests.unit.test_run._Loss"},
            "training": {
                "context_length": 1,
                "batch_size": 16,
                "max_steps": 6,
                "optimizer": {"_target_": "optax.adam", "learning_rate": lr},
            },
        }
    )


def _train(run: Run, num_steps, state=None, seed=0):
    cfg = run.config
    trainer = Trainer(runs.build_optimizer(cfg), _Loss())
    start = state if state is not None else runs.build_model(cfg)
    with run.checkpointer(save_every=2) as ckpt:
        out = trainer.train(
            start,
            _loader(100, seed=seed),
            num_steps=num_steps,
            key=jax.random.key(0),
            logger=ListLogger(),
            checkpointer=ckpt,
            evaluate=trainer.evaluator(_loader(100, seed=1)),
            eval_every=2,
        )
        run.finish(ckpt)
    return out


def test_from_config_writes_layout_and_refuses_reuse(tmp_path):
    run = Run.from_config(_cfg(), tmp_path / "r", dataset_id="toy:train:0000")
    assert run.config_path.exists() and run.metadata_path.exists()
    assert run.record.status == "running"
    assert run.record.index["dataset_id"] == "toy:train:0000"
    assert run.record.index["model"] == "MLP"
    assert run.record.model_config["in_size"] == 2
    with pytest.raises(FileExistsError):
        Run.from_config(_cfg(), tmp_path / "r")
    reopened = Run.open(tmp_path / "r")
    assert reopened.record == run.record
    assert reopened.config == run.config


def test_train_finish_and_load_run(tmp_path):
    run = Run.from_config(_cfg(), tmp_path / "r")
    state = _train(run, num_steps=6)
    rec = RunRecord.read(run.metadata_path)
    assert rec.status == "finished" and rec.final_step == 6
    assert rec.best_step in {2, 4, 6} and rec.best_metric is not None
    # the checkpoint directory alone identifies the run
    with run.checkpointer() as ckpt:
        assert ckpt.custom_metadata["run_record"]["run_id"] == rec.run_id
    model, rec2 = runs.load_run(run.directory)
    assert rec2 == rec
    assert eqx.tree_equal(
        eqx.filter(model, eqx.is_array), eqx.filter(state.model, eqx.is_array)
    )
    model_best, _ = runs.load_run(run.directory, step=rec.best_step)
    assert isinstance(model_best, eqx.nn.MLP)


def test_resume_continues_step_and_records_changes(tmp_path):
    run = Run.from_config(_cfg(), tmp_path / "r")
    _train(run, num_steps=4)
    run_id = run.record.run_id
    resumed = Run.from_config(_cfg(lr=5e-3), tmp_path / "r", resume=True)
    assert resumed.record.run_id == run_id
    assert resumed.record.status == "running"
    assert resumed.record.resumed["from_step"] == 4
    assert resumed.record.resumed["config_changes"] == {
        "training.optimizer.learning_rate": {"old": 1e-2, "new": 5e-3}
    }
    assert OmegaConf.load(resumed.config_path).training.optimizer.learning_rate == 5e-3
    state, _ = runs.load_state(resumed.directory)
    assert int(state.step) == 4
    out = _train(resumed, num_steps=3, state=state, seed=7)
    assert int(out.step) == 7
    assert Run.open(tmp_path / "r").record.final_step == 7


def test_index_and_select(tmp_path):
    for i, name in enumerate(["a", "b"]):
        run = Run.from_config(_cfg(seed=i), tmp_path / "runs" / name)
        if name == "a":
            _train(run, num_steps=2)
    table = runs.index(tmp_path / "runs")
    assert len(table) == 2
    assert set(table.columns) >= {"run_id", "model", "seed", "status", "run_dir"}
    assert sorted(table["status"]) == ["finished", "running"]
    assert list(runs.select(table, seed=1)["status"]) == ["running"]
    assert len(runs.select(table, seed=[0, 1])) == 2
    assert runs.index(tmp_path / "nothing-here").empty


def test_config_changes_reports_leaves_only():
    old = OmegaConf.create({"a": {"b": 1, "c": [1, 2]}, "d": "x"})
    new = OmegaConf.create({"a": {"b": 2, "c": [1, 2]}, "e": True})
    assert config_changes(old, new) == {
        "a.b": {"old": 1, "new": 2},
        "d": {"old": "x", "new": None},
        "e": {"old": None, "new": True},
    }
