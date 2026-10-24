"""The training loop with injected side effects, on a tiny model (CPU)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from context_flux_no.training import ListLogger, OrbaxCheckpointer, Trainer


class _Loss(eqx.Module):
    """MSE of a 1-D MLP on (x, y) batches; `args` ignored; mirrors AbstractTrainLoss."""

    def __call__(self, model, batch, args, key):
        x, y = batch
        pred = jax.vmap(model)(x)
        loss = jnp.mean((pred - y) ** 2)
        return loss, {"mae": jnp.mean(jnp.abs(pred - y))}


def _loader(n_batches, seed=0, batch=32):
    key = jax.random.key(seed)
    for _ in range(n_batches):
        key, k = jax.random.split(key)
        x = jax.random.normal(k, (batch, 2))
        yield x, (x[:, :1] * 2.0 + 1.0)


def _trainer():
    return Trainer(optax.adam(1e-2), _Loss())


def _model(seed=0):
    return eqx.nn.MLP(2, 1, 16, 1, key=jax.random.key(seed))


def test_loop_logs_evaluates_and_force_checkpoints_final(tmp_path):
    trainer = _trainer()
    log = ListLogger()
    ckpt = OrbaxCheckpointer(tmp_path / "ckpt", save_every=100, metric="valid_loss")
    evaluate = trainer.evaluator(_loader(100, seed=1))
    state = trainer.train(
        _model(),
        _loader(100),
        num_steps=20,
        key=jax.random.key(0),
        logger=log,
        checkpointer=ckpt,
        evaluate=evaluate,
        eval_every=10,
    )
    assert int(state.step) == 20
    steps = [s for s, _ in log.records]
    assert (
        steps.count(10) == 2 and steps.count(20) == 2
    )  # train + valid metrics at eval steps
    assert any("valid_loss" in m for _, m in log.records)
    with OrbaxCheckpointer(tmp_path / "ckpt", metric=None) as c:
        assert c.latest_step == 20  # forced final save despite save_every=100
    train_losses = [m["train_loss"] for _, m in log.records if "train_loss" in m]
    assert train_losses[-1] < train_losses[0]


def test_same_seed_same_state_and_resume_continues(tmp_path):
    trainer = _trainer()
    a = trainer.train(_model(), _loader(10), num_steps=10, key=jax.random.key(0))
    b = trainer.train(_model(), _loader(10), num_steps=10, key=jax.random.key(0))
    assert eqx.tree_equal(eqx.filter(a, eqx.is_array), eqx.filter(b, eqx.is_array))
    c = trainer.train(a, _loader(5, seed=9), num_steps=5)  # resume from a state
    assert int(c.step) == 15


def test_loader_ending_early_warns(tmp_path):
    trainer = _trainer()
    with pytest.warns(UserWarning, match="ended after 3 of 10"):
        state = trainer.train(_model(), _loader(3), num_steps=10, key=jax.random.key(0))
    assert int(state.step) == 3


def test_nonfinite_validation_stops_before_checkpointing(tmp_path):
    trainer = _trainer()
    ckpt = OrbaxCheckpointer(tmp_path / "ckpt", save_every=1, metric="valid_loss")

    def bad_evaluate(state):
        return {"valid_loss": jnp.nan}

    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train(
            _model(),
            _loader(10),
            num_steps=10,
            key=jax.random.key(0),
            checkpointer=ckpt,
            evaluate=bad_evaluate,
            eval_every=5,
        )
    with OrbaxCheckpointer(tmp_path / "ckpt", metric=None) as c:
        assert c.latest_step is None
