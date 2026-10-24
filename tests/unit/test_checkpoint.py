import equinox as eqx
import jax
import optax
import pytest
from context_flux_no.training.checkpoint import NullCheckpointer, OrbaxCheckpointer
from context_flux_no.training.trainer import TrainerState


OPT = optax.adam(1e-3)  # one optimizer object: TrainerState keeps it as a static field,
# and two optax.adam(...) instances are different pytree structures


def _state(seed=0):
    model = eqx.nn.MLP(2, 1, 8, 1, key=jax.random.key(seed))
    return TrainerState.init(model, OPT, key=jax.random.key(seed + 1))


def test_null_checkpointer_never_saves():
    assert NullCheckpointer().save(1, _state(), {"valid_loss": 1.0}) is False


def test_save_restore_roundtrip(tmp_path):
    state = _state()
    with OrbaxCheckpointer(
        tmp_path / "ckpt", metric="valid_loss", custom_metadata={"k": "v"}
    ) as c:
        assert c.save(1, state, {"valid_loss": 0.5}, force=True)
        assert c.latest_step == 1
        assert c.custom_metadata == {"k": "v"}
        restored = c.restore(_state(seed=5), step=1)  # different template values
    assert eqx.tree_equal(
        eqx.filter(restored, eqx.is_array), eqx.filter(state, eqx.is_array)
    )


def test_policy_every_and_force(tmp_path):
    state = _state()
    with OrbaxCheckpointer(tmp_path / "ckpt", save_every=5, metric=None) as c:
        assert c.save(1, state, {}) is False  # off policy
        assert c.save(5, state, {}) is True
        assert c.save(7, state, {}, force=True) is True
        assert c.steps == [7] or set(c.steps) >= {
            7
        }  # LatestN(1) keeps 7; 5 may be gone


def test_best_and_latest_kept(tmp_path):
    s = _state()
    with OrbaxCheckpointer(
        tmp_path / "ckpt", save_every=1, metric="valid_loss", keep_best=1
    ) as c:
        c.save(1, s, {"valid_loss": 0.3})
        c.save(2, s, {"valid_loss": 0.1})  # best
        c.save(3, s, {"valid_loss": 0.2})  # latest
        kept = set(c.steps)
    assert {2, 3} <= kept and 1 not in kept


def test_missing_metric_raises(tmp_path):
    with OrbaxCheckpointer(tmp_path / "ckpt", metric="valid_loss") as c:
        with pytest.raises(KeyError, match="valid_loss"):
            c.save(1, _state(), {"train_loss": 1.0})
