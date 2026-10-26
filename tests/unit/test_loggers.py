import jax.numpy as jnp
from context_flux_no.training.loggers import (
    DelayedLogger,
    ListLogger,
    MultiLogger,
    PrintLogger,
    to_floats,
)


def test_delayed_logger_forwards_previous_step_and_flushes_on_close():
    inner = ListLogger()
    with DelayedLogger(inner) as log:
        log.log({"a": jnp.asarray(1.0)}, 1)
        assert inner.records == []  # held back
        log.log({"a": jnp.asarray(2.0)}, 2)
        assert inner.records == [(1, {"a": 1.0})]
    assert inner.records == [(1, {"a": 1.0}), (2, {"a": 2.0})]  # flushed by close


def test_multi_logger_fans_out_and_closes():
    a, b = ListLogger(), ListLogger()
    with MultiLogger(a, b) as log:
        log.log({"x": 3}, 7)
    assert a.records == b.records == [(7, {"x": 3.0})]


def test_print_logger_thins(capsys):
    log = PrintLogger(every=2, keys=("loss",))
    log.log({"loss": 1.0}, 1)
    log.log({"loss": 0.5, "other": 9}, 2)
    out = capsys.readouterr().out
    assert "step 1" not in out and "step 2 | loss: 0.5" in out


def test_to_floats_syncs_device_scalars():
    assert to_floats({"a": jnp.asarray(2.5), "b": 1}) == {"a": 2.5, "b": 1.0}
