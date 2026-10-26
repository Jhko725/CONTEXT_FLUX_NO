"""`training.io.load_model` rebuilds a legacy model from the config orbax stored; values the
current code no longer accepts are translated to what they meant at training time (F1)."""

import logging
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from context_flux_no.training.io import checkpoint_template, translate_legacy_config
from orbax.checkpoint import v1 as ocp


class _Model(eqx.Module):
    w: jax.Array
    p: float  # a numpy scalar leaf, as DISCO's `DropPath.p` from `np.linspace`
    s: float  # a Python float: never saved

    def __init__(self):
        self.w = jnp.ones(3)
        self.p = np.linspace(0, 0.1, 2)[1]
        self.s = 0.5


def test_template_asks_for_every_saved_leaf(tmp_path):
    """The trainer saved `eqx.filter(model, eqx.is_array)`; a stored leaf the template does
    not ask for is read through the checkpoint's sharding file (orbax warns) and dropped."""
    saved = eqx.filter(_Model(), eqx.is_array)
    saved = eqx.tree_at(
        lambda m: m.p, saved, jnp.asarray(saved.p)
    )  # stored as an array
    with ocp.training.Checkpointer(tmp_path / "ckpt") as c:
        c.save_pytree(0, saved)
        c.close()
    rebuilt = eqx.tree_at(lambda m: m.p, _Model(), np.float64(0.0))  # a fresh rebuild
    template, rest = checkpoint_template(rebuilt)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ocp.training.Checkpointer(tmp_path / "ckpt") as c:
            loaded = c.load_pytree(0, template)
    assert not [w for w in caught if "harding" in str(w.message)]
    model = eqx.combine(loaded, rest)
    assert float(model.p) == np.float32(0.1) and model.s == 0.5  # stored as f32 by jnp
    np.testing.assert_array_equal(model.w, 1.0)


def test_legacy_hypernet_init_spelling_reads_as_default(caplog):
    stored = {"_target_": "x.Model", "hypernet_init": "bias_hyperinit", "width": 8}
    with caplog.at_level(logging.WARNING, logger="context_flux_no.training.io"):
        fixed = translate_legacy_config(stored)
    assert fixed == {"_target_": "x.Model", "hypernet_init": "default", "width": 8}
    assert stored["hypernet_init"] == "bias_hyperinit"  # the input is not mutated
    assert "F1" in caplog.text


def test_current_configs_pass_through_untouched():
    stored = {"_target_": "x.Model", "hypernet_init": "bias-hyperinit"}
    assert translate_legacy_config(stored) == stored
