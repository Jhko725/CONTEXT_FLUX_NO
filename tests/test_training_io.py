"""`training.io.load_model` rebuilds a legacy model from the config orbax stored; values the
current code no longer accepts are translated to what they meant at training time (F1)."""

import logging

from context_flux_no.training.io import translate_legacy_config


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
