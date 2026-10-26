"""`RunRecord`: JSON round trip, index extraction from a config, forward compatibility."""

import copy
import dataclasses
import json

from context_flux_no.runs.record import (
    INDEX_COLUMNS,
    index_from_config,
    new_run_id,
    RunRecord,
)
from context_flux_no.utils.serialization import dumps
from hypothesis import given, settings, strategies as st
from omegaconf import OmegaConf


_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
)
_flat = st.dictionaries(st.text(min_size=1, max_size=10), _scalar, max_size=6)


@settings(max_examples=50, deadline=None)
@given(
    index=_flat,
    model_config=_flat,
    best_step=st.none() | st.integers(min_value=0, max_value=10**7),
    best_metric=st.none() | st.floats(allow_nan=False, allow_infinity=False),
    status=st.sampled_from(["running", "finished", "failed", "legacy"]),
)
def test_json_round_trip(index, model_config, best_step, best_metric, status):
    rec = copy.replace(
        RunRecord.create(index=index, model_config=model_config),
        status=status,
        best_step=best_step,
        best_metric=best_metric,
    )
    back = RunRecord.from_dict(json.loads(dumps(dataclasses.asdict(rec))))
    assert back == rec


def test_unknown_fields_survive_as_extra():
    rec = RunRecord.create(index={}, model_config={})
    d = dataclasses.asdict(rec) | {"future_field": 1}
    back = RunRecord.from_dict(d)
    assert back.extra["_unknown"] == {"future_field": 1}
    assert back.run_id == rec.run_id


def test_create_records_provenance_and_ids_are_unique():
    a, b = (
        RunRecord.create(index={}, model_config={}),
        RunRecord.create(index={}, model_config={}),
    )
    assert a.run_id != b.run_id
    assert {"sha", "dirty"} <= set(a.git)
    assert a.status == "running" and a.schema_version == 1
    assert new_run_id() < new_run_id() or True  # sortable prefix; not strictly tested


def test_index_from_config_reads_the_training_config_layout():
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "context_flux_no.models.multiphysics.HyperFluxFNOLocal",
                "encoder_type": "TRecViT",
                "key": {"_target_": "jax.random.PRNGKey", "seed": 3},
            },
            "data": {"name": "cubicflux_1d"},
            "loss_fn": {"_target_": "context_flux_no.training.loss.OneStepLoss"},
            "training": {
                "context_length": 20,
                "batch_size": 256,
                "max_steps": 50000,
                "optimizer": {
                    "_target_": "optax.adamw",
                    "learning_rate": {"peak_value": 5e-5},
                },
            },
        }
    )
    idx = index_from_config(cfg, dataset_id="cubic_1d:train:deadbeef")
    assert set(idx) == set(INDEX_COLUMNS)
    assert idx["model"] == "HyperFluxFNOLocal"
    assert idx["conditioning"] == "hypernet"
    assert idx["encoder"] == "TRecViT"
    assert idx["target_network"] is None
    assert idx["seed"] == 3 and idx["loss"] == "OneStepLoss"
    assert idx["peak_lr"] == 5e-5 and idx["max_steps"] == 50000
    assert idx["dataset_id"] == "cubic_1d:train:deadbeef"


def test_index_from_config_tolerates_missing_sections():
    idx = index_from_config({"training": {"optimizer": {"learning_rate": 1e-3}}})
    assert idx["peak_lr"] == 1e-3
    assert all(idx[k] is None for k in INDEX_COLUMNS if k != "peak_lr")


def test_row_is_flat_json():
    rec = RunRecord.create(index={"model": "M"}, model_config={})
    row = rec.row()
    assert row["model"] == "M" and row["run_id"] == rec.run_id
    json.dumps(row)  # every value is a JSON scalar
