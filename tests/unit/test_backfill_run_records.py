"""`scripts/backfill_run_records.py` on a synthetic legacy checkpoint (the ICLR layout: model
weights only, model config as orbax custom_metadata), and `runs.load_run` on the result."""

import csv
import importlib.util
import sys
from pathlib import Path

import equinox as eqx
import jax
from context_flux_no import runs
from context_flux_no.runs import Run
from context_flux_no.training.io import LEGACY_CHECKPOINTABLE
from orbax.checkpoint import v1 as ocp


ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "backfill_run_records", ROOT / "scripts" / "backfill_run_records.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MODEL_CONFIG = {
    "_target_": "equinox.nn.MLP",
    "in_size": 2,
    "out_size": 1,
    "width_size": 4,
    "depth": 1,
    "key": {"_target_": "jax.random.key", "seed": 0},
}


def _write_legacy(root: Path) -> Path:
    ckpt = (
        root / "cubicflux_1d" / "MLP" / "OneStepLoss" / "seed=0" / "25-09-01-10_00_00"
    )
    ckpt.mkdir(parents=True)
    model = eqx.nn.MLP(2, 1, 4, 1, key=jax.random.key(0))
    with ocp.training.Checkpointer(
        str(ckpt.resolve()), custom_metadata=dict(MODEL_CONFIG)
    ) as c:
        c.save_checkpointables(  # the orbax 0.11 layout: the tree under "pytree"
            7,
            {LEGACY_CHECKPOINTABLE: eqx.filter(model, eqx.is_array)},
            metrics={"valid_loss": 0.5},
        )
    return ckpt


def test_backfill_and_load(tmp_path):
    mod = _load_script()
    ckpt = _write_legacy(tmp_path / "checkpoints")
    audit = tmp_path / "audit.csv"
    with audit.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "project", "id", "url", "created", "model", "target_network",
                "encoder", "data", "seed", "hypernet_init",
            ],
        )  # fmt: skip
        w.writeheader()
        w.writerow(
            dict(project="p", id="abc", url="u", created="c", model="MLP",
                 target_network="", encoder="", data="cubicflux_1d", seed="0",
                 hypernet_init="bias_hyperinit")
        )  # fmt: skip
    assert mod.legacy_dirs(tmp_path / "checkpoints") == [ckpt]
    msg = mod.backfill(ckpt, mod.load_audit(audit), force=False, dry_run=False)
    assert msg.startswith("wrote (1 wandb match")
    assert mod.backfill(ckpt, [], force=False, dry_run=False) == "skip (exists)"

    run = Run.open(ckpt)
    rec = run.record
    assert rec.status == "legacy" and rec.run_id == "legacy-25-09-01-10_00_00"
    assert rec.index["model"] == "MLP" and rec.index["seed"] == 0
    assert rec.index["data"] == "cubicflux_1d" and rec.index["loss"] == "OneStepLoss"
    assert rec.final_step == 7 and rec.wandb["id"] == "abc"
    assert rec.extra["hypernet_init"] is None  # recorded from the model config (F1)

    table = runs.index(tmp_path / "checkpoints")
    assert list(table["status"]) == ["legacy"]
    model, rec2 = runs.load_run(ckpt)
    assert rec2 == rec and isinstance(model, eqx.nn.MLP)
