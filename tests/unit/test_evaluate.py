"""`scripts/evaluate.py` end to end: a tiny FNO is trained for two steps on a synthetic
cubic-1D store, as `tests/unit/test_experiment.py` trains its runs, then evaluated by run
id on the store's test split. The result file must hold the run's group and the
references with the shapes of the rollouts, and the record must round-trip."""

from pathlib import Path

import h5py
import hydra
import numpy as np
import pytest
from context_flux_no import experiment
from context_flux_no.data import make_loader
from context_flux_no.evaluation import EvaluationRecord
from context_flux_no.runs import Run
from context_flux_no.simulations import Grid, normalize_bcs, WellWriter
from context_flux_no.simulations.pde import CubicFlux1D
from hydra import compose, initialize_config_dir
from jaxtyping import Float
from omegaconf import DictConfig
from scripts.evaluate import CONFIG_DIR, main

from tests.helpers import make_dataset_record, point_hydra_output_dir_at


NUM_FRAMES = 80  # `restrict_trajectory_lengths_to` of the `cubic_1d` data config
NUM_CELLS = 16
CONTEXT_LENGTH = 2
NUM_TRAJECTORIES = {"train": 4, "valid": 4, "test": 3}


def _translating_waves(
    num_trajectories: int, seed: int
) -> Float[np.ndarray, " trajectories frames 1 cells"]:
    """Sine waves of random phase that move by a quarter cell per frame."""
    x = np.linspace(0.0, 1.0, NUM_CELLS, endpoint=False)
    t = np.arange(NUM_FRAMES)[:, None] / (4 * NUM_CELLS)
    phases = np.random.default_rng(seed).uniform(0.0, 1.0, num_trajectories)
    waves = [np.sin(2 * np.pi * (x - t + phase)) for phase in phases]
    return np.stack(waves)[:, :, None].astype(np.float32)


def _write_store(base: Path, split: str, seed: int) -> None:
    """Write the `cubic_1d` store of one split under `base`."""
    pde = CubicFlux1D(1.0, 0.0, 0.0)
    grid = Grid(((0.0, 1.0),), (NUM_CELLS,))
    bcs = normalize_bcs("periodic", 1)
    num_trajectories = NUM_TRAJECTORIES[split]
    record = make_dataset_record(
        split=split,
        n_trajectories=num_trajectories,
        grid=grid.to_dict(),
        time={"t_span": [0.0, 1.0], "Nt": NUM_FRAMES - 1, "dt": 1 / (NUM_FRAMES - 1)},
        seed=seed,
    )
    path = base / "cubic_1d" / "data" / split / f"cubic_1d_seed={seed}.zarr"
    writer = WellWriter.create(
        path,
        pde=pde,
        grid=grid,
        t_out=np.linspace(0.0, 1.0, NUM_FRAMES),
        bcs=bcs,
        record=record,
    )
    for index, wave in enumerate(_translating_waves(num_trajectories, seed)):
        writer.write_trajectory(index, wave, pde.parameters)
    writer.finalize(num_trajectories)


def _training_config() -> DictConfig:
    """The `fno_naive` model on the `cubic_1d` data, shrunk to train in seconds."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                "model=fno_naive",
                "data=cubic_1d",
                "model.lift_dim=8",
                "model.depth=1",
                "model.frequency_modes=4",
                f"training.context_length={CONTEXT_LENGTH}",
                "training.batch_size=2",
                "training.max_steps=2",
                "training.eval_every=1",
                "training.save_every=1",
            ],
        )


def _train_run(output_dir: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Train the run of `_training_config` into `output_dir` and return its run id."""
    point_hydra_output_dir_at(monkeypatch, output_dir)
    cfg = _training_config()
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    window = experiment.segment_length(cfg, loss_fn)
    train, valid = (
        hydra.utils.instantiate(cfg.data.source, well_split_name=s, window_size=window)
        for s in ("train", "valid")
    )
    experiment.train(
        cfg,
        model=experiment.build_model(cfg),
        loss_fn=loss_fn,
        loss_args=train.loss_args,
        train_loader=make_loader(train, batch_size=2, seed=0),
        valid_loader=make_loader(valid, batch_size=2, seed=1),
        dataset_id=train.dataset_id,
    )
    return Run.open(output_dir).record.run_id


def test_evaluate_writes_the_run_group_and_its_record(tmp_path, monkeypatch):
    """`scripts/evaluate.py` writes one group per run with metrics of the expected
    shapes, the identity reference, and an `EvaluationRecord` that survives a round
    trip."""
    monkeypatch.chdir(tmp_path)
    for seed, split in enumerate(NUM_TRAJECTORIES):
        _write_store(tmp_path / "data" / "datasets", split, seed)
    run_id = _train_run(tmp_path / "runs" / "2026-11-25" / "12-00-00", monkeypatch)

    out = tmp_path / "results" / "cubic_1d.h5"
    main(
        [
            "--runs",
            run_id,
            "--data",
            "cubic_1d",
            "--context-length",
            str(CONTEXT_LENGTH),
            "--area-thresholds",
            "0.1",
            "0.2",
            "--batch-size",
            "1",
            "--limit",
            "2",
            "--out",
            str(out),
        ]
    )

    num_steps = NUM_FRAMES - CONTEXT_LENGTH
    with h5py.File(out, "r") as file:
        assert set(file) == {run_id, "references"}
        group = file[run_id]
        assert group["l2"].shape == (2, num_steps)
        assert group["l2_per_channel"].shape == (2, num_steps, 1)
        assert group["area_fraction"].shape == (2, num_steps, 2)
        assert np.all(np.isfinite(group["l2"][()]))
        assert file["references"]["l2_identity"].shape == (2, num_steps)
        assert file.attrs["num_trajectories"] == 2

    record = EvaluationRecord.read(out.with_suffix(".json"))
    assert record.run_ids == [run_id]
    assert record.split == "test"
    assert record.dataset_id.startswith("cubic_1d:test:")
    assert record.arguments["limit"] == 2
    copy_path = tmp_path / "copy.json"
    record.write(copy_path)
    assert EvaluationRecord.read(copy_path) == record
