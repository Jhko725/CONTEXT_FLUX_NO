"""Evaluate trained runs on one test set and write the result to one HDF5 file.

    uv run python scripts/evaluate.py --runs runs/2026-10-24/12-00-00 \\
        20261024-120000-ab12cd34 --data cubic_1d --out results/cubic_1d.h5

A run is named by its directory or by its run id, which is looked up in the index of
`./runs`, where `scripts/train.py` writes its runs. `--data` names a data config under
`scripts/training/configs/data/`, whose `source:` node is read in evaluation order, one
whole trajectory per record. Each trajectory gives one rollout: the first
`--context-length` frames are the context, and the model predicts every remaining frame.

The HDF5 file holds one group per run, named by its run id, with the fields of
`evaluation.RolloutDiagnostics` as datasets, and a group `references` with the fields of
`evaluation.RolloutReferences`. Its `EvaluationRecord` is written beside it, with the
suffix `.json` in place of the file's own.

This script is the only place where a trained model, whose class may live in `legacy/`,
meets `evaluation`; the model reaches `evaluation` through its `Operator` protocol.
"""

import argparse
import dataclasses
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import h5py
import hydra
import numpy as np
from context_flux_no import runs
from context_flux_no.data import make_loader, WellDatasetSourceBase
from context_flux_no.evaluation import (
    evaluate_references,
    evaluate_rollouts,
    EvaluationRecord,
    Operator,
    RolloutEvaluation,
    RolloutReferences,
)
from jaxtyping import Float
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parent / "training" / "configs"
RUNS_ROOT = Path("runs")
"""The root under which run ids are looked up: the parent of `hydra.run.dir` in
`scripts/training/configs/config.yaml`, relative to the working directory."""

AREA_THRESHOLDS = (0.05, 0.1, 0.2)
"""The default thresholds of the error area fraction: the rebuttal branch's
`scripts/evaluation/rollout_diagnostics.py` reports 0.1 and uses the other two as a
sensitivity check."""


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line, or `argv` when it is given."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories or run ids; an id is looked up under ./runs.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="A data config name under scripts/training/configs/data/.",
    )
    parser.add_argument(
        "--split", default="test", choices=("train", "valid", "test"), help="The split."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=20,
        help="The number of context frames of each rollout.",
    )
    parser.add_argument(
        "--area-thresholds",
        nargs="+",
        type=float,
        default=list(AREA_THRESHOLDS),
        help="The thresholds of the error area fraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Trajectories per loader batch and rollouts run in parallel.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate the first N trajectories only.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="The HDF5 file to write; it must not exist.",
    )
    return parser.parse_args(argv)


def resolve_runs(entries: Sequence[str], root: Path) -> list[Path]:
    """Turn each entry of `--runs` into a run directory.

    An entry that is a directory is taken as it is. Any other entry is a run id, which
    must match exactly one run of `runs.index(root)`.

    **Arguments:**

    - `entries`: Run directories or run ids.
    - `root`: The root under which run ids are looked up.

    **Returns:**

    The run directories, in the order of `entries`.

    **Raises:**

    `ValueError` naming every entry that matches no run or several runs, or every
    run directory that more than one entry names.
    """
    needs_index = any(not Path(entry).is_dir() for entry in entries)
    table = runs.index(root) if needs_index and root.is_dir() else None
    run_dirs, unresolved = [], []
    for entry in entries:
        if Path(entry).is_dir():
            run_dirs.append(Path(entry))
            continue
        if table is None or table.empty:
            unresolved.append(f"{entry!r} (0 runs)")
            continue
        matches = runs.select(table, run_id=entry)
        if len(matches) == 1:
            run_dirs.append(Path(matches["run_dir"].iloc[0]))
        else:
            unresolved.append(f"{entry!r} ({len(matches)} runs)")
    if unresolved:
        raise ValueError(
            f"No single run directory or run id under {root} for: "
            f"{', '.join(unresolved)}."
        )
    repeated = sorted({str(d) for d in run_dirs if run_dirs.count(d) > 1})
    if repeated:
        raise ValueError(f"--runs names these runs more than once: {repeated}.")
    return run_dirs


def build_source(data: str, split: str) -> WellDatasetSourceBase:
    """Build the source of a data config whose records are whole trajectories.

    The window is the trajectory length: the config's `restrict_trajectory_lengths_to`
    when it sets one, and otherwise the length that every file of the split shares.
    A first source with a window of one frame and no preloading reads that length.

    **Arguments:**

    - `data`: A data config name under `scripts/training/configs/data/`.
    - `split`: The split to read.

    **Returns:**

    The source, with one record per trajectory.

    **Raises:**

    `FileNotFoundError` if no data config has that name, and `ValueError` if the files
    of the split hold trajectories of different lengths.
    """
    path = CONFIG_DIR / "data" / f"{data}.yaml"
    if not path.is_file():
        names = sorted(p.stem for p in path.parent.glob("*.yaml"))
        raise FileNotFoundError(f"No data config {data!r}; the configs are {names}.")
    config = OmegaConf.load(path).source
    no_preload = {"preload_into_ram": False} if "preload_into_ram" in config else {}
    probe = hydra.utils.instantiate(
        config, well_split_name=split, window_size=1, **no_preload
    )
    lengths = set(probe.metadata_varying["len_trajectories"])
    if probe.restrict_trajectory_lengths_to is not None:
        length = probe.restrict_trajectory_lengths_to
    elif len(lengths) == 1:
        length = lengths.pop()
    else:
        raise ValueError(
            f"The {split} files of {data!r} hold trajectories of lengths "
            f"{sorted(lengths)}; a rollout needs one length."
        )
    return hydra.utils.instantiate(config, well_split_name=split, window_size=length)


def read_trajectories(
    source: WellDatasetSourceBase, *, batch_size: int, limit: int | None
) -> Float[np.ndarray, " trajectories frames channels *spatial_dims"]:
    """Read the trajectories of `source` in evaluation order onto the host.

    The loader drops an incomplete last batch, so the trajectories past the last whole
    batch are left out, and their number is printed.

    **Arguments:**

    - `source`: A source with one record per trajectory.
    - `batch_size`: The trajectories per batch of the loader.
    - `limit`: The number of trajectories to read, or `None` for all.

    **Returns:**

    The trajectories, in the order of the source's index.

    **Raises:**

    `ValueError` if fewer than `batch_size` trajectories are to be read.
    """
    wanted = len(source) if limit is None else min(limit, len(source))
    num_batches = wanted // batch_size
    if num_batches == 0:
        raise ValueError(
            f"{wanted} trajectories make no batch of {batch_size}; lower --batch-size."
        )
    loader = make_loader(
        source, batch_size=batch_size, seed=0, shuffle=False, repeat=False
    )
    batches = [np.asarray(batch) for batch in islice(loader, num_batches)]
    left_out = wanted - num_batches * batch_size
    if left_out:
        print(f"Left out the last {left_out} of {wanted} trajectories.")
    return np.concatenate(batches)


def _write_fields(group: h5py.Group, container: Any) -> None:
    """Write every field of a dataclass or `eqx.Module` of arrays as a dataset."""
    for field in dataclasses.fields(container):
        group.create_dataset(
            field.name, data=np.asarray(getattr(container, field.name))
        )


def write_results(
    path: Path,
    *,
    evaluations: dict[str, RolloutEvaluation],
    run_dirs: dict[str, Path],
    references: RolloutReferences,
    attributes: dict[str, Any],
) -> None:
    """Write the references and the evaluation of every run to one HDF5 file.

    **Arguments:**

    - `path`: The file to create.
    - `evaluations`: The evaluation of each run, by run id.
    - `run_dirs`: The directory of each run, by run id.
    - `references`: The model-independent quantities of the rollouts.
    - `attributes`: The attributes of the file's root.
    """
    with h5py.File(path, "w") as file:
        for name, value in attributes.items():
            file.attrs[name] = value
        _write_fields(file.create_group("references"), references)
        for run_id, evaluation in evaluations.items():
            group = file.create_group(run_id)
            group.attrs["run_dir"] = str(run_dirs[run_id])
            _write_fields(group, evaluation.diagnostics)
            for name in ("minima", "error_spectrum", "num_finite_spectra"):
                value = getattr(evaluation, name)
                if value is not None:
                    group.create_dataset(name, data=value)


def main(argv: Sequence[str] | None = None) -> None:
    """Evaluate the runs of `argv`, or of the command line, and write the result."""
    args = parse_arguments(argv)
    record_path = args.out.with_suffix(".json")
    for path in (args.out, record_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; choose another --out.")
    run_dirs = resolve_runs(args.runs, RUNS_ROOT)

    source = build_source(args.data, args.split)
    trajectories = read_trajectories(
        source, batch_size=args.batch_size, limit=args.limit
    )
    model_args = source.loss_args
    area_thresholds = tuple(args.area_thresholds)

    evaluations, dirs_by_id = {}, {}
    for run_dir in run_dirs:
        model, run_record = runs.load_run(run_dir)
        # The positive quantities are left out. A store records its family's class
        # name, but its parameters vary per trajectory, and `PositiveQuantities` takes
        # one frame without them, so one family instance cannot serve the whole store.
        # The cast is type-level only: a trained model satisfies `Operator` by its
        # `rollout` method.
        evaluations[run_record.run_id] = evaluate_rollouts(
            cast(Operator, eqx.nn.inference_mode(model)),
            trajectories,
            model_args,
            context_length=args.context_length,
            area_thresholds=area_thresholds,
            positive_quantities=None,
            batch_size=args.batch_size,
        )
        dirs_by_id[run_record.run_id] = run_dir
    references = evaluate_references(trajectories, context_length=args.context_length)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_results(
        args.out,
        evaluations=evaluations,
        run_dirs=dirs_by_id,
        references=references,
        attributes={
            "context_length": args.context_length,
            "area_thresholds": area_thresholds,
            "model_args": model_args,
            "num_trajectories": trajectories.shape[0],
        },
    )
    arguments = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    EvaluationRecord.create(
        run_ids=list(evaluations),
        dataset_id=source.dataset_id,
        split=args.split,
        arguments=arguments,
    ).write(record_path)
    print(f"Wrote {args.out} and {record_path}.")


if __name__ == "__main__":
    main()
