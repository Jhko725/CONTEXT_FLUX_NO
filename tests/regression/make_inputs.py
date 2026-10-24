"""Extract one context window per dataset into tests/regression/inputs/<input>.npz.

Self-contained inputs mean the tests never need the datasets. Run once, on the machine with the
data, then commit the .npz files (small: ~10 kB for 1D, ~5 MB for 2D Euler):

    JAX_PLATFORMS=cpu uv run python tests/regression/make_inputs.py --data-root ./data/datasets

Takes trajectory 0, time steps 0:CONTEXT_LENGTH of the named split, through the same
ZarrWellDatasetSource the training script uses (so normalisation and channel packing match), and
records loss_args = (dt, *dx) from the source metadata exactly as train_multiphysics.get_loss_args.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tests.regression.common import CONTEXT_LENGTH, INPUTS, load_cases


def extract(dataset: dict, data_root: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    from context_flux_no.data import ZarrWellDatasetSource

    source = ZarrWellDatasetSource(
        well_base_path=str(
            data_root if "well_base_path" not in dataset else dataset["well_base_path"]
        ),
        well_dataset_name=dataset["well_dataset_name"],
        well_split_name=dataset.get("split", "test"),
        window_size=CONTEXT_LENGTH,
        restrict_trajectory_lengths_to=None,
        preload_into_ram=False,
    )
    u = np.asarray(source[0])  # trajectory 0, window starting at t=0: [T, C, *grid]
    dt = float(source.metadata_common["temporal_resolution"])
    dxs = tuple(float(x) for x in source.metadata_common["spatial_resolution"])
    return u, (dt, *dxs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-root", type=Path, default=Path("./data/datasets"))
    ap.add_argument("--only", default=None, help="input name to (re)build")
    args = ap.parse_args()

    INPUTS.mkdir(exist_ok=True)
    done: set[str] = set()
    for case in load_cases():
        name = case["input"]
        if name in done or (args.only and name != args.only):
            continue
        u, loss_args = extract(case["dataset"], args.data_root)
        out = INPUTS / f"{name}.npz"
        np.savez_compressed(
            out,
            u=u.astype(np.float32),
            loss_args=np.asarray(loss_args, dtype=np.float64),
        )
        print(f"wrote {out}  u{u.shape}  loss_args={loss_args}")
        done.add(name)


if __name__ == "__main__":
    main()
