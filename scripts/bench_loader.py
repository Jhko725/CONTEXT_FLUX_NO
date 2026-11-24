"""Time the training loader of a data config and print its batches per second.

    uv run python scripts/bench_loader.py well_euler_2d --window-size 21 --batch-size 256

The dataset is named by its data config under `scripts/training/configs/data/`, whose
`source:` node is instantiated as `scripts/train.py` does. The loader is `make_loader`
with its training defaults. Two rates are printed: the host rate counts only the time
until each batch is returned, and the device rate also waits until the batch is on the
device. The first batch is drawn before timing starts, so neither rate includes the
start-up of the loader.
"""

import argparse
import time
from pathlib import Path

import hydra
import jax
from context_flux_no.data import make_loader
from omegaconf import OmegaConf


DATA_CONFIGS = Path(__file__).resolve().parent / "training/configs/data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="name of a data config, such as cubic_1d")
    parser.add_argument("--split", default="train", help="split of the dataset")
    parser.add_argument(
        "--window-size", type=int, required=True, help="time steps per record"
    )
    parser.add_argument(
        "--batch-size", type=int, required=True, help="records per batch"
    )
    parser.add_argument("--batches", type=int, default=100, help="batches to time")
    args = parser.parse_args()

    config_path = DATA_CONFIGS / f"{args.dataset}.yaml"
    if not config_path.is_file():
        known = sorted(path.stem for path in DATA_CONFIGS.glob("*.yaml"))
        raise ValueError(f"Unknown dataset {args.dataset!r}; known: {known}.")
    source = hydra.utils.instantiate(
        OmegaConf.load(config_path).source,
        well_split_name=args.split,
        window_size=args.window_size,
    )
    batches = iter(make_loader(source, batch_size=args.batch_size, seed=0))
    jax.block_until_ready(next(batches))

    host_seconds = device_seconds = 0.0
    for _ in range(args.batches):
        start = time.perf_counter()
        batch = next(batches)
        returned = time.perf_counter()
        jax.block_until_ready(batch)
        ready = time.perf_counter()
        host_seconds += returned - start
        device_seconds += ready - start

    print(f"{args.dataset} ({args.split}), {args.batches} batches of {args.batch_size}")
    print(f"host:   {args.batches / host_seconds:.1f} batches/s")
    print(f"device: {args.batches / device_seconds:.1f} batches/s")


if __name__ == "__main__":
    main()
