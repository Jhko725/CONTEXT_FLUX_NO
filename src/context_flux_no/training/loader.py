"""DEPRECATED — legacy xarray/HDF5 segment loader.

Kept only so that the evaluation notebooks can read the ICLR-era 1D datasets
(cubic 1D, Burgers, shallow water, 2D cubic) until Phase 4 regenerates them in the Well/zarr
schema; removed in Phase 4 (docs/roadmap.md). Do not use in new code — the current data path is
`context_flux_no.data.ZarrWellDatasetSource` with grain.
"""

import queue
import threading

import jax.numpy as jnp
import numpy as np
import xarray as xr


def make_segment_axis(dataset: xr.Dataset, segment_length: int):
    values = dataset["values"]
    if "pde" in values.dims:
        values = dataset["values"].stack(traj=("pde", "ic"))
    else:
        values = values.rename({"ic": "traj"})

    return (
        values.rolling(t=segment_length)
        .construct("segment")
        .isel({"t": slice(segment_length - 1, None)})
        .transpose("traj", "t", "segment", "dim", ...)
    )


class SegmentLoaderBackground:
    """Strongly inspired from levanter's BackgroundIterator class [1].

    [1] https://github.com/marin-community/levanter/blob/main/src/levanter/utils/background_iterable.py"""

    dataset: xr.Dataset
    rng: np.random.Generator
    segment_length: int
    batch_size: int
    batches_per_load: int
    metabatch_queue: queue.Queue
    producer_thread: threading.Thread
    _stop_event: threading.Event

    def __init__(
        self,
        dataset: xr.Dataset,
        segment_length: int,
        batch_size: int,
        batches_per_load: int = 32,
        queue_capacity: int = 5,
        seed: int = 0,
    ):
        self.dataset = make_segment_axis(dataset, segment_length)
        self.rng = np.random.default_rng(seed)
        self.segment_length = segment_length
        self.batch_size = batch_size
        self.batches_per_load = batches_per_load

        self.metabatch_queue = queue.Queue(maxsize=queue_capacity)
        self._stop_event = threading.Event()
        self.producer_thread = threading.Thread(
            target=self._put_metabatch_into_queue, daemon=True
        )
        self.producer_thread.start()

    def __iter__(self):
        while not self._stop_event.is_set():
            metabatch = self.metabatch_queue.get()
            yield from metabatch

    def _put_metabatch_into_queue(self):
        while not self._stop_event.is_set():
            inds_traj = self._sample_traj_indices()
            inds_t0 = self._sample_t0_indices()
            metabatch = self.dataset.isel({"traj": inds_traj, "t": inds_t0}).to_numpy()
            self.metabatch_queue.put(jnp.asarray(metabatch))

    def _sample_traj_indices(self) -> xr.DataArray:
        inds = self.rng.integers(
            0, self.dataset.sizes["traj"], size=(self.batches_per_load, self.batch_size)
        )
        return xr.DataArray(inds, dims=["metabatch", "batch"])

    def _sample_t0_indices(self) -> xr.DataArray:
        inds = self.rng.integers(
            0, self.dataset.sizes["t"], size=(self.batches_per_load, self.batch_size)
        )
        return xr.DataArray(inds, dims=["metabatch", "batch"])


## Older Dataloader code (loads the entire dataset into GPU RAM)
