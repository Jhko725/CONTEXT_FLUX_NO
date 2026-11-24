"""The one way to turn a data source into the batches that training and evaluation read."""

import gc
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import grain
import jax
import jax.extend


def make_loader(
    source: grain.sources.RandomAccessDataSource,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
    repeat: bool = True,
    host_prefetch: int = 20,
    device_prefetch: int = 10,
) -> grain.IterDataset:
    """Build the loader of a random-access source: optional shuffle and repeat, batching
    that drops the incomplete last batch, a host-side prefetch buffer, transfer to JAX's
    default device and a device-side prefetch buffer.

    With `shuffle=False, repeat=False` the loader yields `len(source) // batch_size`
    batches in index order, which is the evaluation order.

    **Arguments:**

    - `source`: the records, read by index.
    - `batch_size`: records per batch.
    - `seed`: seed of the shuffle; unused when `shuffle=False`.
    - `shuffle`: whether to permute the records.
    - `repeat`: whether to cycle through the records without end.
    - `host_prefetch`: batches buffered on the host.
    - `device_prefetch`: batches buffered on the device.

    **Returns:**

    An iterable of batches, each a pytree of `jax.Array`s with a leading axis of length
    `batch_size`.
    """
    dataset = grain.MapDataset.source(source)
    if shuffle:
        dataset = dataset.seed(seed).shuffle()
    if repeat:
        dataset = dataset.repeat()
    loader = (
        dataset.batch(batch_size=batch_size, drop_remainder=True)
        .to_iter_dataset(
            read_options=grain.ReadOptions(prefetch_buffer_size=host_prefetch)
        )
        .map(
            lambda batch: jax.device_put(batch, jax.extend.backend.get_default_device())
        )
    )
    return grain.experimental.ThreadPrefetchIterDataset(
        loader, prefetch_buffer_size=device_prefetch
    )


@contextmanager
def gc_paused(every: int) -> Iterator[Callable[[], None]]:
    """Disable Python's cyclic garbage collector and yield a callable that collects on
    every `every`-th call; on exit the collector is enabled again if it was enabled on
    entry, also after an exception.

    The callable is meant to be invoked once per training step, so that collection runs
    at known steps only. No driver uses it yet:
    the period and the place of the per-step call are still to be chosen.

    **Arguments:**

    - `every`: the number of calls between two collections; at least 1.

    **Returns:**

    A context manager whose value is the per-step callable.
    """
    if every < 1:
        raise ValueError(f"`every` must be at least 1, got {every}.")
    was_enabled = gc.isenabled()
    calls = 0

    def step() -> None:
        nonlocal calls
        calls += 1
        if calls % every == 0:
            gc.collect()

    gc.disable()
    try:
        yield step
    finally:
        if was_enabled:
            gc.enable()
