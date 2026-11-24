"""Laws of `data.make_loader` and `data.gc_paused`, on a small in-memory source."""

import gc
import itertools
from unittest import mock

import numpy as np
import pytest
from context_flux_no.data import gc_paused, make_loader
from hypothesis import given, strategies as st
from jaxtyping import Int


class _IndexSource:
    """Record `i` is the array `[i]`, so a batch shows which indices it holds."""

    def __init__(self, length: int):
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Int[np.ndarray, "1"]:
        return np.array([index])


def _indices(batches: list) -> Int[np.ndarray, " n"]:
    return np.concatenate([np.asarray(batch)[:, 0] for batch in batches])


@given(
    batch_size=st.integers(1, 8),
    batches_per_epoch=st.integers(1, 8),
    seed=st.integers(0, 2**31 - 1),
)
def test_every_index_appears_once_per_epoch(
    batch_size: int, batches_per_epoch: int, seed: int
):
    """Every index of the source appears exactly once in each epoch of shuffled batches."""
    length = batch_size * batches_per_epoch
    loader = make_loader(_IndexSource(length), batch_size=batch_size, seed=seed)
    batches = list(itertools.islice(iter(loader), 2 * batches_per_epoch))
    indices = _indices(batches)
    for epoch in (indices[:length], indices[length:]):
        np.testing.assert_array_equal(np.sort(epoch), np.arange(length))


@given(length=st.integers(1, 40), batch_size=st.integers(1, 8), seed=st.integers(0, 99))
def test_same_seed_gives_same_batches(length: int, batch_size: int, seed: int):
    """Two loaders with the same seed give the same batches."""

    def first_batches() -> Int[np.ndarray, " n"]:
        loader = make_loader(_IndexSource(length), batch_size=batch_size, seed=seed)
        return _indices(list(itertools.islice(iter(loader), 5)))

    np.testing.assert_array_equal(first_batches(), first_batches())


@given(length=st.integers(0, 40), batch_size=st.integers(1, 8))
def test_evaluation_order_is_index_order_without_remainder(
    length: int, batch_size: int
):
    """Without shuffling or repetition, the batches are the indices in order, and the
    incomplete last batch is dropped."""
    loader = make_loader(
        _IndexSource(length), batch_size=batch_size, seed=0, shuffle=False, repeat=False
    )
    batches = list(loader)
    count = length // batch_size
    assert len(batches) == count
    if count:
        np.testing.assert_array_equal(_indices(batches), np.arange(count * batch_size))


@given(every=st.integers(1, 10), calls=st.integers(0, 50))
def test_gc_paused_collects_every_every_calls(every: int, calls: int):
    """`gc_paused(every)` disables the collector and collects once every `every` steps."""
    with mock.patch.object(gc, "collect") as collect, gc_paused(every) as step:
        assert not gc.isenabled()
        for _ in range(calls):
            step()
    assert collect.call_count == calls // every


@given(every=st.integers(1, 10), calls=st.integers(0, 20))
def test_gc_paused_enables_the_collector_after_an_exception(every: int, calls: int):
    """`gc_paused` enables the collector again when its block raises."""
    assert gc.isenabled()
    with pytest.raises(RuntimeError), gc_paused(every) as step:
        for _ in range(calls):
            step()
        raise RuntimeError
    assert gc.isenabled()


def test_gc_paused_rejects_a_period_below_one():
    """`gc_paused` with a period below one raises `ValueError`."""
    with pytest.raises(ValueError, match="at least 1"):
        gc_paused(0).__enter__()
