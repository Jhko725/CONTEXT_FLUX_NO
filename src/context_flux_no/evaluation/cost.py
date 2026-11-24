"""The FLOP count, memory and run time of one compiled program.

`count_flops` reads the FLOPs that XLA counts in a function, and `measure` compiles a
function once and times repeated calls of it. Both take any function of arrays, so
nothing here depends on a model class.
"""

import dataclasses
import time
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Float, PyTree


@dataclasses.dataclass(frozen=True)
class Measurement:
    """The cost of one compiled program, as XLA reports it and as it runs.

    **Attributes:**

    - `flops`: The FLOPs that XLA counts in the program. A loop body is counted once,
        whatever the number of iterations.
    - `argument_bytes`: The size of the inputs of the program on the device.
    - `output_bytes`: The size of its outputs.
    - `temp_bytes`: The size of the buffers that the program allocates while it runs.
    - `seconds`: The wall-clock time of each timed call.
    - `peak_bytes`: The peak memory in use on the device since the process started,
        read after the timed calls. NaN on a device without memory statistics, such as
        the CPU.
    """

    flops: float
    argument_bytes: int
    output_bytes: int
    temp_bytes: int
    seconds: Float[np.ndarray, " repeats"]
    peak_bytes: float


def _compile(
    fn: Callable[..., Any], *args: Any
) -> tuple[jax.stages.Compiled, PyTree[Array]]:
    """Compile `fn` for `args` and return the executable and the arrays it takes.

    Every non-array leaf of `args`, such as the activation function of a model, is
    closed over, and the same leaves are dropped from the output, so that `jax.jit`
    sees arrays only. A function that returns a model, such as a training step, then
    returns its arrays.
    """
    dynamic, static = eqx.partition(args, eqx.is_array)
    jitted = jax.jit(lambda d: eqx.filter(fn(*eqx.combine(d, static)), eqx.is_array))
    return jitted.lower(dynamic).compile(), dynamic


def _flops_of(compiled: jax.stages.Compiled) -> float:
    """Return the FLOPs of the cost analysis of `compiled`, or NaN if it has none."""
    analysis = compiled.cost_analysis()
    # Older JAX versions return one dictionary per device.
    if isinstance(analysis, list):
        analysis = analysis[0]
    return float(analysis.get("flops", np.nan))


def count_flops(fn: Callable[..., Any], *args: Any) -> float:
    """Return the FLOPs that XLA counts in `fn(*args)`.

    The count is that of the compiled program on the default device. A loop body is
    counted once, so a function that loops, such as a rollout or an adaptive ODE
    solve, is counted one iteration at a time.

    **Arguments:**

    - `fn`: The function to count. Its non-array arguments are closed over.
    - `args`: Its arguments.

    **Returns:**

    The FLOP count, or NaN if XLA reports none.
    """
    return _flops_of(_compile(fn, *args)[0])


def measure(
    fn: Callable[..., Any], *args: Any, repeats: int, warmup: int
) -> Measurement:
    """Compile `fn(*args)` once, then time `repeats` calls after `warmup` calls.

    Each call is waited on until its outputs are ready, so the time includes the whole
    computation on the device and not only its dispatch.

    **Arguments:**

    - `fn`: The function to measure. Its non-array arguments are closed over.
    - `args`: Its arguments.
    - `repeats`: The number of timed calls.
    - `warmup`: The number of untimed calls before them.

    **Returns:**

    The `Measurement` of the compiled program.

    **Raises:**

    `ValueError` if `repeats < 1` or `warmup < 0`.
    """
    if repeats < 1 or warmup < 0:
        raise ValueError(f"Need repeats >= 1 and warmup >= 0, got {repeats}, {warmup}.")
    compiled, dynamic = _compile(fn, *args)
    memory = compiled.memory_analysis()
    for _ in range(warmup):
        jax.block_until_ready(compiled(dynamic))
    seconds = np.empty(repeats)
    for i in range(repeats):
        start = time.perf_counter()
        out = jax.block_until_ready(compiled(dynamic))
        seconds[i] = time.perf_counter() - start
    (device,) = jax.tree.leaves(out)[0].devices()
    stats = device.memory_stats()
    peak = np.nan if stats is None else float(stats.get("peak_bytes_in_use", np.nan))
    return Measurement(
        flops=_flops_of(compiled),
        argument_bytes=int(memory.argument_size_in_bytes),
        output_bytes=int(memory.output_size_in_bytes),
        temp_bytes=int(memory.temp_size_in_bytes),
        seconds=seconds,
        peak_bytes=peak,
    )
