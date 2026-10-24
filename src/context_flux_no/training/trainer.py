"""The training loop (ADR-0010).

`TrainerState` holds everything a run needs to continue — model, optimizer state, step and
PRNG key — and is what gets checkpointed. `Trainer(optimizer, loss_fn)` owns the JIT-compiled
`train_step` and `valid_step` and the loop; it has **no side effects of its own**: logging,
checkpointing and evaluation are injected (`training.loggers`, `training.checkpoint`), so a
test runs the loop with `ListLogger()` and a temporary `OrbaxCheckpointer`, and a notebook
can run it with anything. The experiment script (`scripts/training/train_multiphysics.py`)
assembles the run directory (`runs.Run`), the wandb run (`runs.wandb_mirror`) and
the loaders around it.

Design, inspired by Levanter's `TrainerState` and `deep_isochron.training.trainer`:

- the loop never reads a device value: metrics go to the logger as JAX scalars and the
  `DelayedLogger` converts the *previous* step's, so the host does not wait on the step it
  just enqueued;
- validation runs every `eval_every` steps through `evaluate(model) -> metrics`; the result
  is logged and offered to the checkpointer;
- the final state is always evaluated and checkpointed (`force=True`), so `Run.restore()`
  with no step is the trained state whatever `save_every` is;
- `mesh` is stored for a later multi-device step (plan §10.6) and unused.
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from functools import cached_property
from typing import Any, Self, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PRNGKeyArray, PyTree

from context_flux_no.custom_types import FilterSpec, IntScalar
from context_flux_no.training.checkpoint import Checkpointer, NullCheckpointer
from context_flux_no.training.loggers import Logger, NullLogger
from context_flux_no.training.loss import AbstractTrainLoss


Batch = PyTree
Metrics = Mapping[str, Any]
M = TypeVar("M", bound=eqx.Module)


class TrainerState[M: eqx.Module](eqx.Module):
    step: IntScalar
    model: M
    opt_state: optax.OptState
    training_key: PRNGKeyArray

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    is_trainable: FilterSpec = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        model: M,
        optimizer: optax.GradientTransformation,
        is_trainable: FilterSpec = eqx.is_inexact_array,
        *,
        key: PRNGKeyArray,
    ):
        # Making this the __init__ would clash with the use of replace()
        opt_state = optimizer.init(
            eqx.filter(model, is_trainable)  # ty: ignore[invalid-argument-type]
        )
        return cls(
            step=jnp.asarray(0, dtype=int),
            model=model,
            opt_state=opt_state,
            training_key=key,
            optimizer=optimizer,
            is_trainable=is_trainable,
        )

    def take_step(self, grads: M) -> Self:
        """Given a pytree of model gradients, update model parameters using the
        appropriate optimizer update function."""
        grads = self.filter_trainable(grads)
        updates, opt_state_next = self.optimizer.update(
            grads,  # ty: ignore[invalid-argument-type]
            self.opt_state,
            self.filter_trainable(self.model),  # ty: ignore[invalid-argument-type]
        )
        model_next = eqx.apply_updates(self.model, updates)
        return replace(
            self,
            step=self.step + 1,
            model=model_next,
            opt_state=opt_state_next,
            training_key=jax.random.fold_in(self.training_key, self.step),
        )

    def filter_trainable(self, pytree: M) -> M:
        """Filter the given pytree to retain the trainable model parameters, as
        determined by self.is_trainable. Works on all pytrees compatible with self.model."""
        return eqx.filter(pytree, self.is_trainable)


class Trainer:
    """`Trainer(optimizer, loss_fn)`; see the module docstring. `loss_args` is whatever the
    loss needs besides the batch (today `(dt, *dx)`); it is static for a run."""

    def __init__(
        self,
        optimizer: optax.GradientTransformation,
        loss_fn: AbstractTrainLoss,
        *,
        loss_args: Any = None,
        is_trainable: FilterSpec = eqx.is_inexact_array,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_args = loss_args
        self.is_trainable = is_trainable
        self.mesh = mesh  # reserved for multi-device training; unused (plan §10.6)

    # ------------------------------------------------------------------ state ----
    def init(self, model: M, *, key: PRNGKeyArray) -> TrainerState[M]:
        return TrainerState[M].init(model, self.optimizer, self.is_trainable, key=key)

    # ------------------------------------------------------------------ steps ----
    @cached_property
    def train_step(
        self,
    ) -> Callable[[TrainerState[M], Batch], tuple[TrainerState[M], Metrics]]:
        return eqx.filter_jit(self._train_step)

    def _train_step(
        self, state: TrainerState[M], batch: Batch
    ) -> tuple[TrainerState[M], Metrics]:
        model = eqx.nn.inference_mode(state.model, False)
        loss_grad_fn = eqx.filter_value_and_grad(self.loss_fn, has_aux=True)
        (loss, metrics), grads = loss_grad_fn(
            model, batch, self.loss_args, state.training_key
        )
        return state.take_step(grads), {"train_loss": loss, **metrics}

    @cached_property
    def valid_step(self) -> Callable[[TrainerState[M], Batch], Metrics]:
        return eqx.filter_jit(self._valid_step)

    def _valid_step(self, state: TrainerState[M], batch: Batch) -> Metrics:
        model = eqx.nn.inference_mode(state.model, True)
        loss, metrics = self.loss_fn(model, batch, self.loss_args, state.training_key)
        return {"valid_loss": loss, **metrics}

    def evaluator(
        self, valid_loader: Iterable[Batch]
    ) -> Callable[[TrainerState[M]], Metrics]:
        """An `evaluate(state)` that runs `valid_step` on the next batch of
        `valid_loader` — the one-batch validation the scripts have always done. A more
        thorough evaluator (Phase 6) is any callable with the same signature."""
        batches = iter(valid_loader)

        def evaluate(state: TrainerState[M]) -> Metrics:
            return self.valid_step(state, next(batches))

        return evaluate

    # ------------------------------------------------------------------ loop -----
    def train(
        self,
        model_or_state: M | TrainerState[M],
        train_loader: Iterable[Batch],
        *,
        num_steps: int,
        key: PRNGKeyArray | None = None,
        logger: Logger | None = None,
        checkpointer: Checkpointer | None = None,
        evaluate: Callable[[TrainerState[M]], Metrics] | None = None,
        eval_every: int = 1,
        stop_on_nonfinite: bool = True,
    ) -> TrainerState[M]:
        """Run `num_steps` steps (an upper bound: a loader that ends first stops the run
        with a warning). `evaluate(state)` returns validation metrics every `eval_every`
        steps; they are logged and offered to `checkpointer.save`, and the final step is
        evaluated and checkpointed regardless of the save policy. Resume by passing a
        restored `TrainerState`. With `stop_on_nonfinite`, a non-finite validation metric
        stops the run with `FloatingPointError` *before* that state is checkpointed, so the
        last checkpoint on disk is finite. The logger and checkpointer are entered as
        context managers, so they close — and the `DelayedLogger` flushes — even if a step
        raises."""
        if isinstance(model_or_state, TrainerState):
            state = model_or_state
        else:
            if key is None:
                raise ValueError("key is required when starting from a model.")
            state = self.init(model_or_state, key=key)
        if eval_every < 1:
            raise ValueError("eval_every must be a positive integer.")
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative.")
        log: Logger = NullLogger() if logger is None else logger
        ckpt: Checkpointer = (
            NullCheckpointer() if checkpointer is None else checkpointer
        )

        saved_step = int(state.step) if isinstance(model_or_state, TrainerState) else -1
        last_eval: tuple[int, dict[str, float]] | None = None

        def evaluate_and_save(step: int, *, force: bool = False) -> None:
            nonlocal saved_step, last_eval
            assert evaluate is not None
            val_metrics = {k: float(v) for k, v in evaluate(state).items()}
            last_eval = (step, val_metrics)
            log.log(val_metrics, step)
            bad = sorted(k for k, v in val_metrics.items() if not math.isfinite(v))
            if stop_on_nonfinite and bad:
                raise FloatingPointError(
                    f"non-finite validation metrics at step {step}: {bad}; the state was "
                    f"not checkpointed (last checkpoint: step {saved_step})."
                )
            if ckpt.save(step, state, val_metrics, force=force):
                saved_step = step

        batches = iter(train_loader)
        step = int(state.step)  # Python-side counter: the device value is never read
        with log, ckpt:
            for i in itertools.count():
                if i >= num_steps:
                    break
                try:
                    batch = next(batches)
                except StopIteration:
                    warnings.warn(
                        f"the loader ended after {i} of {num_steps} steps.",
                        stacklevel=2,
                    )
                    break
                state, metrics = self.train_step(state, batch)
                step += 1
                log.log(metrics, step)
                if evaluate is not None and step % eval_every == 0:
                    evaluate_and_save(step)
            # the final state is always evaluated and checkpointed, unless the loop's
            # last step already did both
            if step != saved_step:
                if evaluate is None:
                    ckpt.save(step, state, {}, force=True)
                elif last_eval is not None and last_eval[0] == step:
                    ckpt.save(
                        step, state, last_eval[1], force=True
                    )  # already evaluated
                else:
                    evaluate_and_save(step, force=True)
        return state
