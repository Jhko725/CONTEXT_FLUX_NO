"""Checkpointing of the whole `TrainerState` with Orbax (ADR-0010). Ported from
`deep_isochron.training.checkpoint`, against orbax-checkpoint ≥ 0.12.8's v1 API (0.12 is the
first version whose `load_pytree` accepts typed PRNG keys in the abstract tree; review round 1).

`OrbaxCheckpointer(directory, ...)` saves the **array leaves** of a `TrainerState` (model,
optimizer state, step, key) so a run can resume, and restores them into a template state
with `eqx.combine`. Which steps are saved is Orbax's `save_decision_policy` (default: every
`save_every` steps) and which are kept is its `preservation_policy` (default: the best
`keep_best` by `metric` plus the latest). The directory and `custom_metadata` are the
caller's (`runs.Run`); nothing here invents paths or metadata.

`NullCheckpointer` is for tests and notebooks."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast, TypeVar

import equinox as eqx
import jax
from orbax.checkpoint import v1 as ocp


S = TypeVar("S", bound=eqx.Module)


class Checkpointer(abc.ABC):
    """Base class: subclasses implement `save(step, state, metrics, force=False) -> saved`.

    `save` is *offered* a state at every evaluation step; the save policy decides whether to
    write it, and the return value says whether it did. `force=True` writes this call's
    checkpoint whatever the policy says — per call, not a mode: `Trainer.train` passes it
    exactly once, for the run's final step, so a run always ends with its last state on
    disk. Every checkpointer is a context manager whose exit calls `close()`."""

    @abc.abstractmethod
    def save(
        self,
        step: int,
        state: Any,
        metrics: Mapping[str, float],
        *,
        force: bool = False,
    ) -> bool: ...

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullCheckpointer(Checkpointer):
    def save(
        self,
        step: int,
        state: Any,
        metrics: Mapping[str, float],
        *,
        force: bool = False,
    ) -> bool:
        return False


class OrbaxCheckpointer(Checkpointer):
    """See the module docstring. `metric` is the key of `metrics` the best checkpoint is
    chosen by (smaller is better unless `maximize`); `None` keeps only the latest."""

    def __init__(
        self,
        directory: str | Path,
        *,
        save_every: int = 1,
        metric: str | None = "valid_loss",
        maximize: bool = False,
        keep_best: int = 1,
        custom_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        policies: list[Any] = [ocp.training.preservation_policies.LatestN(1)]
        if metric is not None:
            policies.append(
                ocp.training.preservation_policies.BestN(
                    # BestN keeps the *last* n of the sorted metrics, so ascending order
                    # (reverse=False) keeps the largest; invert for minimisation.
                    get_metric_fn=lambda m: m[metric],
                    reverse=not maximize,
                    n=keep_best,
                )
            )
        self.directory = Path(directory).resolve()  # orbax wants absolute paths
        self.metric = metric
        self.maximize = maximize
        self._ckptr = ocp.training.Checkpointer(
            str(self.directory),
            save_decision_policy=cast(
                ocp.training.save_decision_policies.SaveDecisionPolicy,
                ocp.training.save_decision_policies.FixedIntervalPolicy(save_every),
            ),
            preservation_policy=cast(
                ocp.training.preservation_policies.PreservationPolicy,
                ocp.training.preservation_policies.AnyPreservationPolicy(policies),
            ),
            custom_metadata=dict(custom_metadata) if custom_metadata else None,
        )

    def save(
        self,
        step: int,
        state: Any,
        metrics: Mapping[str, float],
        *,
        force: bool = False,
    ) -> bool:
        """Offer `state` at `step`: written if `step` is a multiple of `save_every`, or
        unconditionally with `force`. Returns whether a checkpoint was written."""
        if self.metric is not None and self.metric not in metrics:
            raise KeyError(
                f"checkpoint metric {self.metric!r} missing from metrics {sorted(metrics)}"
            )
        arrays = eqx.filter(state, eqx.is_array)
        json_metrics = cast(dict[str, Any], {k: float(v) for k, v in metrics.items()})
        return bool(self._ckptr.save(step, arrays, metrics=json_metrics, force=force))

    def restore(self, template: S, step: int | None = None) -> S:
        """The state at `step` (latest if `None`) with `template`'s static parts."""
        arrays, static = eqx.partition(template, eqx.is_array)
        abstract = jax.tree.map(
            lambda a: jax.ShapeDtypeStruct(
                a.shape, a.dtype, sharding=getattr(a, "sharding", None)
            ),
            arrays,
        )
        loaded = self._ckptr.load(step, abstract)
        return eqx.combine(loaded, static)

    @property
    def steps(self) -> list[int]:
        return sorted(c.step for c in self._ckptr.checkpoints)

    @property
    def latest_step(self) -> int | None:
        """The most recent saved step, `None` when the directory holds none."""
        latest = self._ckptr.latest
        return None if latest is None else int(latest.step)

    @property
    def best(self) -> tuple[int, float] | None:
        """`(step, metric)` of the best checkpoint still on disk by `metric`, `None` when
        there is no metric or no checkpoint carries it."""
        if self.metric is None:
            return None
        scored = [
            (int(c.step), float(c.metrics[self.metric]))
            for c in self._ckptr.checkpoints
            if c.metrics and self.metric in c.metrics
        ]
        if not scored:
            return None
        return (
            max(scored, key=lambda s: s[1])
            if self.maximize
            else min(scored, key=lambda s: s[1])
        )

    @property
    def custom_metadata(self) -> dict[str, Any]:
        """The `custom_metadata` the directory was created with (`{}` if none)."""
        raw = self._ckptr.root_metadata().custom_metadata
        meta = cast(Mapping[str, Any] | None, raw)
        return dict(meta) if meta else {}

    def close(self) -> None:
        self._ckptr.close()
