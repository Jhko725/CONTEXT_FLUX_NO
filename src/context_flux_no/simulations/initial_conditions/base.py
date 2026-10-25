"""The initial-condition contract (ADR-0011 §Initial conditions, 2026-10-25).

An initial condition is a *function* of space. Each backend discretizes it the way its scheme
needs — PyClaw at cell centers, Trixi's DG at the Lobatto–Legendre nodes, the own FD-WENO at
cell centers — by calling `at_points` with per-axis coordinate arrays already broadcast to
the evaluation shape (`Grid.meshgrid()` for a grid; `[n_elements, n_nodes]` arrays for DG
nodes). Nothing in the pipeline passes an array of initial values around: a sampler returns an
`InitialCondition` object, `AbstractPDEFamily.solve` evaluates it, the writer stores what the
solver advanced.

`InitialConditionSampler.sample(key, x_spans)` draws one such function; the domain is an
argument so that positions (step locations, Voronoi seeds) are drawn in physical coordinates
and the drawn field does not depend on the grid it is later evaluated on."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from jaxtyping import Float, PRNGKeyArray


Coords = Sequence[Float[np.ndarray, "*shape"]]  # one array per spatial axis, same shape


class InitialCondition(abc.ABC):
    """A function `x ↦ u0(x)` with `n_eqn` components on `n_spatial_dims` dimensions."""

    @property
    @abc.abstractmethod
    def n_eqn(self) -> int: ...

    @property
    @abc.abstractmethod
    def n_spatial_dims(self) -> int: ...

    @abc.abstractmethod
    def at_points(self, coords: Coords) -> Float[np.ndarray, "n_eqn *shape"]:
        """Values at the points whose per-axis coordinates are `coords` (all of one shape)."""

    def check_coords(self, coords: Coords) -> tuple[np.ndarray, ...]:
        arrays = tuple(np.asarray(c, dtype=float) for c in coords)
        if len(arrays) != self.n_spatial_dims:
            raise ValueError(
                f"{type(self).__name__} is {self.n_spatial_dims}-D, got {len(arrays)} "
                "coordinate arrays"
            )
        shapes = {a.shape for a in arrays}
        if len(shapes) != 1:
            raise ValueError(f"coordinate arrays must share one shape, got {shapes}")
        return arrays


class FromFunction(InitialCondition):
    """`InitialCondition` from a plain function of per-axis coordinates returning
    `[n_eqn, *shape]` — analytic initial data, test cases, manufactured solutions."""

    def __init__(
        self,
        fn: Callable[[Coords], Float[np.ndarray, "n_eqn *shape"]],
        *,
        n_eqn: int,
        n_spatial_dims: int,
    ) -> None:
        self.fn = fn
        self._n_eqn = int(n_eqn)
        self._n_spatial_dims = int(n_spatial_dims)

    @property
    def n_eqn(self) -> int:
        return self._n_eqn

    @property
    def n_spatial_dims(self) -> int:
        return self._n_spatial_dims

    def at_points(self, coords: Coords) -> Float[np.ndarray, "n_eqn *shape"]:
        arrays = self.check_coords(coords)
        out = np.asarray(self.fn(arrays), dtype=float)
        if out.shape != (self.n_eqn, *arrays[0].shape):
            raise ValueError(
                f"initial condition returned {out.shape}, expected "
                f"{(self.n_eqn, *arrays[0].shape)}"
            )
        return out


class Stacked(InitialCondition):
    """Channels of several initial conditions on the same domain, in order."""

    def __init__(self, parts: Sequence[InitialCondition]) -> None:
        self.parts = tuple(parts)
        dims = {p.n_spatial_dims for p in self.parts}
        if len(self.parts) == 0 or len(dims) != 1:
            raise ValueError("Stacked needs >= 1 part, all of one spatial dimension")

    @property
    def n_eqn(self) -> int:
        return sum(p.n_eqn for p in self.parts)

    @property
    def n_spatial_dims(self) -> int:
        return self.parts[0].n_spatial_dims

    def at_points(self, coords: Coords) -> Float[np.ndarray, "n_eqn *shape"]:
        return np.concatenate([p.at_points(coords) for p in self.parts], axis=0)


class Conserved(InitialCondition):
    """A primitive-variable initial condition converted with the family's `to_conserved`
    (e.g. `(rho, u, v, p)` → `(rho, rho u, rho v, E)` for `Euler2D`)."""

    def __init__(self, primitive: InitialCondition, pde: Any) -> None:
        if not hasattr(pde, "to_conserved"):
            raise TypeError(f"{type(pde).__name__} has no primitive-variable form")
        self.primitive = primitive
        self.pde = pde

    @property
    def n_eqn(self) -> int:
        return self.pde.n_eqns

    @property
    def n_spatial_dims(self) -> int:
        return self.primitive.n_spatial_dims

    def at_points(self, coords: Coords) -> Float[np.ndarray, "n_eqn *shape"]:
        return np.asarray(self.pde.to_conserved(*self.primitive.at_points(coords)))


class InitialConditionSampler(abc.ABC):
    """Draws an `InitialCondition` on the domain `x_spans` from a PRNG key."""

    @property
    @abc.abstractmethod
    def n_eqn(self) -> int: ...

    @property
    @abc.abstractmethod
    def n_spatial_dims(self) -> int: ...

    @abc.abstractmethod
    def sample(
        self, key: PRNGKeyArray, x_spans: Sequence[Sequence[float]]
    ) -> InitialCondition: ...


class Multichannel(InitialConditionSampler):
    """One sampler per channel, same domain; `sample` returns a `Stacked` condition."""

    def __init__(self, samplers: Sequence[InitialConditionSampler]) -> None:
        self.samplers = tuple(samplers)
        if len({s.n_spatial_dims for s in self.samplers}) != 1:
            raise ValueError("Multichannel samplers must share one spatial dimension")

    @property
    def n_eqn(self) -> int:
        return sum(s.n_eqn for s in self.samplers)

    @property
    def n_spatial_dims(self) -> int:
        return self.samplers[0].n_spatial_dims

    def sample(self, key: PRNGKeyArray, x_spans: Sequence[Sequence[float]]) -> Stacked:
        import jax

        keys = jax.random.split(key, len(self.samplers))
        return Stacked([s.sample(k, x_spans) for s, k in zip(self.samplers, keys)])
