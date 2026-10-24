"""`Grid`: a uniform Cartesian cell-centred grid, the one geometry every backend and the
dataset writer agree on (`docs/design/fv-conventions.md` §1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class Grid:
    """`x_spans[d] = (lo, hi)` and `Nxs[d]` cells along axis `d`; cell centres at
    `lo + (i + ½) dx`. The Well's `dimensions/{x,y,z}` arrays are `centers`."""

    x_spans: tuple[tuple[float, float], ...]
    Nxs: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.x_spans) != len(self.Nxs):
            raise ValueError("x_spans and Nxs must have the same length.")
        if len(self.Nxs) > len(AXIS_NAMES):
            raise ValueError("at most three spatial dimensions are supported.")
        if any(n < 1 for n in self.Nxs):
            raise ValueError("every Nx must be >= 1.")
        object.__setattr__(
            self, "x_spans", tuple((float(a), float(b)) for a, b in self.x_spans)
        )
        object.__setattr__(self, "Nxs", tuple(int(n) for n in self.Nxs))

    @classmethod
    def from_sequences(
        cls, x_spans: Sequence[Sequence[float]], Nxs: Sequence[int]
    ) -> Grid:
        return cls(
            tuple((float(a), float(b)) for a, b in x_spans), tuple(int(n) for n in Nxs)
        )

    @property
    def n_spatial_dims(self) -> int:
        return len(self.Nxs)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.Nxs

    @property
    def axis_names(self) -> tuple[str, ...]:
        return AXIS_NAMES[: self.n_spatial_dims]

    @property
    def dxs(self) -> tuple[float, ...]:
        return tuple((hi - lo) / n for (lo, hi), n in zip(self.x_spans, self.Nxs))

    @property
    def centers(self) -> list[np.ndarray]:
        """Cell centres per axis, `float64`, `Nx` entries each: `lo + (i + ½) dx`, the same
        arithmetic as PyClaw's `Dimension.centers`, so the two agree bit for bit."""
        return [
            lo + (np.arange(n, dtype=float) + 0.5) * ((hi - lo) / float(n))
            for (lo, hi), n in zip(self.x_spans, self.Nxs)
        ]

    def meshgrid(self) -> list[np.ndarray]:
        return list(np.meshgrid(*self.centers, indexing="ij"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_spans": [list(s) for s in self.x_spans],
            "Nxs": list(self.Nxs),
            "dxs": list(self.dxs),
        }
