---
type: decision
id: ADR-0002
status: accepted
updated: 2026-09-26
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0002 — No per-dimension solver hierarchy; FD-WENO, unsplit dimension-by-dimension

**Status**: accepted (2026-09-26).

## Context

The code had separate 1D and 2D paths (`FluxNO` vs `NDFluxNO`, `GaussianRandomField1D/2D`, …).
On Cartesian grids the method-of-lines semi-discretisation is a sum over axes of 1D operators
applied with the other axes as batch dimensions. [cited] Finite-*difference* WENO keeps full order
in multi-D with plain dimension-by-dimension application on uniform grids, whereas finite-*volume*
WENO needs face quadrature to exceed second order (Shu, SIAM Review 51, 2009; Titarev & Toro, JCP
201, 2004).

## Decision

`fv/` is written once for N-D. Every function takes `axis`; the stepper evaluates all directional
fluxes on the same state and sums the divergences (unsplit, method of lines); transverse axes are
batch axes (`vmap`/`swapaxes`). The reconstruction is FD-WENO in Shu's formulation. Systems with
rotational invariance implement one normal-direction flux plus `rotate_to_axis`. Boundary
conditions are per (axis, side). No `Solver1D/2D/3D`; tests run at N = 1, 2 and (tiny grids) 3.

## Rejected alternatives

- *Sequential Strang/Godunov splitting*: second order at best and the scheme Roe (1991) criticised
  for oblique discontinuities; the project already moved away from it on the model side.
- *Genuinely multidimensional schemes* (CTU): dimension-specific; unnecessary at WENO5 + SSPRK3
  accuracy on periodic/uniform boxes.
- *FV-WENO*: needs dimension-specific face quadrature for order > 2.

## Consequences

FD-WENO requires a *uniform* grid (not periodicity). If non-uniform or curvilinear grids ever
arrive, the reconstruction layer must be revisited. Recorded so it is not rediscovered.
