---
type: decision
id: ADR-0003
status: accepted
updated: 2026-09-26
verified_by: joon (discussed 2026-09-26/27)
---

# ADR-0003 — Boundary conditions are first-class objects applied by ghost-cell padding

**Status**: accepted (2026-09-26).

## Context

Periodicity was hard-coded (`wrap` padding everywhere; the only BC hook raised
`NotImplementedError`). Joon intends non-periodic problems soon. [cited] The standard FV treatment
is ghost cells filled per boundary type before every flux evaluation, with the interior scheme
unchanged (LeVeque 2002, Ch. 7; Clawpack `bc_lower`/`bc_upper`). Zero-order extrapolation into
ghost cells costs accuracy near the boundary; inverse Lax–Wendroff restores it (Tan & Shu, JCP 229, 2010).

## Decision

`fv/boundary.py` defines `BoundaryCondition.fill_ghost(u, n_ghost, *, axis, side, pde, params, t)`
with `Periodic`, `Extrapolate(order)`, `Reflective` (uses `PDEFamily.reflect`), `Dirichlet(value |
fn(x, t))`, `Inflow(primitive_state)`; later `NoSlipWall`, `AdiabaticWall`. `pad_ghost_cells` applies a
pair per axis before *every* flux evaluation, classical or learned. BCs travel with the data
(`DataSpec` ← Well metadata `boundary_conditions`) and are never inferred by the model.

## Rejected alternatives

- *Boundary flux imposed weakly* (DG-style, Trixi): natural for DG; for wide FV stencils ghost
  cells are needed anyway, and one mechanism for both classical and learned fluxes is simpler.
- *Learning boundary behaviour*: the in-context task is to infer coefficients, not boundary types;
  mixing them would confound the method's claim.

## Consequences

Learned fluxes become BC-aware for free. Coordinate channels become redundant for BC purposes,
which may resolve the `stack_grid` ambivalence. Near-boundary order with `Extrapolate(0)` is one;
the verification suite measures interior error separately or ILW is implemented before claiming
fifth order on non-periodic data.
