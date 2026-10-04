---
type: glossary
status: current
updated: 2026-10-04
sources: [claude, joon]
---

# Glossary

One line per term; `architecture.md` and the design documents use only these.

- **cell** — a finite-volume control volume; `u_i` is the cell average (treated as a point value in FD-WENO).
- **face** — the interface `i + ½` between cells `i` and `i+1`; face arrays have `N + 1` entries along their axis.
- **stencil `(a, b)`** — face `i+½` sees cells `i−a … i+b`.
- **ghost cells** — padding cells outside the domain filled by a `BoundaryCondition`; the only place BCs enter.
- **axis** — a spatial dimension index; every `fv/` function takes one (ADR-0002).
- **FluxPredictor** — anything mapping a padded state to `N + 1` face fluxes along an axis; declares `stencil` and `requires`.
- **NumericalFlux** — a `FluxPredictor` composed of `reconstruction`, `base`, `dissipation` (ADR-0005).
- **reconstruction** — stencil → face states `(u_L, u_R)` (or stencil features, for `Fused`).
- **base flux** — `(u_L, u_R) → F`, the non-dissipative or Riemann part.
- **dissipation** — `(u_L, u_R) → D`, subtracted from the base flux (Lax–Friedrichs, learned, none).
- **requires / capabilities** — the `PDEFamily` methods a flux needs / provides; checked at build time (ADR-0001).
- **tier 0/1/2/3** — groups of `PDEFamily` capabilities: required · wave speeds & eigensystem · Riemann flux · viscous flux.
- **RHS** — the structured right-hand side `(hyperbolic, parabolic, source)` handed to an `Integrator`.
- **Integrator** — owns `Δt` policy and stages; `ForwardEuler`, `SSPRK2/3`, later `CFLAdaptive`, IMEX.
- **FluxStepper** — `(flux, integrator, source)`; the target of an `InContextFluxOperator`.
- **encoder / conditioner / target** — the three slots of `InContextOperator` (ADR-0006).
- **Scaled** — the RevIN-style scale wrapper around the conditioned module (ADR-0006).
- **DataSpec** — shapes, channels, `dt`, `dx`, `bcs` read off a dataset; feeds `build()`.
- **PDEFamily** — a parametrised conservation law with a state layout and tiered capabilities.
- **InitialCondition** — a sampler producing primitive fields per field name, valid for the family.
- **SolverBackend** — `solve(pde, u0_batch, grid, t_out, bcs)`: `JaxBackend`, `PyClawBackend`, `TrixiBackend`.
- **reference backend** — PyClaw/SharpClaw or Trixi.jl, used as an oracle and as a fallback data generator (ADR-0004).
- **AnalyticSolution** — `(x, t, params) → u`, used as IC, boundary data and error reference.
- **RunRecord / DatasetRecord** — provenance objects written by the trainer / dataset writer (Phase 5 / 4).
- **golden** — a stored reference output of an ICLR checkpoint for the regression tier.
- **tier (tests)** — unit · property · regression · verification (ADR-0008).
- **track** — [C] Common · [M] MVP · [R] Rest (ADR-0009).
- **Well schema** — The Well dataset layout (`t0_fields`, `t1_fields`, …, `boundary_conditions` metadata) used for all datasets.
