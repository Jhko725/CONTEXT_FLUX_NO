---
type: glossary
status: current
updated: 2026-10-25
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
- **PDEFamily / `AbstractPDEFamily`** — a parametrized conservation law with a state layout and tiered capabilities (`simulations/pde/base.py`; tier 0's numerics methods arrive with `fv/`).
- **InitialCondition** — a function of space, `at_points(coords) -> [n_eqn, *shape]`, that each backend evaluates where its scheme needs it (`simulations/initial_conditions/base.py`); an **InitialConditionSampler** draws one with `sample(key, x_spans)`.
- **SolverBackend** — `solve(pde, ic, grid, t_out, bcs)` (`simulations/backends/base.py`): `PyClawBackend` (Classic · SharpClaw; landed), `TrixiBackend` (juliacall; Python side landed, Julia package pending), `JaxBackend` (pending).
- **Classic / SharpClaw** — PyClaw's two schemes: wave propagation (2nd order, the ICLR data) · WENO5 + SSP-RK method of lines (the higher-order reference).
- **PyClawSpec** — how a PDE class describes itself to PyClaw (Riemann solver, waves, kernel language, per-scheme options).
- **TrixiSpec** — how a PDE class describes itself to `ContextFluxTrixi` (equations constructor name, parameter kwargs).
- **Grid** — `Grid(x_spans, Nxs)`, uniform Cartesian cell centres shared by backends and the writer.
- **mutual convergence** — the backend-agreement criterion: the distance between two schemes' solutions shrinks under refinement (V18a/V18/V19).
- **reference backend** — PyClaw/SharpClaw or Trixi.jl, used as an oracle and as a fallback data generator (ADR-0004).
- **AnalyticSolution** — `(x, t, params) → u`, used as IC, boundary data and error reference.
- **run directory** — Hydra's output dir holding `config.yaml`, `metadata.json`, `checkpoints/` (and `.hydra/`, `wandb/`); the identity of a training run (ADR-0010).
- **RunRecord** — the typed `metadata.json`: locally generated `run_id`, provenance, index columns, outcome, resume notes; mirrored to orbax `custom_metadata` and wandb.
- **index columns** — the dozen `RunRecord.index` keys that distinguish runs (`runs.index()` columns; all that goes to `wandb.config`).
- **DatasetRecord** — `<store>.dataset.json` + root attrs written by `WellWriter`: id, family, grid, parameters, IC, backend, seed, provenance, written/failed counts.
- **dataset_id** — `<dataset>:<split>:<8 hex>` from `WellDatasetSourceBase`: a hash of the files' `DatasetRecord` ids when present, else the provisional metadata hash (ICLR-era stores).
- **Logger / Checkpointer** — the trainer's injected side effects (`training/loggers.py`, `training/checkpoint.py`); `DelayedLogger` is the one-step delay that keeps the host from waiting on the step just enqueued.
- **resume / warm start** — continue a run in place (`cfg.resume`, same `run_id`) vs start a new run from another run's state (`cfg.warm_start`, `resumed_from`).
- **golden** — a stored reference output of an ICLR checkpoint for the regression tier.
- **tier (tests)** — unit · property · regression · verification (ADR-0008).
- **track** — [C] Common · [M] MVP · [R] Rest (ADR-0009).
- **Well schema** — The Well dataset layout (`t0_fields`, `t1_fields`, …, `boundary_conditions` metadata) used for all datasets.
