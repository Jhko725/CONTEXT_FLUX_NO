---
type: decision
id: ADR-0011
status: accepted
updated: 2026-10-25
verified_by: joon (ledger confirmed 2026-10-25, "Go ahead with 2a")
sources: ADR-0004; Clawpack/PyClaw docs and source (5.14.0); The Well data format (the_well/docs/data_format.md); ADR-0010 for the record pattern
---

# ADR-0011 — Reference solvers behind one `SolverBackend`; every store carries a `DatasetRecord`

**Status**: accepted (2026-10-25). Implemented on `phase2a-pyclaw-backend` (PyClaw half);
the Trixi backend follows the same interface when it lands.

## Context

ADR-0004 decided that two external codes — PyClaw (Classic and SharpClaw) and Trixi.jl — are
built *before* the in-house WENO5, as oracles for cross-code verification and as fallback data
generators. At `iclr2027-submission` each PDE class built its own PyClaw solver inside its
`solve()` method with its own defaults, two classes used their own JAX steppers, five of the
seven classes did not implement the abstract base the dataset writer required (so the data
script only ran for two families), and a generated store recorded nothing about how it was
made — not the solver, not the boundary conditions, not the parameter ranges, not the code
version (pain point P9 on the data side; findings F7/F8).

## Decision

1. **One `SolverBackend` interface** [code]: `solve(pde, u0[n_eqn, *N], grid, t_out, bcs) →
   u[time, n_eqn, *N]`. The backend knows nothing about files or sampling. `Grid` is the
   shared geometry (uniform Cartesian, cell centres computed with PyClaw's arithmetic).
   Boundary conditions are the three names every backend can honour — `periodic`, `extrap`,
   `wall` — per axis and side, until `fv/boundary.py` (Phase 2b) provides objects; the names
   are what gets recorded.

2. **`PyClawBackend(scheme)`** [code] with `scheme ∈ {classic, sharpclaw}`; a PDE class
   describes itself to PyClaw through a `PyClawSpec` (Riemann solver, number of waves, kernel
   language, per-scheme solver options). The **Classic path is bit-identical** to the old
   `solve()` methods (goldens made at `1ff31bb`), so existing datasets stay reproducible until
   Phase 4.4 regenerates them on purpose. PyClaw runs only Fortran kernels in 2-D [code], so
   the Python-kernel families are 1-D on PyClaw; 2-D cubic keeps its JAX Rusanov stepper (F8)
   until Phase 4.

3. **`AbstractPDEFamily` starts now and grows** [code]. All seven classes subclass it with
   `n_spatial_dims`, `field_rank_names`, `parameters`, optional `pyclaw_spec`, and a default
   `solve()` that runs a backend — the data-facing part of ADR-0001's tier 0. The numerics
   part (`flux`, `max_wave_speed`, `reflect`) is added to the same class as `fv/` consumes it
   (ADR-0001 amendment, 2026-10-25); Phase 4.1 is the conformance check, not a rewrite.

4. **Backend agreement by mutual convergence** [code]: V18a (Classic vs SharpClaw) requires
   the L1 distance between the two solutions on a shared grid to shrink by ≥ 1.5× per
   refinement on a smooth horizon. Measured ratios are ≈ 4 (second-order dominated), recorded
   in `verification.md`. V18 (PyClaw vs Trixi) and V19 (own WENO5) reuse the protocol.

5. **Every store carries a `DatasetRecord`** [code]: `<store>.dataset.json` beside the zarr
   and the same dict in the root attributes — locally generated `id`, family, split, grid,
   time grid, boundary conditions, parameter ranges and sampling, initial-condition config,
   `backend.describe()` (name, scheme, Clawpack version, overrides), seed, git SHA/dirty,
   package version, `n_trajectories` written and `n_failed`. `WellDatasetSourceBase.dataset_id`
   hashes the record ids when every file has one, so `RunRecord.index.dataset_id` identifies
   the data a model was trained on across machines.

6. **The writer follows The Well's specification** [cited: `the_well/docs/data_format.md`],
   now including the `boundary_conditions` group (one subgroup per axis-side with `bc_type ∈
   {periodic, wall, open}`, `associated_dims`, `associated_fields`, `sample_varying`,
   `time_varying`, and a boolean `mask`), so The Well's own reader and ours agree.

## Rejected alternatives

- **Backend chosen inside each PDE class** (as before). Hides the solver from the record and
  duplicates the controller setup seven times; a backend argument with a per-class default
  keeps the old behaviour as the default and makes the scheme a one-key config change.
- **A throw-away interim base, replaced wholesale in 4.1** (the first draft of this ADR). Joon's
  review preferred growing one class: the base already carried half of tier 0, so a 4.1 rewrite
  would have done it twice. Hence `AbstractPDEFamily` now, numerics methods added as `fv/` needs
  them (ADR-0001 amendment).
- **BC objects now.** ADR-0003's objects belong with ghost-cell padding in `fv/`; strings are
  what PyClaw and Trixi consume anyway, and the record stores strings regardless.
- **A fixed tolerance for backend agreement.** Two schemes of different order and dissipation
  do not agree to a fixed number near shocks; a convergence ratio is the honest criterion
  (ADR-0004 §Consequences).
- **Content hash as the dataset id.** A locally generated id plus the record's full
  description is enough to tell datasets apart and does not require reading terabytes;
  a content digest can be added to `extra` later if needed.

## Consequences

- `scripts/data/generate_dataset.py` works for every family again (two bug fixes on the way:
  `GaussianRandomField` defaults, `ViscousBurgers1D.solve` signature + F8 last frame) and
  takes `backend={default,classic,sharpclaw}`.
- `simulations/{dataset,utils,pdesolve,pyclaw_utils}.py` are gone; `simulations/` is
  `grid`, `backend`, `pyclaw`, `record`, `writer`, `generate`, `pde/`. The Well layout split
  (`solution_to_well_schema`) moved from the family to the writer, which is created with the
  layout: the family states its layout once, the store owns how it is laid out on disk.
- Julia/Trixi enters through the same interface (`simulations/trixi.py`) as
  `docs/design/python-julia-repo.md` describes; nothing in this decision depends on it.
- Phase 4.4 regeneration picks a backend per family in its config and the choice is in every
  `dataset.json`.
