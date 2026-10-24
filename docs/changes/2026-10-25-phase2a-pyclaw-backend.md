---
type: change
status: planned
updated: 2026-10-25
branch: phase2a-pyclaw-backend
roadmap: Phase 2a (PyClaw half), Phase 4.3
base: phase5-run-identity
---

# 2026-10-25 — `phase2a-pyclaw-backend`

Roadmap 2a, PyClaw half, plus 4.3 (`DatasetRecord` + backend-agnostic writer), which 2a needs
because a backend has to write *something*. The Trixi half of 2a is **not** on this branch: Julia
cannot be installed in Claude's container (download hosts are blocked), so `reference/trixi/` and
`TrixiBackend` wait for Joon's machine or the cluster; §Trixi hand-off below lists what they need.

**Base.** `phase5-run-identity` (for `provenance.py` and the record pattern). Merge order:
`phase1-prune` → `phase5-run-identity` → this. `check_docs.py --files --base phase5-run-identity`.

**Independent of the regression gate.** Nothing here touches a model or a loader; the data
generation path is refactored behind a backend with the Classic scheme kept bit-identical (a test
pins it), so existing datasets stay reproducible until Phase 4.4 regenerates them deliberately.

## Ledger (to confirm before execution)

| # | Item | Done when |
|---|---|---|
| 1 | `sim/` package: `sim/grid.py::Grid(x_spans, Nxs)` (cell centres, `dx`, `n_spatial_dims`), `sim/backend.py::SolverBackend` ABC with `solve(pde, u0, grid, t_out, bcs, *, params) -> u[time, n_eqn, *Nxs]` and `solve_batch` (loop, `vmap` later), `BCName = Literal["periodic", "extrap", "wall"]` per axis-side (strings until ADR-0003's objects exist in 2b; recorded in metadata) | imports; `SolverBackend` has one concrete subclass |
| 2 | `sim/pyclaw.py::PyClawBackend(scheme="classic" \| "sharpclaw", *, cfl=0.5, cfl_max=0.9, limiter="MC", weno_order=5, time_integrator="SSP104", transverse_waves=2)`; folds `simulations/pdesolve.py` + `pyclaw_utils.py` (deleted). Needs from a PDE class a `PyClawSpec(riemann_solver, num_waves, kernel_language, problem_data_fn, fwave)` — added as a `pyclaw_spec` property on the six existing PDE classes; their `solve()` methods become one-line delegations to `PyClawBackend("classic")` with the old signature kept for `generate_dataset` | **bit-identical test**: every family, fixed IC and params, `solve()` before vs after on a 64-cell grid (goldens made from `main` in this branch, stored as small `.npz` under `tests/unit/goldens_sim/`) |
| 3 | SharpClaw per family: Python-kernel Riemann solvers (cubic, sine, parametrised SW) go through `SharpClawSolver1D` with `lim_type=2, weno_order=5, char_decomp=0`; built-in Fortran kernels (`riemann.burgers_1D`, `riemann.euler_4wave_2D`) use `kernel_language="Fortran"`; 2D via `SharpClawSolver2D` | each family runs one step under both schemes without error |
| 4 | **Backend-agreement test** (V18's PyClaw-internal precursor; new row V18a): Classic vs SharpClaw, smooth IC, short horizon, `N ∈ {64, 128, 256}` — the L1 difference must shrink by ≥ 1.5× per refinement (mutual convergence, ADR-0004 §Consequences). 1D families in `tests/unit` behind `pytest.importorskip("clawpack")` (seconds); 2D Euler/cubic in `tests/verification` (`-m slow`) | green; measured ratios recorded in `verification.md` |
| 5 | `sim/record.py::DatasetRecord` (mirror of `runs.RunRecord`): `dataset_id` (`<family>-<split>-<utc stamp>-<8 hex>`), `family`, `split`, `n_trajectories`, `grid` (spans, Nxs, dx), `t_out` (span, Nt, dt), `bcs`, `params` (names, ranges, sampling strategy), `ic` (waveform name + kwargs), `backend` (name, scheme, options, `clawpack.__version__`), `seed`, `git {sha, dirty}`, `package_version`, `created_at`, `status` (`writing` → `complete`, `n_failed`), `schema_version`; `to_json/from_json`, Hypothesis round trip | tests green |
| 6 | `sim/writer.py::WellWriter` — `simulations/dataset.py::ZarrWellDataset` moved and extended: `boundary_conditions` group per The Well's layout (one subgroup per axis-side with `bc_type`, `associated_dims`, `associated_fields`, `sample_varying`, `time_varying` attrs; **to verify against the Well format doc at implementation**), `dataset.json` (= `DatasetRecord`) beside the store and in root attrs, `open()` for append/inspection, `finalize(n_written)` replaces `resize_dataset`; `write_trajectory(idx, u, scalars)` takes `[time, n_eqn, *Nxs]` + the PDE's `solution_to_well_schema` | Well schema round trip through `ZarrWellDatasetSource` in a test; `WellDatasetSourceBase.dataset_id` returns `DatasetRecord.id` when `dataset.json` exists, else the provisional hash |
| 7 | `sim/generate.py::generate_dataset(..., backend: SolverBackend)` — `simulations/utils.py` moved; per-trajectory failures counted into the record instead of silently dropped; `scripts/data/generate_dataset.py` gains a `backend` config group (`configs/backend/{classic,sharpclaw}.yaml`, default **classic**) | one tiny end-to-end run (cubic 1D, 2 coefficients × 2 ICs, N = 32) in a test, loaded back with `ZarrWellDatasetSource` |
| 8 | `simulations/` → `sim/`: `simulations/pde/*` stays where it is until Phase 4.1 (`PDEFamily`) but is re-exported as `sim.pde`; `simulations/{dataset,utils,pdesolve,pyclaw_utils}.py` deleted; `scripts/import_all.py` skip list updated (clawpack now builds here with `apt-get install gfortran`, note added to README) | `import_all` clean; `check_configs` unchanged |
| 9 | Docs: ADR-0011 (reference backends as a `SolverBackend` protocol + dataset provenance; why strings-for-BCs now), `design/sim-backends.md` (how to add a backend / a family; SharpClaw options; agreement-test protocol), `verification.md` V18a + the measured ratios, roadmap (2a split into 2a-pyclaw done / 2a-trixi pending; 4.3 done), architecture, glossary, index, this document's Files/Design/Tests/Review-notes | `check_docs --files --base phase5-run-identity` clean |

Not in this ledger (deliberately): BC objects and ghost-cell padding (2b — Joon), `PDEFamily`
(4.1), regenerating any dataset (4.4), `JaxBackend` (4.7), Trixi (below).

## Trixi hand-off (for Joon's machine; not executed here)

What `reference/trixi/` + `sim/trixi.py::TrixiBackend` need, so it can be started independently:

1. Julia ≥ 1.10 via `juliaup`; `reference/trixi/Project.toml` + `Manifest.toml` pinning Trixi.jl,
   OrdinaryDiffEq.jl (SSPRK43 or CarpenterKennedy2N54), HDF5.jl, JSON3.jl.
2. One elixir per family under `reference/trixi/elixirs/`: `TreeMesh` 1D/2D periodic,
   `DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)` with a shock-capturing volume integral
   where the family shocks (`VolumeIntegralShockCapturingHG`), `initial_condition(x, t, equations)`
   reading `u0` from the HDF5 input by nearest cell, output projected to cell averages on the
   requested grid (`Trixi.integrate` per element or `PlotData` sampling — decide in the elixir).
3. The protocol: `TrixiBackend.solve` writes `{u0.h5, request.json}` to a temp dir, runs
   `julia --project=reference/trixi run.jl request.json`, reads `u.h5` back. `request.json` =
   `{family, params, grid, t_out, bcs, polydeg}`.
4. The scalar families (cubic, sine) need custom `AbstractEquations` in the elixir (Trixi ships
   `LinearScalarAdvection`, `Burgers`, `ShallowWater`, `CompressibleEuler`; a scalar law with
   user `flux(u)` is ~30 lines following `BurgersEquation1D`).
5. V18 then runs against `PyClawBackend("sharpclaw")` with the same mutual-convergence protocol as
   V18a.

## Files

*(planned; filled in as the work lands)*

## Design

*(planned)*

## Bugs fixed

*(planned)*

## Findings

*(planned)*

## Tests

*(planned)*

## Open issues

*(planned)*

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| *(rows added per file as the work lands)* | | |
