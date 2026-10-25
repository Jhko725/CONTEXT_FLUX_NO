---
type: change
status: in-progress
updated: 2026-10-25
branch: phase2a-trixi-backend
roadmap: Phase 2a (Trixi half); the InitialCondition contract
base: phase2a-pyclaw-backend
---

# 2026-10-25 — `phase2a-trixi-backend`

The Trixi.jl half of roadmap 2a, designed by interview on 2026-10-25 (decisions below), and
the contract change it needs: initial conditions become *functions* that each backend
discretizes itself. The Python side (contract, `trixi_spec`s, `simulations/trixi.py`, tests
that skip without Julia, documents) is done in Claude's cloud session; the Julia package
`reference/trixi/ContextFluxTrixi/` and the first runs are done in a Claude Code CLI session
on Joon's laptop, where Julia can run. Steps for that session are in §Laptop steps.

**Base.** `phase2a-pyclaw-backend` (merge order: phase1 → phase5 → 2a-pyclaw → this).
`check_docs.py --files --base phase2a-pyclaw-backend`.

## Decisions (grill of 2026-10-25; Design section carries the rejected alternatives)

| # | Decision |
|---|---|
| D-IC | An initial condition is an `InitialCondition` object with `at_points(coords) -> [n_eqn, *shape]` (per-axis coordinate arrays, one shape); samplers return one via `sample(key, x_spans)`; `AbstractPDEFamily.solve(ic, …)` evaluates it at the cell centers; no array-backed adapter (no adopter). **ADR-0011 §Initial conditions.** |
| D-TX | Oracle = Trixi DGSEM `polydeg=3` on `StructuredMesh` (one element per FV cell), surface flux Lax–Friedrichs, Hennemann–Gassner shock capturing; output = element means by LGL quadrature; time stepping SSPRK33 at a fixed CFL via `StepsizeCallback`, `tstops = saveat = t_out`. |
| D-FAM | Six families (five PyClaw ones + `CubicFlux2D`); `ViscousBurgers1D` waits for tier 3. Custom Trixi equations for cubic 1D/2D, sine, parametrized shallow water; Trixi's `InviscidBurgersEquation1D` and `CompressibleEulerEquations2D`. |
| D-BC | Periodic only; `extrap`/`wall` raise `ValueError` at `solve` (V17 adds Trixi's BCs). |
| D-JL | Julia `~1.12` via `juliapkg.json` (Trixi's compat is `julia = "1.10"` = `[1.10, 2)`, CI on 1.10/1.11/1.12); `reference/trixi/Project.toml` + committed `Manifest.toml` with `ContextFluxTrixi` `dev`-ed in; `PYTHON_JULIACALL_EXE`/`_PROJECT` point juliacall at it. |
| D-TR | Transport = juliacall in-process, Python evaluates the IC at the LGL nodes Julia reports and passes arrays both ways (no Python-from-Julia calls); `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`, `_THREADS=auto`, set before import in `simulations/trixi.py`; no `_jl_call_nogil`; the subprocess transport only if juliacall proves unworkable. **ADR-0004 amendment.** |
| D-SPEC | `trixi_spec` on each family (equations constructor name + parameter kwargs), parallel to `pyclaw_spec`. |
| D-API | `TrixiBackend(polydeg=3, cfl=0.5, surface_flux="lax_friedrichs", shock_capturing=True)`; `describe()` = those + Trixi version, Julia version, thread count. Julia-side failures → `SolverFailure("trixi", reason)`; configuration failures propagate. |
| D-V18 | `tests/verification/test_crosscode.py::test_reference_backends`: SharpClaw (`SSP33`) vs Trixi (`SSPRK33`), N = 64/128/256, ratio ≥ 1.5, skipping without Julia; no CI Julia job until the laptop proves the setup. Generator defaults untouched (SharpClaw keeps `SSP104`). |
| F12 | FV-generated stores hold cell averages labeled as center point values (O(h²)); what a store declares is decided at 4.4. |

## Ledger

| # | Where | Item | Done when |
|---|---|---|---|
| 1 | cloud | `InitialCondition`/`InitialConditionSampler` contract; samplers rewritten as functions of position (GRF via gstools unstructured evaluation — identical values on grid points; 1-D steps with jump *positions* drawn in the domain; 2-D Voronoi with seeds in the domain, torus distance); `FromFunction`, `Stacked`, `Conserved`, `Multichannel`; `solve(ic, …)` on every family; `generate_dataset(initial_condition=sampler, ic_in_primitive=)`; configs `sampler:`; goldens through the new path | goldens bit-identical; `test_initial_conditions.py` green; every IC config samples; script runs on cubic 1D (step), Euler 2D (primitive GRF), cubic 2D |
| 2 | cloud | `trixi_spec` on the six families; `simulations/trixi.py`: env setup, lazy `juliacall` import, `TrixiBackend` per D-API with `SolverFailure` seam, `describe()`; `scripts/setup_julia.py`; `juliapkg.json`; `juliapkg`+`juliacall` in the dev group | imports without Julia; `TrixiBackend()` raises a clear configuration error when Julia/the project is absent |
| 3 | cloud | V18 test (skips), ADR-0011 §Initial conditions, ADR-0004 amendment, F12, `python-julia-repo.md` + `simulation-backends.md` updated, roadmap/architecture/glossary/index | `check_docs --files` clean |
| 4 | laptop | `reference/trixi/`: `Project.toml`, `Manifest.toml`, `ContextFluxTrixi/` (equations, `node_coordinates(grid, polydeg)`, `solve(family, params, u0_nodes, grid, t_out, bcs; polydeg, cfl, surface_flux, shock_capturing) -> element means`, `test/runtests.jl`) | `Pkg.test` green; `setup_julia.py` green |
| 5 | laptop | `TrixiBackend.solve` end to end on the six families; V18 ratios measured and recorded in `verification.md` | V18 green |
| 6 | laptop | CI Julia job (opt-in), `scripts/data/configs/backend/trixi.yaml` | CI green |

## Laptop steps (Claude Code CLI, repo root, this branch)

1. `uv sync --group dev` (brings `juliapkg`, `juliacall`); `uv run python scripts/setup_julia.py`
   (installs Julia `~1.12` if absent, instantiates `reference/trixi`, precompiles, runs
   `Pkg.test`).
2. Write `reference/trixi/ContextFluxTrixi/src/ContextFluxTrixi.jl` from the `trixi_spec`s and
   `simulations/trixi.py`'s expectations (documented in that module's docstring).
3. `JAX_PLATFORMS=cpu uv run pytest tests/verification/test_crosscode.py -k reference_backends`
   until green; paste the ratios into `docs/verification.md` V18.

## Files

Commit 1 — the `InitialCondition` contract (ledger 1):
- `src/context_flux_no/simulations/initial_conditions/base.py` — new: `InitialCondition` (`n_eqn`, `n_spatial_dims`, `at_points`, `check_coords`), `FromFunction`, `Stacked`, `Conserved`, `InitialConditionSampler`, `Multichannel`.
- `src/context_flux_no/simulations/initial_conditions/grf.py` — rewritten: `GaussianRandomField` sampler → `GaussianRandomFieldCondition` (gstools unstructured evaluation); circulant-embedding `GaussianRandomField1D`, `GaussianRandomField2D`, the covariance classes deleted (only legacy notebooks referenced them).
- `src/context_flux_no/simulations/initial_conditions/step.py` — rewritten: `PeriodicRandomStepFunction1D` → `PeriodicStep1D` (jump positions in the domain), `PeriodicRandomStepFunction2D` → `PeriodicVoronoi2D` (seeds in the domain, torus distance); `MultichannelWaveform` → `base.Multichannel`.
- `src/context_flux_no/simulations/initial_conditions/fourier.py` — `TruncatedFourier1D` is an `InitialCondition`.
- `src/context_flux_no/simulations/initial_conditions/__init__.py` — exports.
- `src/context_flux_no/simulations/pde/base.py` — `solve(ic, …)` evaluates `ic.at_points(grid.meshgrid())` after checking `n_eqn`/`n_spatial_dims`; `ICFactory` gone.
- `src/context_flux_no/simulations/pde/euler.py` — the `solve` override and `use_rho_v_p_ics` gone (`Conserved` replaces them).
- `src/context_flux_no/simulations/pde/cubic.py` — `CubicFlux2D.solve` takes `ic` and uses `Grid` (float64 centers; its float32 `jnp.linspace` grid failed the writer's coordinate check).
- `src/context_flux_no/simulations/pde/burgers.py` — `ViscousBurgers1D.solve` takes `ic`.
- `src/context_flux_no/simulations/generate.py` — `initial_condition: InitialConditionSampler`, `ic_in_primitive`; the record's `ic` carries `in_primitive`.
- `scripts/data/generate_dataset.py` — instantiates `cfg.initial_condition.sampler`, passes `cfg.pde.ic_in_primitive`.
- `scripts/data/configs/initial_condition/*.yaml` — `waveform:` → `sampler:`; `grf_2d.yaml` on `GaussianRandomField`; `step_shallow.yaml` on `Multichannel(samplers=)`.
- `scripts/data/configs/pde/euler_2d.yaml` — `ic_in_primitive: true` replaces `solve_kwargs.use_rho_v_p_ics`; `scripts/data/configs/pde/cubic_2d.yaml` — grid/time keys added (were missing; values mirror `euler_2d`, to confirm at 4.4).
- `tests/unit/make_goldens_simulations.py` — cases wrap formulas in `FromFunction`/`Conserved`; goldens unchanged and still bit-identical.
- `tests/unit/test_generate_dataset.py` — a sampler instead of a function.
- `tests/unit/test_initial_conditions.py` — new.
- `docs/changes/2026-10-25-phase2a-trixi-backend.md` — this document; `docs/index.md` — its row.

Commit 2 — the backend takes the function, not the array (D-IC, second half):
- `src/context_flux_no/simulations/backend.py` — `SolverBackend.solve(pde, ic, grid, t_out, bcs)` and `check_inputs(pde, ic, grid, t_out)` take the `InitialCondition` (checks `n_eqn`/`n_spatial_dims` against the family and grid); `solve_batch` deleted (no caller).
- `src/context_flux_no/simulations/pyclaw.py` — `solve` evaluates `ic.at_points(grid.meshgrid())` as the cell state (the ICLR convention; F12).
- `src/context_flux_no/simulations/pde/base.py` — `solve` passes `ic` through; the check moved into `check_inputs`.
- `tests/unit/test_pyclaw_backend.py` — input checks use a two-channel `FromFunction` instead of a wrong-shaped array.

## Design

- **Callable initial conditions, one contract** (D-IC; ADR-0011 amendment). Rejected: arrays everywhere (Trixi and any high-order oracle would interpolate, an O(h⁴) transfer error of the same order as the schemes V19 compares); `Array | Callable` (two routes); an array-backed `GridValues` adapter (no adopter: goldens use formulas, samplers evaluate anywhere, regression inputs never run a solver).
- **Per-axis coordinate arrays** rather than a stacked `[n_dims, *shape]`: what gstools, the step functions and `Grid.meshgrid()` already produce, and easier to read.
- **Samplers draw positions in the domain** (`sample(key, x_spans)`), so a drawn field does not depend on the grid. This changes the 1-D step distribution (jumps were grid indices) and the 2-D Voronoi domain inference; the ICLR datasets are frozen and 4.4 regenerates, so no reported number moves.
- **`Conserved(ic, pde)` chosen by `ic_in_primitive` in the pde config** rather than a `solve()` flag: the conversion needs the pde instance, which only `generate_dataset` has per trajectory; the family loses its only `solve` override.
- **The backend receives the function** (commit 2). `AbstractPDEFamily.solve` first evaluated `ic` at the centers and handed the array to the backend, which would have forced Trixi to interpolate; now each backend evaluates where its scheme needs it. `solve_batch` (a Python loop over arrays, no caller) went with it — the deletion test: nothing concentrated elsewhere.
- **Deleted samplers**: circulant-embedding GRF and the 2-D single-channel GRF were a second route to what the gstools sampler does; the deletion test passes (nothing concentrates elsewhere). Two legacy data-generation notebooks referenced them; they are Phase 6's to thin.

## Bugs fixed

- `scripts/data/configs/pde/cubic_2d.yaml` had no `x_spans`/`Nxs`/`t_span`/`Nt`, so the config could not run; keys added (values mirror `euler_2d`, flagged to confirm).
- `CubicFlux2D.solve` built its grid with float32 `jnp.linspace`, so its coordinates failed `generate_dataset`'s check against `Grid.centers`; now on `Grid`.

## Findings

## Tests

Fast tier: `tests/unit/test_initial_conditions.py` (16) — every sampler's condition is a function of position (same object on two grids agrees at shared points; same key, same function), step ranges and periodicity, Voronoi periodicity and shared geometry across channels, GRF equals gstools' structured evaluation, period mismatch refused, `Stacked`/`Conserved`/`FromFunction` shapes and errors, truncated Fourier periodicity. `test_pyclaw_backend.py` goldens still bit-identical through the callable path. Script run on cubic 1D (step IC), Euler 2D (primitive GRF through `Conserved`), cubic 2D.

## Open issues

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
