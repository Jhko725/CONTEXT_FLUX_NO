---
type: change
status: landed
updated: 2026-10-26
branch: phase2a-trixi-backend
roadmap: Phase 2a (Trixi half); the InitialCondition contract
base: phase2a-pyclaw-backend
---

# 2026-10-25 — `phase2a-trixi-backend`

The Trixi.jl half of roadmap 2a, designed by interview on 2026-10-25 (decisions below), and
the contract change it needs: initial conditions become *functions* that each backend
discretizes itself. The Python side (contract, `trixi_spec`s, `simulations/backends/trixi.py`, tests
that skip without Julia, documents) was done in Claude's cloud session on 2026-10-25; the Julia
package `reference/trixi/ContextFluxTrixi/`, the committed lock and the V18 runs in a Claude Code
CLI session on Joon's laptop on 2026-10-26 (§Laptop steps says what happened; the Julia-side
decisions were grilled first and are rows D-VF … D-CI below).

**Base.** `phase2a-pyclaw-backend` (merge order: phase1 → phase5 → 2a-pyclaw → this).
`check_docs.py --files --base phase2a-pyclaw-backend`.

## Decisions (grill of 2026-10-25; Design section carries the rejected alternatives)

| # | Decision |
|---|---|
| D-IC | An initial condition is an `InitialCondition` object with `at_points(coords) -> [n_eqn, *shape]` (per-axis coordinate arrays, one shape); samplers return one via `sample(key, x_spans)`; `AbstractPDEFamily.solve(ic, …)` evaluates it at the cell centers; no array-backed adapter (no adopter). **ADR-0011 §Initial conditions.** |
| D-TX | Oracle = Trixi DGSEM `polydeg=3` on `StructuredMesh` (one element per FV cell), surface flux Lax–Friedrichs, Hennemann–Gassner shock capturing; output = element means by LGL quadrature; time stepping SSPRK33 at a fixed CFL via `StepsizeCallback`, `tstops = saveat = t_out`. |
| D-FAM | Six families (five PyClaw ones + `CubicFlux2D`); `ViscousBurgers1D` waits for tier 3. Custom Trixi equations for cubic 1D/2D, sine, parametrized shallow water; Trixi's `InviscidBurgersEquation1D` and `CompressibleEulerEquations2D`. |
| D-BC | Periodic only; `extrap`/`wall` raise `ValueError` at `solve` (V17 adds Trixi's BCs). |
| D-JL | Julia `~1.12` via `juliapkg.json` (Trixi's compat is `julia = "1.10"` = `[1.10, 2)`, CI on 1.10/1.11/1.12); `reference/trixi/Project.toml` + committed `Manifest.toml` with `ContextFluxTrixi` `dev`-ed in. *Mechanism refined in implementation* (Design): juliapkg's shared-project mode (`PYTHON_JULIAPKG_PROJECT`) instead of `PYTHON_JULIACALL_EXE`/`_PROJECT` — same files, one installer. |
| D-TR | Transport = juliacall in-process, Python evaluates the IC at the LGL nodes Julia reports and passes arrays both ways (no Python-from-Julia calls); `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`, `_THREADS=auto`, set before import in `simulations/backends/trixi.py`; no `_jl_call_nogil`; the subprocess transport only if juliacall proves unworkable. **ADR-0004 amendment.** |
| D-SPEC | `trixi_spec` on each family (equations constructor name + parameter kwargs), parallel to `pyclaw_spec`. |
| D-API | `TrixiBackend(polydeg=3, cfl=0.5, surface_flux="lax_friedrichs", shock_capturing=True)`; `describe()` = those + Trixi version, Julia version, thread count. Julia-side failures → `SolverFailure("trixi", reason)`; configuration failures propagate. |
| D-V18 | `tests/verification/test_crosscode.py::test_reference_backends`: SharpClaw (`SSP33`) vs Trixi (`SSPRK33`), N = 64/128/256, ratio ≥ 1.5, skipping without Julia; no CI Julia job until the laptop proves the setup. Generator defaults untouched (SharpClaw keeps `SSP104`). |
| F12 | FV-generated stores hold cell averages labeled as center point values (O(h²)); what a store declares is decided at 4.4. |

Julia side (grill of 2026-10-26; Design carries the rejected alternatives):

| # | Decision |
|---|---|
| D-VF | `volume_flux_dg = flux_central` (standard weak-form DGSEM) for every family; one `volume_flux_dg(eq)` method per family so an entropy-conservative flux (e.g. `flux_ranocha` for Euler) is a one-method swap — not a `TrixiBackend` option. |
| D-WS | Exact local Lax–Friedrichs bound for the non-convex scalar fluxes: `max |f'|` over `[u_ll, u_rr]` (cubic: the vertex `u* = −b/(3a)` checked; sine: `|ab|` when the interval contains an extremum of `cos`). Shallow water: `max |λ±|` over both states. Burgers/Euler: Trixi's own. **F13.** |
| D-HG | `IndicatorHennemannGassner(alpha_max=0.5, alpha_min=0.001, alpha_smooth=true)` with `variable` = the state (scalar laws), the depth (shallow water), `density_pressure` (Euler); constants fixed in the package. |
| D-FAIL | `SolveFailed` for a non-success return code, a non-finite state, a saved-time mismatch, or any exception raised inside the ODE solve; an `ArgumentError` (propagates) for unknown equations/flux names, missing or extra parameters, wrong shapes, a dimension mismatch. |
| D-NODES | `node_coordinates` computed directly from the LGL nodes (no semidiscretization); `runtests.jl` pins them to Trixi's own `cache.elements.node_coordinates` (element and node order) at 1e-12 in 1-D and 2-D. |
| D-PROV | `versions()` carries `"context_flux_trixi"`; `describe()` records it as `julia_package_version`; the package version is bumped whenever D-VF/D-WS/D-HG constants change, so a `DatasetRecord` pins the oracle's numerics. |
| D-CONV | Convergence oracle: method of characteristics on `CubicFluxEquation1D(1, −0.5, 0.25)` (valid for any scalar law before breaking), exact cell averages by LGL quadrature, N = 16 → 32 ratio ≥ 8 at `polydeg = 3`, CFL 0.1 so SSPRK33's O(dt³) stays below the spatial error. |
| D-PINS | Package compat `Trixi = "0.17"`, `OrdinaryDiffEqSSPRK = "2"`, `SciMLBase = "2.154, 3"`, `StaticArrays = "1"`, `julia = "1.10"`; exact versions only in the committed `reference/trixi/Manifest.toml`. |
| D-JLTEST | `Pkg.test` runs through `setup_julia.py --test` and the opt-in CI job only, never from pytest. |
| D-CLEAN | The `.venv/julia_env` install that a bare `import juliacall` probe created was deleted; the one install is `reference/trixi/pyjuliapkg/`. |
| D-DOC | `CLAUDE.md` repository facts: probe the Julia side with `trixi.available()`, never by importing; `uv sync --group dev --extra cuda` on this machine. |
| D-CI | Ledger 6 (CI job) built only after `Pkg.test` and V18 are green. |
| F13 | The LF bound for non-convex fluxes is the interval maximum of `|f'|` (Trixi does this; the Python Riemann solvers are Roe-type and stay; 2b's `LocalLaxFriedrichs` inherits the rule). |
| F14 | `CubicFlux2D.solve` resampled time with `jax.image.resize` and stepped once too often; fixed here (Bugs fixed). |

## Ledger

| # | Where | Item | Done when |
|---|---|---|---|
| 1 ✓ | cloud | `InitialCondition`/`InitialConditionSampler` contract; samplers rewritten as functions of position (GRF via gstools unstructured evaluation — identical values on grid points; 1-D steps with jump *positions* drawn in the domain; 2-D Voronoi with seeds in the domain, torus distance); `FromFunction`, `Stacked`, `Conserved`, `Multichannel`; `solve(ic, …)` on every family; `generate_dataset(initial_condition=sampler, primitive_ic=)`; configs `sampler:`; goldens through the new path | goldens bit-identical; `test_initial_conditions.py` green; every IC config samples; script runs on cubic 1D (step), Euler 2D (primitive GRF), cubic 2D |
| 2 ✓ | cloud | `trixi_spec` on the six families; `simulations/backends/trixi.py`: env setup, lazy `juliacall` import, `TrixiBackend` per D-API with `SolverFailure` seam, `describe()`; `scripts/setup_julia.py`; `juliapkg.json`; `juliapkg`+`juliacall` in the dev group | imports without Julia; `TrixiBackend()` raises a clear configuration error when Julia/the project is absent |
| 3 ✓ | cloud | V18 tests (skip), ADR-0011 §Initial conditions, ADR-0004 amendment, F12, `python-julia-repo.md` + `simulation-backends.md` updated, roadmap/architecture/glossary/index | `check_docs --files` clean |
| 4 ✓ | laptop | `reference/trixi/`: `Project.toml`, `Manifest.toml` (written by juliapkg), `ContextFluxTrixi/` (`equations.jl`, `solve.jl`: `node_coordinates`, `solve`, `versions`; `test/runtests.jl`, 72 checks) | `Pkg.test` green 2026-10-26; `setup_julia.py` prints `describe()` |
| 5 ✓ | laptop | `TrixiBackend.solve` end to end on the six families; V18 ratios measured and recorded in `verification.md`; `julia_package_version` in `describe()` (D-PROV) | V18 green 2026-10-26 (ratios in Tests) |
| 6 | laptop | CI Julia job (opt-in: `workflow_dispatch` or `[julia]` in the commit message; depot and install cached on the lock files) written after 4–5 (D-CI); `scripts/data/configs/backend/trixi.yaml` ✓, README line ✓ | CI green — **unverified**: CI runs on Joon's push (a `[julia]` commit or a manual dispatch) |

## Laptop steps (done 2026-10-26; what happened)

1. `uv sync --group dev --extra cuda` — the CUDA extra is installed on this machine; a sync
   without it removes the CUDA packages (happened once, restored).
2. `reference/trixi/ContextFluxTrixi/` written to the interface in `simulations/backends/trixi.py`'s
   docstring with a complete `Project.toml` (UUIDs from the General registry).
3. `touch reference/trixi/Project.toml`; `uv run python scripts/setup_julia.py --test` — one
   run installed Julia 1.12.7 into `reference/trixi/pyjuliapkg/install` (1.1 GB; the `~/.julia`
   depot grows to 1.2 GB), wrote `Project.toml` (`ContextFluxTrixi` dev-ed, `PythonCall =
   "=0.9.37"`, `OpenSSL_jll`), resolved `Manifest.toml` (Trixi 0.17.16, OrdinaryDiffEqSSPRK
   2.3.2, SciMLBase 3.58.0), precompiled (~15 min). Three `Pkg.test` rounds: `Random` missing
   from the test target; Trixi 0.17 requires `boundary_conditions` explicitly and no longer
   ships `ShallowWaterEquations1D` (oracle → textbook formulas); `saveat` does not save at the
   end time (→ `save_start`/`saveat` interior/`save_end`, each output time by one mechanism).
   Then 72/72; `describe()` printed in 34 s cold.
4. V18 pytest: five failures were on the SharpClaw side — Clawpack has no CFL default for
   `SSP33` (`tests/helpers.py::SHARPCLAW_SSP33`); cubic-2D saturated because of F14 (fixed).
   Ratios in Tests.
5. Import order (juliacall after numpy/JAX in the same process): the pytest runs and the
   generate script import JAX first and start Julia at the first `solve`; no shared-library
   clash in any of them (`generate_dataset.py pde=cubic_1d backend=trixi`: 2 trajectories, 0
   failed, the record's `backend` carries the Julia-side versions). The risk is retired.

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
- `src/context_flux_no/simulations/backends/base.py` — `SolverBackend.solve(pde, ic, grid, t_out, bcs)` and `check_inputs(pde, ic, grid, t_out)` take the `InitialCondition` (checks `n_eqn`/`n_spatial_dims` against the family and grid); `solve_batch` deleted (no caller).
- `src/context_flux_no/simulations/backends/pyclaw.py` — `solve` evaluates `ic.at_points(grid.meshgrid())` as the cell state (the ICLR convention; F12).
- `src/context_flux_no/simulations/pde/base.py` — `solve` passes `ic` through; the check moved into `check_inputs`.
- `tests/unit/test_pyclaw_backend.py` — input checks use a two-channel `FromFunction` instead of a wrong-shaped array.

Commit 3 — the Trixi backend's Python side (ledger 2–3):
- `src/context_flux_no/simulations/backends/trixi.py` — new: `TrixiSpec`, `TrixiBackend` (D-API; periodic only; `SolverFailure("trixi", …)` seam; `describe()`), `TrixiUnavailable`, `available()`, `configure_environment()`, `REFERENCE_TRIXI`; the Julia interface in the docstring.
- `src/context_flux_no/simulations/pde/{base,cubic,sine_1d,shallow_water,burgers,euler}.py` — `trixi_spec` (base raises `NotImplementedError`; six families return a `TrixiSpec`).
- `src/context_flux_no/simulations/__init__.py` — exports `TrixiBackend`, `TrixiSpec`.
- `src/context_flux_no/juliapkg.json` — new: Julia `~1.12`, `ContextFluxTrixi` `dev`-ed from `reference/trixi/ContextFluxTrixi`.
- `scripts/setup_julia.py` — new: resolve + precompile + `describe()`; `--test`.
- `pyproject.toml`, `uv.lock` — `juliacall`, `juliapkg` in the dev group.
- `.gitignore` — `reference/trixi/pyjuliapkg/`.
- `tests/unit/test_trixi_backend.py` — new: specs on six families, option/BC/input checks before Julia starts, the configuration error.
- `tests/verification/test_crosscode.py` — V18: `test_reference_backends_{1d,euler_2d,cubic_2d}` (skip without Julia).
- `docs/decisions/0011-solver-backends-and-dataset-provenance.md` — §Initial conditions (amendment); Decision 1 signature.
- `docs/decisions/0004-weno5-and-reference-backends.md` — §Amendment: in-process transport.
- `docs/findings.md` — F12.
- `docs/design/python-julia-repo.md` — `agreed`; §Decided (what the code does, two corrected deductions), laptop steps; superseded passages marked.
- `docs/design/simulation-backends.md` — pieces, `solve(ic, …)`, §Trixi, record fields, V18 protocol.
- `docs/verification.md` — V18 row.
- `docs/roadmap.md` — 2a-trixi row; Done ledger.
- `docs/architecture.md`, `docs/glossary.md`, `docs/index.md` — trixi.py, initial_conditions, setup script, terms, statuses.

Commit 4 — the Julia side (ledger 4–5, laptop):
- `reference/trixi/Project.toml`, `reference/trixi/Manifest.toml` — new: the committed lock, written by juliapkg (shared-project mode).
- `reference/trixi/ContextFluxTrixi/Project.toml` — new: package 0.1.0, deps/compat (D-PINS), test target.
- `reference/trixi/ContextFluxTrixi/src/ContextFluxTrixi.jl` — new: module, `SolveFailed`, exports.
- `reference/trixi/ContextFluxTrixi/src/equations.jl` — new: `CubicFluxEquation1D/2D`, `SineFluxEquation1D`, `ParametrizedShallowWaterEquations1D` (exact LF bound, D-WS), `indicator_variable`, `volume_flux_dg`, the `EQUATIONS` table with strict parameter checking.
- `reference/trixi/ContextFluxTrixi/src/solve.jl` — new: `node_coordinates`, `solve` (StructuredMesh periodic, DGSEM + HG, state injected via `wrap_array`, SSPRK33 + `StepsizeCallback`, element means), `versions`.
- `reference/trixi/ContextFluxTrixi/test/runtests.jl` — new: 72 checks (node order pin, D-WS bound, flux oracles, conservation, Burgers equivalence, characteristics convergence, failure seam, versions).
- `src/context_flux_no/simulations/backends/trixi.py` — `describe()` records `julia_package_version` (D-PROV); `faulthandler.disable()` before Julia starts; docstring: `versions()` keys, import order observed, the faulthandler fact.
- `src/context_flux_no/simulations/pde/cubic.py` — `CubicFlux2D.solve` steps exactly onto the output times (F14; Bugs fixed).
- `tests/helpers.py` — new: `SHARPCLAW_SSP33`.
- `tests/verification/test_crosscode.py` — the SharpClaw partner uses `SHARPCLAW_SSP33`.
- `tests/unit/test_trixi_backend.py` — `describe()` keys pinned where Julia is available.
- `scripts/data/configs/backend/trixi.yaml` — new (ledger 6b); `scripts/data/README.md` — the `backend=trixi` line.
- `.github/workflows/test.yml` — `workflow_dispatch` trigger; opt-in `julia` job (ledger 6a).
- `CLAUDE.md` — Julia probing and sync facts (D-DOC).
- `docs/findings.md` — F13, F14. `docs/verification.md` — V18 ratios. `docs/design/{python-julia-repo,simulation-backends}.md`, `docs/architecture.md`, `docs/roadmap.md` — the Julia side as it is.

Commit 5 — review round 1, items 1–2 (initial conditions):
- `src/context_flux_no/simulations/initial_conditions/base.py` — `n_eqn`/`n_spatial_dims` are attributes set once by `InitialCondition.__init__` / `InitialConditionSampler.__init__` (validated ≥ 1), not abstract properties re-implemented per class; `Stacked`/`Conserved`/`Multichannel` compute and pass them.
- `src/context_flux_no/simulations/initial_conditions/{grf,step}.py` — subclasses call `super().__init__`; the ten property pairs gone.
- `src/context_flux_no/simulations/initial_conditions/fourier.py` — `TruncatedFourier1D.with_uniform_rand_coeffs` → `RandomTruncatedFourier1D(num_modes, coeff_range, offset_range)`, an `InitialConditionSampler` whose period is the domain length.
- `src/context_flux_no/simulations/initial_conditions/__init__.py` — export.
- `tests/unit/test_initial_conditions.py` — the Fourier sampler in the function-of-position law; period = domain length. `tests/unit/test_generate_dataset.py` — the test sampler calls `super().__init__`.

Commit 6 — review round 1, item 3 (`backends/` subpackage):
- `src/context_flux_no/simulations/backends/{__init__,base,pyclaw,trixi}.py` — `simulations/{backend,pyclaw,trixi}.py` moved (`git mv`; `backend.py` → `base.py`); `REFERENCE_TRIXI` one level deeper; `simulations/__init__.py` re-exports from the subpackage; every import site (`simulations/{generate,writer}.py`, `pde/*`, `scripts/setup_julia.py`, `tests/unit/test_{pyclaw,trixi}_backend.py`, `tests/verification/test_{crosscode,backend_agreement}.py`), config target, document and `CLAUDE.md` updated (`docs/changes/2026-10-25-phase2a-pyclaw-backend.md` kept as the historical record).

Commit 7 — review round 1, item 4:
- `src/context_flux_no/simulations/generate.py` — `ic_in_primitive` → `primitive_ic` (record key `ic.primitive`); `generate_dataset` docstring in the equinox/diffrax form.
- `scripts/data/configs/pde/euler_2d.yaml`, `scripts/data/generate_dataset.py`, `tests/unit/test_generate_dataset.py`, `docs/design/simulation-backends.md`, ADR-0011 — the key.

## Design

- **Callable initial conditions, one contract** (D-IC; ADR-0011 amendment). Rejected: arrays everywhere (Trixi and any high-order oracle would interpolate, an O(h⁴) transfer error of the same order as the schemes V19 compares); `Array | Callable` (two routes); an array-backed `GridValues` adapter (no adopter: goldens use formulas, samplers evaluate anywhere, regression inputs never run a solver).
- **Per-axis coordinate arrays** rather than a stacked `[n_dims, *shape]`: what gstools, the step functions and `Grid.meshgrid()` already produce, and easier to read.
- **Samplers draw positions in the domain** (`sample(key, x_spans)`), so a drawn field does not depend on the grid. This changes the 1-D step distribution (jumps were grid indices) and the 2-D Voronoi domain inference; the ICLR datasets are frozen and 4.4 regenerates, so no reported number moves.
- **`Conserved(ic, pde)` chosen by `ic_in_primitive` in the pde config** rather than a `solve()` flag: the conversion needs the pde instance, which only `generate_dataset` has per trajectory; the family loses its only `solve` override.
- **The backend receives the function** (commit 2). `AbstractPDEFamily.solve` first evaluated `ic` at the centers and handed the array to the backend, which would have forced Trixi to interpolate; now each backend evaluates where its scheme needs it. `solve_batch` (a Python loop over arrays, no caller) went with it — the deletion test: nothing concentrated elsewhere.
- **juliapkg's shared-project mode carries D-JL** (`PYTHON_JULIAPKG_PROJECT=reference/trixi`). The interview's mechanism — `PYTHON_JULIACALL_EXE`/`_PROJECT` set by hand, `Pkg.instantiate` run by the script — would have re-implemented what juliapkg does (find/install Julia, pin PythonCall to the juliacall version) around it, using its non-public `find_julia`. Shared mode keeps the committed `Manifest.toml` untouched and `dev`s `ContextFluxTrixi` from `juliapkg.json` [code: juliapkg 0.1.28 `deps.py:560-607`]. Same files committed, one installer. Rejected: juliapkg's default per-venv project (`.venv/julia_env`, Manifest not committed — loses the lock) and the hand-rolled route above.
- **`available()` = "juliapkg has resolved this project here"** (`reference/trixi/pyjuliapkg/meta.json`), not "Julia is importable": importing juliacall would *download Julia* on a machine without it, which a skipping test must never do. Rejected: an opt-in environment flag (a switch nobody asked for) and probing for a `julia` on `PATH` (misses juliapkg's own install).
- **V18 partner for `CubicFlux2D`** is its own first-order Rusanov stepper (it has no `pyclaw_spec`; `solve` refuses `backend=`), so that test calls `TrixiBackend.solve` directly and expects the first-order rate (ratio → 2 ≥ 1.5).
- **Julia receives names and plain data, never Python callables**: the equations constructor name and a `dict` of floats (D-SPEC), numpy arrays for nodes/state/times. PythonCall hands them over as `PyDict`/`PyArray` [cited: PythonCall docs, "Conversion to Julia": arguments are converted with `T=Any`], so the Julia functions are typed on the abstract containers and copy.
- **Julia side, per the grill** (D-VF … D-CI). *Central volume flux* over entropy-conservative ones: a single rule, and no EC two-point flux exists for the custom families; the swap is one method per family. *Exact LF bound* over the endpoint max: the oracle should be the more trustworthy side (F13), V18 is indifferent to the convention. *Direct node coordinates* over reading Trixi's cache: cheap, and the test pins the order so a Trixi change cannot silently corrupt the injection. *State injected through `wrap_array`* over an `initial_condition` closure looking coordinates up: no interpolation, no lookup. *Textbook shallow-water oracle* over a TrixiShallowWater.jl test dependency: one flux formula does not justify a package. *Failure boundary at the ODE solve*: construction errors are the caller's (D-API "configuration propagates"), stepping errors are the oracle's. *Each output time saved by exactly one mechanism* (`save_start`, interior `saveat`, `save_end`): OrdinaryDiffEq does not save a `saveat` entry that coincides with the end time.
- **`faulthandler.disable()` in `trixi._julia()`**, next to the environment variables, rather than `-p no:faulthandler` in `pyproject.toml`: the collision (Python's SIGSEGV handler vs Julia's segfault-based GC safepoints) is a property of the process once Julia is in it, not of pytest, so the module that owns the other process-wide Julia invariants owns this one; a pytest option would leave a notebook or script that enables `faulthandler` exposed.
- **`SHARPCLAW_SSP33` as a named configuration** in `tests/helpers.py` rather than literals at the call site: Clawpack provides CFL defaults only for SSP104, so the SSP33 numbers are a choice that needs its reason next to it (same CFL per unit SSP coefficient as Clawpack's SSP104 defaults).
- **Review item 1 — the overlap between a condition and its sampler.** Kept the distribution-vs-draw split (the draw is evaluated on several grids in V18, so it must be an object, and the sampler is what Hydra instantiates and `DatasetRecord.ic` describes); removed the duplication by declaring `n_eqn`/`n_spatial_dims` once in each base `__init__`. Rejected: one class with an "unsampled" state (`at_points` would have to raise before `sample`); a generic sampler around a closure (hyperparameters in a `partial`, worse for Hydra and the record); deriving the sampler's dims by a dry draw (a sample at construction). Jit/vmap for the JAX backend (4.7) is unaffected — a backend evaluates `at_points` once on the host and jits the stepping; the only thing that would keep an IC *out* of a jitted computation is the `np.asarray` coercion in `FromFunction`/`Stacked`/`Conserved`, left until 4.7 names a use.
- **Review item 2 — `RandomTruncatedFourier1D`.** The classmethod took a key but not the domain, so `L` could disagree with `x_spans`; as a sampler the period is the domain length. Its default `coeff_range=(1, 1)` drew nothing (every coefficient 1); the sampler's default is `(-1, 1)`. Only a legacy notebook used the classmethod (Phase 6).
- **Review item 3 — `simulations/backends/`.** Three adopters (`base`, `pyclaw`, `trixi`) and a fourth known (`jax`, 4.7); moved now rather than at 4.7 so no later branch rebases across it. `grid.py` stays at `simulations/` (shared with the writer and the families).
- **Deleted samplers**: circulant-embedding GRF and the 2-D single-channel GRF were a second route to what the gstools sampler does; the deletion test passes (nothing concentrates elsewhere). Two legacy data-generation notebooks referenced them; they are Phase 6's to thin.

## Bugs fixed

- `scripts/data/configs/pde/cubic_2d.yaml` had no `x_spans`/`Nxs`/`t_span`/`Nt`, so the config could not run; keys added (values mirror `euler_2d`, flagged to confirm).
- `CubicFlux2D.solve` built its grid with float32 `jnp.linspace`, so its coordinates failed `generate_dataset`'s check against `Grid.centers`; now on `Grid`.
- `CubicFlux2D.solve` resampled its trajectory onto the output times with `jax.image.resize(method="cubic")` — pixel-center alignment along *time* — and stepped once too often (`len(t_solve - 1)`): with `Nt = 1` the "final" frame was the state at ≈ 0.75 T. Found by V18 (the distance to Trixi saturated at an N-independent 1.8e-3). Now equal substeps that land exactly on every output time, dt from the CFL number; the Rusanov flux is unchanged. **F14.**
- `tests/verification/test_crosscode.py` asked SharpClaw for `SSP33` without CFL numbers, which Clawpack only defaults for SSP104 (`KeyError: 'Maximum CFL number is not provided.'`); the test could never have run.

## Findings

- **F13** — the Lax–Friedrichs bound for non-convex fluxes is the interval maximum of `|f'|`; the Trixi oracle uses it, the own schemes inherit it at 2b.
- **F14** — `CubicFlux2D.solve`'s time resampling mislabeled the ICLR cubic-2D frames' times; fixed here, data regenerate at 4.4.

## Tests

`tests/unit/test_trixi_backend.py` (4, fast): six `trixi_spec`s distinct and `ViscousBurgers1D` without one; option errors, non-periodic BCs and `n_eqn` mismatch raise before Julia starts; `TrixiUnavailable` from `solve`/`describe` on a machine without the setup (skips once Julia is set up); `describe()` carries the four version keys where Julia is set up (starts Julia: ~7 s warm, 34 s cold).

Julia: `uv run python scripts/setup_julia.py --test` → `Pkg.test("ContextFluxTrixi")`, 72 checks in ~85 s: node coordinates (shapes, every node in its cell, LGL quadrature exact on `x²`, **equal to Trixi's own node coordinates in Trixi's element/node order**, 1-D and 2-D); the D-WS bound (≥ the sampled `max |f'|` and within 1e-3 of it on 200 random intervals for two cubics and two sines; pointwise at a degenerate interval; the 2-D normal-vector flux and speed); flux oracles (cubic `(0, ½, 0)` ≡ Trixi Burgers flux and speeds; parametrized SW at `γ = α = 1` ≡ textbook `(hv, hv² + ½gh²)`, `|v| + √(gh)`); `solve` shapes, first slice = LGL means of `u0`, total mass conserved to 1e-12 (cubic 1D, SW, cubic 2D) / 1e-11 (Euler 2D), weak form and single-time paths; cubic `(0, ½, 0)` reproduces `"InviscidBurgersEquation1D"` to 1e-10; characteristics convergence ratio ≥ 8 at N = 16 → 32 (polydeg 3, CFL 0.1); the failure seam (six `ArgumentError`s, `SolveFailed` for a non-finite input and for Euler at negative pressure); `versions()` keys and values.

V18 (`JAX_PLATFORMS=cpu uv run pytest tests/unit/test_trixi_backend.py tests/verification/test_crosscode.py -m "slow or not slow" -k "trixi or reference_backends or describe"`), SharpClaw `SSP33` (`SHARPCLAW_SSP33`) vs Trixi, L1 of the final frame, ratio per halving ≥ 1.5 — measured 2026-10-26: Burgers 3.98/4.01, cubic 3.82/4.79, SW 3.99/4.00, sine 4.53/4.16 (N = 64/128/256, T = 0.05 / SW 0.03), Euler-2D 3.98/4.00 (N = 16/32/64, T = 0.02), cubic-2D vs its own stepper 2.33/2.24 (first-order rate). Every SharpClaw pairing sits at ≈ 4: the distance is dominated by the O(h²) difference between SharpClaw's point-value state and Trixi's element means (F12), not by either scheme's own error; the cubic-2D pairing shows the Rusanov stepper's first-order rate. Wall time ≈ 6 min (sine 127 s, cubic 99 s). The pytest process first exited 139 after reporting a green run — pytest's `faulthandler` SIGSEGV handler against Julia's threaded GC at interpreter exit (reproduced on the 7-second `describe` test; gone with `-p no:faulthandler` or one Julia thread) — fixed by `faulthandler.disable()` in `trixi._julia()` before Julia starts; the same test then exits 0.

Fast tier 2026-10-26 (`JAX_PLATFORMS=cpu uv run pytest -q`, Julia set up): 103 passed, 1 skipped (`test_unavailable…`), 3 min 12 s, exit 0. After review round 1 (commits 5–7; cloud, no Julia): 105 passed, 1 skipped (`test_describe…`), 4 min 04 s; `import_all` and `check_configs` unchanged. `tests/unit/test_initial_conditions.py` (16) — every sampler's condition is a function of position (same object on two grids agrees at shared points; same key, same function), step ranges and periodicity, Voronoi periodicity and shared geometry across channels, GRF equals gstools' structured evaluation, period mismatch refused, `Stacked`/`Conserved`/`FromFunction` shapes and errors, truncated Fourier periodicity. `test_pyclaw_backend.py` goldens still bit-identical through the callable path. Script run on cubic 1D (step IC), Euler 2D (primitive GRF through `Conserved`), cubic 2D.

## Open issues

- Import order: juliacall is imported lazily after numpy/JAX in every run (pytest with JAX x64 enabled, `setup_julia.py`, the generate script) without a shared-library clash; the PySR precaution is not needed here. Observed on one machine (WSL2, CPU JAX); a GPU-JAX process is the remaining unobserved case.
- Euler with `flux_central` + HG at longer horizons than V18's `T = 0.02`: if a generation run ever fails, the first thing to try is `volume_flux_dg(::CompressibleEulerEquations2D) = flux_ranocha` (one method, package version bump).
- `SHARPCLAW_SSP33`'s CFL numbers (0.4 / 0.45) are Clawpack's SSP104 defaults per unit SSP coefficient, not a measured stability limit; a tighter or looser choice re-measures V18.
- `Burgers1D.nu` is recorded as a Well scalar but used by neither reference (`pyclaw_spec` and `trixi_spec` both ignore it); a `nu != 0` request is silently inviscid. Pre-existing; belongs with the viscous-Burgers tier-3 work.
- `CubicFlux2D` on Trixi and its own stepper are the only two solvers of that family; V18's cubic-2d case therefore tests first-order convergence, not two references against each other — JaxBackend (4.7) or a SharpClaw 2-D kernel would close that.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `simulations/initial_conditions/base.py` — `InitialCondition`, `FromFunction`, `Stacked`, `Conserved`, `InitialConditionSampler`, `Multichannel` | May have to revisit later when we roll the JAX WENO5 to make more jit and vmap friendly. There is a bit of overlap between the initial conditions and its sampler. Are there potential alternate design than the current style? Or ways to minimize overlap. | To be discussed with Claude. |
| `simulations/initial_conditions/grf.py` — gstools sampler evaluable anywhere; circulant GRFs deleted | Looks good. Agree on the deletions. | None |
| `simulations/initial_conditions/step.py` — `PeriodicStep1D` / `PeriodicVoronoi2D` with positions in the domain | Looks good. | None |
| `simulations/initial_conditions/fourier.py` — `TruncatedFourier1D` is an `InitialCondition` | Agreed. But its `with_uniform_rand_coeffs` correspond to the functionality of its sampler. | Deferred changes to Claude. |
| `simulations/initial_conditions/__init__.py` — exports | Trivial changes. | None |
| **Round 2** `initial_conditions/base.py`, `grf.py`, `step.py` — dims set once in the base `__init__`s (item 1) | | |
| **Round 2** `simulations/backends/` — `backend.py` → `backends/base.py`, `pyclaw.py`, `trixi.py` moved; call sites, configs, docs (item 3) | | |
| **Round 2** `generate.py` — `primitive_ic`, docstring; config key, script, test, docs (item 4) | | |
| **Round 2** `initial_conditions/fourier.py` — `RandomTruncatedFourier1D` sampler (item 2); `test_initial_conditions.py`, `test_generate_dataset.py` | | |
| `simulations/backends/base.py` — `solve(pde, ic, grid, t_out, bcs)`; `check_inputs(ic)`; `solve_batch` deleted | Looks good. | None |
| `simulations/backends/pyclaw.py` — evaluates `ic` at the centers | Looks good. | None |
| `simulations/backends/trixi.py` — `TrixiSpec`, `TrixiBackend`, `available()`, environment, Julia interface spec | Looks good. Feel that `simulations` need a `backends` subpackage though. It will contain the current `backends..py`, `pyclaw.py` and `trixi.py`. | To be discussed with Claude. |
| `simulations/pde/base.py` — `solve(ic, …)` passes the function; `trixi_spec` default | Looks good. | None |
| `simulations/pde/{cubic,sine_1d,shallow_water,burgers,euler}.py` — `trixi_spec`; Euler `solve` override gone; cubic-2D on `Grid` | Looks good. | None |
| `simulations/generate.py` — sampler + `ic_in_primitive` | Logic is good, but let's rename `ic_in_primitive` to `ic_to_primitive`. Also, want proper docstings (`equinox`/`diffrax` style) for this function. | Changes deferred to Claude. |
| `simulations/__init__.py` — exports | Trivial changes. | None |
| `juliapkg.json`, `scripts/setup_julia.py`, `pyproject.toml`/`uv.lock`, `.gitignore` — Julia set-up | Looks good. | None |
| `scripts/data/generate_dataset.py` + configs — `sampler:`, `ic_in_primitive`, cubic-2D keys | Looks good. | None |
| `tests/unit/make_goldens_simulations.py`, `test_generate_dataset.py`, `test_pyclaw_backend.py` — through the callable path | 
Confirmed. | None |
| `tests/unit/test_initial_conditions.py`, `tests/unit/test_trixi_backend.py` — new | Looks good. | None |
| `tests/verification/test_crosscode.py` — V18 tests (skip) | Looks good. | None |
| `reference/trixi/{Project,Manifest}.toml` — the committed Julia lock (juliapkg shared mode; Trixi 0.17.16) | Looks good. | None |
| `reference/trixi/ContextFluxTrixi/{Project.toml, src/ContextFluxTrixi.jl}` — package, deps/compat, `SolveFailed` | Looks good. | None |
| `reference/trixi/ContextFluxTrixi/src/equations.jl` — four custom equations, exact LF bound (D-WS), indicator variable, volume flux, `EQUATIONS` table | Looks good. | None |
| `reference/trixi/ContextFluxTrixi/src/solve.jl` — `node_coordinates`, `solve`, `versions` | Had a skim. Looks good. | None |
| `reference/trixi/ContextFluxTrixi/test/runtests.jl` — 72 checks | Skimmed for now. Will revisit later. | None for now. |
| `simulations/backends/trixi.py` — `julia_package_version` (D-PROV); `faulthandler.disable()` before Julia starts | Confirmed. | None |
| `simulations/pde/cubic.py` — `CubicFlux2D.solve` time stepping (F14) | Looks good. | None |
| `tests/helpers.py` — `SHARPCLAW_SSP33`; `tests/verification/test_crosscode.py` uses it; `tests/unit/test_trixi_backend.py` — `describe()` keys | Confirmed. | None |
| `scripts/data/configs/backend/trixi.yaml`, `scripts/data/README.md`, `CLAUDE.md` (D-DOC) | Looks good. | None |
| `.github/workflows/test.yml` — opt-in `julia` job (unverified until a `[julia]` push) | COnfirmed. | None |
| F13, F14; `verification.md` V18 ratios; `python-julia-repo.md` §what happened; `simulation-backends.md` §Trixi; architecture; roadmap | Looks good. | None |
| ADR-0011 §Initial conditions, ADR-0004 §Amendment, F12 | Looks good. | None |
| `design/python-julia-repo.md` (agreed; §Decided), `design/simulation-backends.md` (§Trixi), `verification.md`, roadmap, architecture, glossary, index | Looks good. | None |
