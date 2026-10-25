---
type: change
status: landed
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

## Ledger (confirmed 2026-10-25; all nine items landed)

| # | Item | Done when |
|---|---|---|
| 1 | `simulations/` package extended in place (Joon: keep the name `simulations`, not `sim`): `simulations/grid.py::Grid(x_spans, Nxs)` (cell centres, `dx`, `n_spatial_dims`), `simulations/backend.py::SolverBackend` ABC with `solve(pde, u0, grid, t_out, bcs, *, params) -> u[time, n_eqn, *Nxs]` and `solve_batch` (loop, `vmap` later), `BCName = Literal["periodic", "extrap", "wall"]` per axis-side (strings until ADR-0003's objects exist in 2b; recorded in metadata) | imports; `SolverBackend` has one concrete subclass |
| 2 | `simulations/pyclaw.py::PyClawBackend(scheme="classic" \| "sharpclaw", *, cfl=0.5, cfl_max=0.9, limiter="MC", weno_order=5, time_integrator="SSP104", transverse_waves=2)`; folds `simulations/pdesolve.py` + `pyclaw_utils.py` (deleted). Needs from a PDE class a `PyClawSpec(riemann_solver, num_waves, kernel_language, problem_data_fn, fwave)` — added as a `pyclaw_spec` property on the six existing PDE classes; their `solve()` methods become one-line delegations to `PyClawBackend("classic")` with the old signature kept for `generate_dataset` | **bit-identical test**: every family, fixed IC and params, `solve()` before vs after on a 64-cell grid (goldens made from `main` in this branch, stored as small `.npz` under `tests/unit/goldens_simulations/`) |
| 3 | SharpClaw per family: Python-kernel Riemann solvers (cubic, sine, parametrised SW) go through `SharpClawSolver1D` with `lim_type=2, weno_order=5, char_decomp=0`; built-in Fortran kernels (`riemann.burgers_1D`, `riemann.euler_4wave_2D`) use `kernel_language="Fortran"`; 2D via `SharpClawSolver2D` | each family runs one step under both schemes without error |
| 4 | **Backend-agreement test** (V18's PyClaw-internal precursor; new row V18a): Classic vs SharpClaw, smooth IC, short horizon, `N ∈ {64, 128, 256}` — the L1 difference must shrink by ≥ 1.5× per refinement (mutual convergence, ADR-0004 §Consequences). 1D families in `tests/unit` behind `pytest.importorskip("clawpack")` (seconds); 2D Euler/cubic in `tests/verification` (`-m slow`) | green; measured ratios recorded in `verification.md` |
| 5 | `simulations/record.py::DatasetRecord` (mirror of `runs.RunRecord`): `dataset_id` (`<family>-<split>-<utc stamp>-<8 hex>`), `family`, `split`, `n_trajectories`, `grid` (spans, Nxs, dx), `t_out` (span, Nt, dt), `bcs`, `params` (names, ranges, sampling strategy), `ic` (waveform name + kwargs), `backend` (name, scheme, options, `clawpack.__version__`), `seed`, `git {sha, dirty}`, `package_version`, `created_at`, `status` (`writing` → `complete`, `n_failed`), `schema_version`; `to_json/from_json`, Hypothesis round trip | tests green |
| 6 | `simulations/writer.py::WellWriter` — `simulations/dataset.py::ZarrWellDataset` moved and extended: `boundary_conditions` group per The Well's layout (one subgroup per axis-side with `bc_type`, `associated_dims`, `associated_fields`, `sample_varying`, `time_varying` attrs; **to verify against the Well format doc at implementation**), `dataset.json` (= `DatasetRecord`) beside the store and in root attrs, `open()` for append/inspection, `finalize(n_written)` replaces `resize_dataset`; `write_trajectory(idx, u, scalars)` takes `[time, n_eqn, *Nxs]` + the PDE's `solution_to_well_schema` | Well schema round trip through `ZarrWellDatasetSource` in a test; `WellDatasetSourceBase.dataset_id` returns `DatasetRecord.id` when `dataset.json` exists, else the provisional hash |
| 7 | `simulations/generate.py::generate_dataset(..., backend: SolverBackend)` — `simulations/utils.py` moved; per-trajectory failures counted into the record instead of silently dropped; `scripts/data/generate_dataset.py` gains a `backend` config group (`configs/backend/{classic,sharpclaw}.yaml`, default **classic**) | one tiny end-to-end run (cubic 1D, 2 coefficients × 2 ICs, N = 32) in a test, loaded back with `ZarrWellDatasetSource` |
| 8 | `simulations/pde/*` stays where it is until Phase 4.1 (`PDEFamily`); `simulations/{dataset,utils,pdesolve,pyclaw_utils}.py` deleted in favour of the modules above; `scripts/import_all.py` skip list updated (clawpack now builds here with `apt-get install gfortran`, note added to README) | `import_all` clean; `check_configs` unchanged |
| 9 | Docs: ADR-0011 (reference backends as a `SolverBackend` protocol + dataset provenance; why strings-for-BCs now), `design/simulation-backends.md` (how to add a backend / a family; SharpClaw options; agreement-test protocol), `verification.md` V18a + the measured ratios, roadmap (2a split into 2a-pyclaw done / 2a-trixi pending; 4.3 done), architecture, glossary, index, this document's Files/Design/Tests/Review-notes | `check_docs --files --base phase5-run-identity` clean |

Not in this ledger (deliberately): BC objects and ghost-cell padding (2b — Joon), `PDEFamily`
(4.1), regenerating any dataset (4.4), `JaxBackend` (4.7), Trixi (below).

## Trixi hand-off (for Joon's machine; not executed here)

What `reference/trixi/` + `simulations/trixi.py::TrixiBackend` need, so it can be started independently:

1. Julia and the Julia environment managed as `docs/design/python-julia-repo.md` describes:
   `juliapkg` (a dev dependency) installs/locates a Julia matching the compat in
   `juliapkg.json`; `reference/trixi/Project.toml` + `Manifest.toml` pin Trixi.jl,
   OrdinaryDiffEq.jl (SSPRK43 or CarpenterKennedy2N54), HDF5.jl, JSON3.jl exactly.
2. One elixir per family under `reference/trixi/elixirs/`: `TreeMesh` 1D/2D periodic,
   `DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)` with a shock-capturing volume integral
   where the family shocks (`VolumeIntegralShockCapturingHG`), `initial_condition(x, t, equations)`
   reading `u0` from the HDF5 input by nearest cell, output projected to cell averages on the
   requested grid (`Trixi.integrate` per element or `PlotData` sampling — decide in the elixir).
3. Two transports over one Julia function `ContextFluxTrixi.solve` (`design/python-julia-repo.md`
   §C, revised): `TrixiBackend(transport="juliacall")` (default; numpy arrays in and out,
   `_jl_call_nogil`, env set before import) and `transport="subprocess"` (`run.jl` with
   `request.json` = `{family, params, grid, t_out, bcs, polydeg}` + `u0.h5`/`u.h5`) for Slurm
   jobs needing failure isolation.
4. The scalar families (cubic, sine) need custom `AbstractEquations` in the elixir (Trixi ships
   `LinearScalarAdvection`, `Burgers`, `ShallowWater`, `CompressibleEuler`; a scalar law with
   user `flux(u)` is ~30 lines following `BurgersEquation1D`).
5. V18 then runs against `PyClawBackend("sharpclaw")` with the same mutual-convergence protocol as
   V18a.

## Files

Also on this branch before the code (docs commits `d95900a`, `1ff31bb`): `docs/design/fv-core-plan.md`
(Phase 2b/2c plan for Joon), `docs/design/python-julia-repo.md` (Julia layout, juliacall assessment),
`docs/index.md`, `docs/roadmap.md`, `docs/architecture.md` (`sim` → `simulations` naming).

Commit 1 — `7f055c1` backend (ledger 1–4):
- `src/context_flux_no/simulations/grid.py` — new: `Grid` (PyClaw's centre arithmetic, `dxs`, `axis_names`, `to_dict`).
- `src/context_flux_no/simulations/backend.py` — new: `SolverBackend` ABC, `BCName`/`BCs`, `normalize_bcs`, `check_inputs`.
- `src/context_flux_no/simulations/pyclaw.py` — new: `PyClawSpec`, `PyClawBackend(scheme, **solver_overrides)` (folds the old `pdesolve.py` + `pyclaw_utils.py`), `describe()`.
- `src/context_flux_no/simulations/pde/base.py` — rewritten: minimal common interface + default `solve()` through a backend; `ICFactory`.
- `src/context_flux_no/simulations/pde/{cubic,sine_1d,shallow_water,burgers,euler}.py` — the five PyClaw classes subclass the base, gain `parameters`/`field_rank_names`/`pyclaw_spec`, lose their hand-built solvers; `SineFlux1D.solve_pyclaw` → `solve`; `Euler2D.to_conserved`; `riemann_cubic_1D` silences its masked divide; `CubicFlux2D`/`ViscousBurgers1D` on the base too (own steppers kept, `backend=` refused).
- `src/context_flux_no/simulations/pde/__init__.py` — exports all seven classes + base.
- `tests/unit/goldens_simulations/{cubic_1d,sine_1d,burgers_1d,shallow_water_1d,euler_2d}.npz` — new: Classic output of the pre-2a `solve()` at `1ff31bb`.
- `tests/unit/make_goldens_simulations.py` — new: the cases + regeneration script.
- `tests/unit/test_pyclaw_backend.py` — new: bit identity (V25), SharpClaw on every family, `Grid`, `normalize_bcs`, input checks, non-periodic run.
- `tests/unit/test_backend_agreement.py` — new: V18a, 1-D families.
- `tests/verification/{__init__,conftest,test_crosscode}.py` — new: the verification tier scaffold (`x64`, `-m slow`) and V18a for 2-D Euler.

Commit 2 — `db62a07` records and writer (ledger 5–8):
- `src/context_flux_no/simulations/record.py` — new: `DatasetRecord`, `sidecar_path`.
- `src/context_flux_no/simulations/writer.py` — new: `WellWriter` (replaces `simulations/dataset.py`; adds `boundary_conditions`, record in attrs + sidecar, `finalize(n_written, n_failed)`).
- `src/context_flux_no/simulations/generate.py` — new: `generate_dataset(..., backend, ic_description)` + `sample_coefficients` (replaces `simulations/utils.py`).
- `src/context_flux_no/simulations/__init__.py` — package exports.
- `src/context_flux_no/simulations/{dataset,utils,pdesolve,pyclaw_utils}.py` — deleted.
- `src/context_flux_no/data/sources.py` — `dataset_record_ids`; `dataset_id` prefers record ids.
- `src/context_flux_no/waveforms/grf.py` — `means`/`scales` default to `None` (F11).
- `src/context_flux_no/simulations/pde/burgers.py` — `ViscousBurgers1D.solve` takes `(x_spans, Nxs)`, writes the last frame, samples cell centres (F8).
- `scripts/data/generate_dataset.py` — on `simulations.generate_dataset`; `backend`/`bc`; passes the IC config into the record.
- `scripts/data/configs/config.yaml`, `scripts/data/configs/backend/{default,classic,sharpclaw}.yaml` — backend group, `bc` key.
- `scripts/data/README.md` — families, ICs, backends, `dataset.json`.
- `tests/unit/test_generate_dataset.py` — new.

Commit 3 — docs:
- `docs/decisions/0011-solver-backends-and-dataset-provenance.md`, `docs/design/simulation-backends.md` — new.
- `docs/changes/2026-10-25-phase2a-pyclaw-backend.md` — this document.
- `docs/roadmap.md` (2a split into 2a-pyclaw done / 2a-trixi pending, 4.3 done, 4.4 note, Done rows), `docs/architecture.md`, `docs/verification.md` (V18a, V25), `docs/findings.md` (F8 update, F11), `docs/glossary.md`, `docs/index.md`.

## Design

- **`solve()` on the PDE class stays, the backend is an argument** rather than scripts calling backends directly: `generate_dataset`, the regression inputs and notebooks keep one call shape, and `backend=None` is exactly the old behaviour. Rejected: removing `solve()` from the classes (every caller changes for no gain before Phase 4.1).
- **`PyClawSpec` carries per-scheme solver options** (`classic_options`, `sharpclaw_options`) so the Classic defaults each family had (MC + CFL 0.5/0.9, van Leer, transverse waves) survive bit for bit while SharpClaw gets PyClaw's own defaults; `PyClawBackend(**solver_overrides)` applies last. Rejected: one global option set (would have changed Classic output for some family).
- **`Grid.centers` uses PyClaw's `lo + (i + ½) dx` arithmetic**, not `linspace` midpoints — the two differ in the last ulp and the goldens would not have matched.
- **Interim base class, not `PDEFamily`** (ADR-0011 §3): ~40 lines, dies in 4.1; the alternative coupled 2a to the `fv/` flux interface Joon is about to write.
- **BC names, not objects** (ADR-0011): strings are what PyClaw/Trixi consume and the record stores; objects arrive with ghost-cell padding in 2b.
- **Agreement = mutual convergence** with a ratio threshold of 1.5 against a measured ≈ 4: loose enough not to flake on a different BLAS, tight enough to catch a wrong flux (a sign error gives O(1) distance that does not shrink).
- **Record id is local, not a content hash** (ADR-0011): identity without reading the data; a digest can go in `extra` later.
- **`boundary_conditions` per axis-side with masks** exactly as The Well's reader reconstructs them (`_padding_bcs`), one `*_periodic` group when periodic; `extrap` is stored as the Well's `open`.
- **Trixi deferred, not stubbed**: no `simulations/trixi.py` placeholder — an untestable module is worse than a documented hand-off (this document §Trixi hand-off, `design/python-julia-repo.md`).
- **`tests/verification/` created now** (conftest with `x64`) so the 2-D agreement case has a home; V10–V17 land there in 2f.

## Bugs fixed

- `GaussianRandomField(means, scales)` required arguments since `b918750` while every 1-D IC config omits them: the data script could not build `grf`/`step` ICs. Defaults restored (F11).
- `ViscousBurgers1D.solve` took `(x_span, Nx)` while `generate_dataset` passes sequences, so the `burgers_1d` config could not run; it also never wrote the last frame (`while save_idx < Nt`) and evaluated the IC on left edges (F8). Fixed; `CubicFlux2D`'s Rusanov stepper untouched.
- Five of seven PDE classes did not implement the base the writer needs (`parameters`, `n_spatial_dims`, `field_rank_names`, `solution_to_well_schema`), so only `cubic_1d` and `euler_2d` could be generated. All seven conform now (`scripts/data/generate_dataset.py` verified on sine, cubic, Burgers, shallow water with N = 32).
- `riemann_cubic_1D` raised a NumPy `invalid value` warning on every call from the division inside `np.where`; masked with `errstate` (no numerical change — the goldens still match).

## Findings

- **F8 updated** — `ViscousBurgers1D` last-frame and centre bugs fixed here; regeneration of Burgers/2-D cubic still Phase 4.
- **F11 (new)** — the `GaussianRandomField` signature change of 2026-08-29 broke the 1-D IC configs; no dataset in the paper is affected (all predate it), but any attempt to regenerate since then would have failed. Appended to `docs/findings.md`.
- The Classic-vs-SharpClaw ratios (≈ 4) say the ICLR data are second-order Classic solutions; a future comparison against SharpClaw-generated data would see O(Δx²) differences near smooth features and scheme-dependent shock profiles — worth remembering when 4.4 switches backends (recorded in `design/simulation-backends.md` §Verification protocol).

## Tests

Fast tier (`JAX_PLATFORMS=cpu uv run pytest tests/unit`, now 59 tests, ~65 s; the simulation ones need `clawpack`, which builds here after `apt-get install gfortran`):
- `test_pyclaw_backend.py` — V25 bit identity on 5 families; SharpClaw runs on 5 families and starts from the identical state; `Grid` == PyClaw centres; `normalize_bcs` rules; input validation; `describe`; a non-periodic run.
- `test_backend_agreement.py` — V18a on the four 1-D families (ratios ≥ 1.5 over 64/128/256).
- `test_generate_dataset.py` — Hypothesis round trip of `DatasetRecord`; sidecar naming; writer layout incl. both BC layouts and `finalize`; `generate_dataset` → store + `dataset.json` → `ZarrWellDatasetSource` → `dataset_id` from the record.

Verification tier (`JAX_PLATFORMS=cpu uv run pytest tests/verification -m slow`): `test_crosscode.py::test_euler_2d_classic_vs_sharpclaw` (V18a, 2-D).

`scripts/import_all.py`: 0 failed, 0 skipped (clawpack present). `scripts/check_configs.py`: the same 14 dimension-mismatch pairings as before.

## Open issues

- **2a-trixi** — needs Julia; hand-off above. Until then V18 is not runnable and ADR-0004's "both oracles before WENO5" is satisfied by PyClaw alone for 2b–2e(C); proposal to amend ADR-0004 accordingly is in the Phase 2b plan.
- **`CubicFlux2D` has no PyClaw path** (no 2-D Python kernels); options: a Fortran `rpn2` for a scalar cubic flux (~40 lines, built with the clawpack extension machinery), Trixi, or `JaxBackend` (4.7). Decide at 4.4.
- **`ViscousBurgers1D` is parabolic** — outside ADR-0001's hyperbolic scope until tier-3 `viscous_flux`; its own stepper stays.
- **`sources.py` prints metadata on every construction** (`print(metadata_name, val, len(val))`) — pre-existing noise, left alone (not this branch's file to clean).
- **Python-kernel SharpClaw is slow** (pure-Python WENO): fine for tests and small regen, not for 100 × 100 trajectories at N = 100 in 2-D. Fortran kernels for cubic/sine (Clawpack ships `riemann.cubic_1D`) would fix it; not needed yet.
- **`zarr` shard-shape inference warning** (`shards="auto"`, pre-existing) — harmless; pin explicit shards when the dataset sizes are settled in 4.4.
- **Goldens policy** — `goldens_simulations/` are regenerated only with a change-document entry (as the regression goldens).

## Review round 1 (2026-10-25)

Joon's Review notes below. Resolutions, one commit per logical group (`code-conventions` §1.3):

- **`grid.py`** — the review commit's `__init__` read `self.x_spans`/`self.Nxs` before assigning them (every `Grid(...)` raised `AttributeError`), and `from_sequences` still had callers at `pde/base.py:86` and `test_pyclaw_backend.py:71`. Fixed: arguments are normalized, validated, then assigned. The `__init__`-over-`__post_init__` form is kept. Instruction gap closed in `CLAUDE.md` (a *Modifications* entry is committed with the touched tests green).
- **`pde/base.py`** — renamed `AbstractPDEFamily`; tier 0 grows in place (ADR-0001 amended, roadmap 4.1 reworded to the conformance check). `solution_to_well_schema` and `WellSchema` deleted; `WellWriter.write_trajectory(idx, u, scalars)` splits the state by the `field_rank_names` stored in the root attrs at creation (checked on rank-0 and rank-1 layouts).
- **`record.py` (both records)** — `replace`/`to_dict`/`to_json`/`from_json` deleted; callers use `copy.replace` and `dataclasses.asdict`; `read(path)`/`from_dict` remain (a `str` input would be ambiguous between JSON text and a path, so no polymorphic loader). `create` stays as the named constructor that fills provenance; the dataclass constructor stays the all-fields one `from_dict` needs. `sidecar_path` is a function in `writer.py` (store-layout knowledge). One JSON encoder: `utils/serialization.py::dumps`. `utils/` package: `arrays.py` (was `utils.py`), `provenance.py` (was top-level), `serialization.py`. Persistence verbs recorded in `coding-standards.md`.
- **`writer.py`** — `__len__`, `__getitem__` and `open` deleted: reading back is `ZarrWellDatasetSource`'s job and the record is read from the sidecar; `record` is no longer optional.
- **`pyclaw.py`** — the >1-D Python-kernel refusal is PyClaw's constraint, cited in the docstring and the error (`pyclaw/classic/solver.py:215`, `pyclaw/sharpclaw/solver.py:890`, `:975`; clawpack 5.14.0).
- **`generate.py`** — `except Exception` narrowed: PyClaw raises bare `Exception` for CFL/step-limit/invalid-state (`pyclaw/solver.py:651,664`, `controller.py:228-234`), so `PyClawBackend.solve` converts those (and a non-finite final state) into `simulations.SolverFailure(backend, reason)`; `generate_dataset` catches only `SolverFailure` and counts it; anything else propagates. Test: `test_blow_up_is_a_solver_failure`.
- **`waveforms/` → `simulations/initial_conditions/`** — pure move (`git mv`; config `_target_`s updated; all seven IC configs instantiate). The `InitialCondition.sample(key, grid, layout, bcs)` contract and the IC-config cleanup wait for tier 0 (roadmap 4.2), since the samplers then need `layout`/`bcs` from the family.
- *(further rows resolved below as their commits land)*

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `simulations/grid.py` — `Grid` | Logic is clean, but some changes to be made. <ul><li>American Englished to be used. <li>If we want to restict the types of a generic constructor input, the correct place would be in the `__init__` function or the `converter` argument of `equinox.field` if we are using `equinox.Module`s and the conversion is simple. Not sure why we need to reach for `__post_init__`. <li> `from_sequence` seems redundant with the new constructor. <li>`dxs` and `centers` share the dx computation logic, but letting it slide since `zip(self.x_spans, self.Nxs, self.dxs)` does not seem elegant either. <ul> |<ul><li>Changed "centre" -> "center" <li> Moved contents of `__post_init__` into `__init__`. <li> `from_sequence` removed. <ul> |
| `simulations/backend.py` — `SolverBackend`, BC names, `normalize_bcs` | Looks good. | None |
| `simulations/pyclaw.py` — `PyClawSpec`, `PyClawBackend` | Looks good with one question and a minor correction. `build_solver` rejects python kernels for >1D problems. Speed concern, or PyClaw constraint? If the latter, cite relevant documentation. | Dictionary unpacking changed to dictionary union. |
| `simulations/pde/base.py` — minimal interface + default `solve()` | Looks good, but maybe it might be better to change the roadmap a bit and start introducing relevant parts of the PDEFamily abstraction to avoid doing similar work twice? Also `solution_to_well_schema` might be the writer's job, assuming the writer has access to the PDE object as well (for the field names, ranks, etc; or read in from solution metadata)? | To be discussed with Claude. |
| `simulations/pde/{cubic,sine_1d,shallow_water,burgers,euler}.py` — classes on the base, `pyclaw_spec`; `ViscousBurgers1D` fixes | Confirmed. | None |
| `simulations/record.py` — `DatasetRecord` | Logic seems okay. Couple thoughts. <ul><li> `create` essentially is the constructor. Need to make into a separate classmethod? <li> `replace` and `to_dict` are superthin wrappers around `copy.replace` (use `copy` not `dataclasses`) and `asdict`. Do we really need these wrappers? <li> Want consistent namings across codebase for savind and loading files. Is `read`/`write` our current convention (if none exist, decide now). <li> Three class methods to load objecty from dict, json, and text file. Why not a single method using `match`? Argue your preference - small classmethods vs one polymorphic. <li> Not a big fan of staticmethods and this logic is not specific to the class either. Separate out into a freestanding function in the same file, or if to be used in multiple places in the codebase, into some utilities submodule. <li> Is Json encoding with `_default` pattern that is to be used across the codebase? If so, better to create a wrapped version of `json.dumps` to reuse. <ul>| To be discussed with Claude. |
| `simulations/writer.py` — `WellWriter`, `boundary_conditions` | Do we need `__len__` and `__getitem__` on the Writer? When I was first writing the code, I conflated its responsibility with the datasource, hence these methods. How about in the current refactored state? | To be discussed with Claude. |
| `simulations/generate.py` — `generate_dataset(backend)` | Would it be possible to tighten `except Exception` to a more concrete exception type? That was a quick hack from my prototyping. | `Grid` instantiation fixed to account for the new constructor. |
| `simulations/__init__.py`, deleted `dataset.py` / `utils.py` / `pdesolve.py` / `pyclaw_utils.py` | Confirmed. | None |
| `data/sources.py` — `dataset_record_ids`, `dataset_id` | `dataset_id` returns `None` if `dataset_record_ids` is an empty list, corresponding to the case of any dataset not having an id. Intended behavior? Thought a source should have an id in any case, but may be mistaken. | To be discussed with Claude. |
| `waveforms/grf.py` — defaults (F11) | Confirmed. `waveforms` might have to be renamed and moved somewhere else in the module hierarchy since it pertains to the generation of initial conditions for simulations. Suitable location and place? | To be discussed with Claude. |
| `scripts/data/generate_dataset.py`, `configs/config.yaml`, `configs/backend/*.yaml`, `README.md` | Looks good. Also possible to clean up the initial condition configs, by bring ahead relevant parts of the `PDEFamily` refactor and reorganizing `waveforms`? Or defer later in the roadmap? | To be discussed with Claude. |
| `tests/unit/goldens_simulations/*.npz`, `make_goldens_simulations.py`, `test_pyclaw_backend.py` | Confirmed. | Golden tests to run when the cluster is back online. |
| `tests/unit/test_backend_agreement.py`, `tests/verification/*` | Confirmed. Maybe `l1_distance` needs a better name? Something like `l1_sharpclaw_classic`? Also have we laid out the contract for the different types of tests (regression / unit / verification / rest) | To be discussed with Claude. |
| `tests/unit/test_generate_dataset.py` | Confirmed. | None |
| `docs/decisions/0011-*.md`, `docs/design/simulation-backends.md` | Looks good. | None |
| `docs/roadmap.md`, `architecture.md`, `verification.md`, `findings.md`, `glossary.md`, `index.md` | Had a readthrough. | None |

