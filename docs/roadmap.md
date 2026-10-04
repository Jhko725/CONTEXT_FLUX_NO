---
type: roadmap
status: current
updated: 2026-10-04
sources: [joon, claude]
---

# Roadmap

The single current plan for the cleanup of `context_flux_no`. Kept *correct now*: finished
items move to the **Done** ledger at the bottom (append-only, chronological, one row per item,
linking the branch's change document); parked work is listed with its reason. Updated in the
same commit as the work that changes it. History: `git log -p docs/roadmap.md`.

Decisions are ADRs in `decisions/`; design notes in `design/`; branch-level detail in
`changes/`; the pre-cleanup state of the repository is `assessment-2026-09-26.md` (P1–P10
are Joon's pain points, referenced below).

## Principles

1. Every phase ends with a repo that trains the current 2D Euler model and loads the ICLR checkpoints.
2. Refactor only behind the regression harness built from the ICLR checkpoints.
3. Delete by default; the `iclr2027-submission` tag is the archive. No `legacy/` directory.
4. Fix interfaces now; defer implementations that are undecided.
5. Move code, don't rewrite — except in Phases 2–3, where the rewrite is the point.
6. Classical and learned numerics share one interface (`FluxPredictor` + `Integrator`); simulators and models are built from the same `fv/` package.
7. In-house numerics is verified against two independent external solvers before its output trains anything.

## Tracks (ADR-0009)

| Track | Meaning | Phases |
|---|---|---|
| **[C] Common** | needed by everything; no research until done | 0, 1, 5, 2 (C items), 3 |
| **[M] MVP** | smallest addition that lets post-submission experiments run on clean infrastructure | 4 (M items), 6 (M items); then FiLM as the first research item |
| **[R] Rest** | valuable, not blocking | WENO5 + full verification, `JaxBackend`, figures, 7 |

Rule: [R] is not started while a [C] or [M] item is open — except WENO5, which may run in
its one-week time-box alongside Phase 3.

```
[C]  P0 ─► P1 ─► P5 ─► P2a ─► P2b ─► P2c ─► P2d ─► P2e ─► P2g ─► P3 ─┐
                        └─► [R] WENO5 (time-box, parallel to P3) ─────┤
[M]                                                    P4 ─► P6 ─► research resumes
[R]                                   JaxBackend ─► full verification ─► figures ─► P7
```

---

## Phase 0 — Freeze, safety net, hygiene  [C]  (current; branch `phase0-freeze`)

Change document: `changes/2026-10-04-phase0-freeze.md`. Mechanical items by Claude; the
substantive ones (0.2, 0.5–0.8) by Joon.

| # | Item | Done when | Status |
|---|---|---|---|
| 0.1 | Tag `iclr2027-submission` at `5f67a31`; branch `phase0-freeze` | tag on origin | tag local only (proxy blocks tag pushes) — Joon pushes |
| 0.2 | Reproduce the five breakages (assessment §3) | each failure seen once | |
| 0.3 | Untrack `.pyc`, `pyclaw.log`, `icon/test_traj_seq.h5`; fix `.gitignore` | `git ls-files` shows none | |
| 0.4 | Tag `pre-nbstripout`; install `nbstripout` filter; strip outputs | `notebooks/` < 1 MB | |
| 0.5 | `icon/` → `third_party/icon` submodule (upstream if unmodified) | `git submodule status` lists it | |
| 0.6a | `bias_hyperinit → bias-hyperinit` in 10 yamls; runtime check; default `bias-hyperinit` (decision D15); unit test | `grep bias_hyperinit` empty; test passes | |
| 0.6b | Delete 3 `HyperFluxFNO` configs + `train_multiphysics_2d.py` | `grep HyperFluxFNO\b` empty | |
| 0.6c | `data/__init__` exports | `from context_flux_no.data import TheWellDataSource` works | |
| 0.6d | Delete/rename the misnamed `lskflux` config | a ConvNeXt config still exists | |
| 0.6e | `scripts/check_configs.py`; census → `docs/config_census_phase0.txt` | no Import/Attribute/init-string failures | |
| 0.7 | Bias-HyperInit audit of wandb runs → `findings.md` | conclusion recorded | |
| 0.8 | Regression harness: cases, inputs, goldens (CPU), tests green twice | `pytest tests/regression` green ×2 | |
| 0.9 | Tooling: dev deps, ruff format, pre-commit, CUDA as optional extra, CPU CI, pytest markers | CI green | |
| 0.10 | README, `pyproject` metadata, hello-world removed, training README fixed | — | |
| 0.11 | `docs/` tree, `CLAUDE.md`, `scripts/check_docs.py` | `check_docs.py` passes | |

Exit: `pytest -m "not regression"` green; regression tier green on CPU; `check_configs.py`
has no import/attribute/init-string failures; `train_multiphysics.py
model=hyperfluxno_2d_euler_v2 training.max_steps=20` runs and checkpoints.

## Phase 1 — Delete dead and legacy code  [C]  (branch `phase1-prune`)

Change document: `changes/2026-10-04-phase1-prune.md` (planned). Dependency-ordered
deletion; `ruff check` (F401) and the fast tiers after each group; regression tier at the
end of each sub-phase.

| # | Item | Done when |
|---|---|---|
| 1.1 | 14 dead symbols (assessment §5 + D1: `DPOTEncoder`, `LSKFluxNOTargetNetwork`) | grep list empty; `import_all.py` clean |
| 1.1 | Keep `AbstractTrainLoss`; wire into `Trainer` type hints | — |
| 1.2 | Legacy xarray path removed except `SegmentLoaderBackground` (+`make_segment_axis`), kept with DEPRECATED header until Phase 4 because seven evaluation notebooks still read the ICLR-era 1D datasets | `grep xarray src` → only `training/loader.py` |
| 1.2 | Legacy PDE classes return `(u, t, xs)`; `solution_to_dataset` gone | — |
| 1.3 | Safe correctness fixes, one commit each: `test_models_disco.py:286` assert; DISCO `split(key, 4)`; `HyperNeuralOperator` key reuse; `train_the_well.py` valid sampler length | regression unchanged |
| 1.3 | Deferred correctness items recorded in `findings.md` (rfftn axes, `stack_grid`, UNet keys/mask, DISCO double standardize, four face conventions) | — |

Exit: `import_all.py` and `ruff check` clean; fast and regression tiers green;
`evaluate_cubic.ipynb` reaches its first metric cell.

## Phase 5 — Run identity  [C]  (directly after Phase 1; P9, P10)

| # | Item | Done when |
|---|---|---|
| 5.1 | `RunRecord` (run id/name/URL, git SHA + dirty, full config, `ModelConfig` yaml, `DataSpec`, `dataset_id`, index columns, checkpoint path, best step/metric) | dataclass + tests |
| 5.2 | Trainer writes it to orbax `custom_metadata`, `run.json`, wandb (`config` = index columns only; full config as file; `group`/`job_type`/`tags`); checkpoint folder = run id | one training run shows all three |
| 5.3 | `training/runs.py`: `index(root) → DataFrame`, `load_run`, `from_wandb` | notebooks lose `CHECKPOINT_DICT` |
| 5.4 | Backfill `run.json` for ICLR checkpoints | — |
| 5.5 | One line of design so `Trainer` does not preclude `jax.sharding` | — |

## Phase 2 — FV core, reference backends, classical fluxes, verification  [C] / [R]

Ordered so that external references exist before any in-house numerics is written.
Conventions: `design/fv-conventions.md`. Decisions: ADR-0001 (tiered coupling), ADR-0002
(N-D), ADR-0003 (BCs), ADR-0004 (WENO5 + reference backends), ADR-0005 (three-part flux).

| # | Track | Item | Done when |
|---|---|---|---|
| 2a | C | `sim/pyclaw.py::PyClawBackend` (Classic + SharpClaw) and `reference/trixi/` + `sim/trixi.py::TrixiBackend`; both write Well-schema files for every family | backend-agreement test green |
| 2b | C | `fv/stencil.py`, `boundary.py` (Periodic, Extrapolate, Reflective, Dirichlet, Inflow), `divergence.py`, `split.py` | property tests green at N = 1, 2, 3 |
| 2c | C | `fv/integrate.py`: `RHS(hyperbolic, parabolic, source)`, ForwardEuler, SSPRK2/3, `CFLAdaptive` stub; `fv/analytic.py` | — |
| 2d | C | `FluxPredictor` with `requires`; `NumericalFlux(reconstruction, base, dissipation)`; `StencilMLPFlux` (Fused), `NDStencilFlux` | — |
| 2e | C | `RusanovFlux`, `LaxFriedrichsFlux`, `GodunovScalarFlux`, `fv/riemann.py` | — |
| 2e | R | `WENO5Flux` (FD-WENO, LF splitting, JS/Z), **time-box 1 week** — start ____ end ____; `design/weno5.md` | EOC ≥ 4.8 on scalar families |
| 2f | C | property tier for `fv/` (`verification.md` rows marked property) | green |
| 2f | R | verification suite: EOC + MMS; exact discontinuous; non-periodic; three-way cross-code (own / SharpClaw / Trixi); benchmarks notebook | `verification.md` rows green |
| 2g | C | Models migrated onto `fv/`: v2 → HNO → Local → Appended → ConvNeXt; regression after each; face-convention mismatches → `findings.md` | `grep "dt \*" src/context_flux_no/models` empty |

## Phase 3 — In-context operator hierarchy + typed configs  [C]

ADR-0006 (hierarchy, scale placement), ADR-0007 (dataclass configs).

| # | Item | Done when |
|---|---|---|
| 3a | `InContextOperator` / `InContextFluxOperator` / `FluxStepper`; conditioners `Hypernet`, `ContextAppend`, `NoCondition`; `Scaled(...)` wraps the conditioned module | property test over random (encoder × conditioner × target) |
| 3b | Dataclass configs + `ConfigStore`; `DataSpec` (shapes + `bcs`); `ModelConfig.to_yaml/from_yaml`; `load_model` via config | `hydra.utils.instantiate` unused for models |
| 3c | ICLR shim; regression runs through it; **review gate**: keep or retrain-and-drop → decision: ____ | regression green via shim |

Exit: `hyperfluxfno.py` deleted; conditioner ablation is a one-key config change.

## Phase 4 — PDE families, ICs, data regeneration  [M] / `JaxBackend` [R]

| # | Track | Item | Done when |
|---|---|---|---|
| 4.1 | M | `pde/PDEFamily` tier 0 (`flux(u, params, axis, x)`, `max_wave_speed`, `reflect`, conversions, flags); all families conform | `requires ⊆ capabilities` check passes for every config |
| 4.2 | M | `ic/InitialCondition.sample(key, grid, layout, bcs)`; one GRF, one RandomStep | property: valid for the PDE for any seed |
| 4.3 | M | `DatasetRecord`; backend-agnostic writer; `bcs` + backend in Well metadata | — |
| 4.4 | M | Regen cubic 1D · sine · Burgers · SW · cubic 2D via reference backends (backend: ____); Euler stays on Clawpack | datasets on disk with `dataset.json` |
| 4.5 | M | `TheWellDataSource` folded into base; `SegmentLoaderBackground` deleted; xarray/h5netcdf/netcdf4/dask gone from `src` | `grep xarray src` empty |
| 4.6 | M | `bench_loader.py`; GC/prefetch fixes into the loader factory (P3) | before/after numbers in change doc |
| 4.7 | R | `sim/jax.py::JaxBackend`; second regen with verified WENO5 (new `DatasetRecord` version) | cross-code agreement |
| 4.8 | R | Baselines under non-periodic data (FNO padding; DISCO mask) | — |

## Phase 6 — Evaluation module  [M] / [R]

| # | Track | Item | Done when |
|---|---|---|---|
| 6.1 | M | `evaluation/`: `rollout`, `metrics` (rel ℓ2/ℓ∞ curves, conservation drift, effective flux split by part), selection via `runs.index()`; `scripts/evaluate.py` | no `def compute_metrics` in any notebook |
| 6.2 | R | `tables`, `figures`; `scripts/figures/` regenerates every paper figure | — |

## Phase 7 — Config surface  [R]

| # | Item | Done when |
|---|---|---|
| 7.1 | One `train.py`; component groups `model/{encoder,conditioner,target\|flux,integrator}`, `data/`, `ic/`, `training/`, `loss/`; `experiment/*` presets | `check_configs.py` → 0 failures |
| 7.2 | Dependencies pruned; README final | — |

## Research items waiting on the interfaces

Land only as implementations of the new interfaces after their phase closes: FiLM
conditioner (after 3a); substencil-softmax / LSK-style flux and learned dissipation (after
2d); SSP-RK and adaptive-Δt training (after 2c); source terms and spatially varying fluxes
(after 2c/4.1); well-balanced reconstruction, compressible Navier–Stokes (`viscous_flux`,
IMEX) — see `design/fv-conventions.md` §Extensibility.

## Parked (with reason)

- Multi-device training — single GPU assumed throughout; Phase 5.5 only keeps the door open.
- Nonconservative products, incompressible Navier–Stokes — out of `fv/` scope (ADR-0001 §Extensibility).
- `CFLAdaptive` implementation (`scan` with bounded substeps vs diffrax controller) — when first needed.
- `orbax-checkpoint==0.11.25` pin — choose a version deliberately in Phase 5 and re-save the ICLR checkpoints once under the shim.
- Whether `TrixiBackend` remains a permanent generator after WENO5 is verified.

---

## Done

| When | Branch | What | Record |
|---|---|---|---|
| 2026-10-04 | `phase0-freeze` | 0.11: `docs/` tree (roadmap, architecture, index, verification, findings, glossary, assessment, ADR-0001…0009, `design/fv-conventions.md`, change documents for Phases 0–1), `CLAUDE.md`, `scripts/check_docs.py` | `changes/2026-10-04-phase0-freeze.md` |
