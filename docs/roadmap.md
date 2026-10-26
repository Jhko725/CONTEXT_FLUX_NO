---
type: roadmap
status: current
updated: 2026-10-25
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

## Phase 0 — Freeze, safety net, hygiene  [C]  (current; branch `phase0-freeze`; remaining: 0.7 run, 0.8 goldens)

Change document: `changes/2026-10-04-phase0-freeze.md`. Mechanical items by Claude; the
substantive ones (0.2, 0.5–0.8) by Joon.

| # | Item | Done when | Status |
|---|---|---|---|
| 0.1 | Tag `iclr2027-submission` at `5f67a31`; branch `phase0-freeze` | tag on origin | done (Joon pushed both tags 2026-10-24) |
| 0.2 | Reproduce the five breakages (assessment §3) | each failure seen once | done |
| 0.3 | Untrack `.pyc`, `pyclaw.log`, `icon/test_traj_seq.h5`; fix `.gitignore` | `git ls-files` shows none | done (`bfc5a40`) |
| 0.4 | Tag `pre-nbstripout`; install `nbstripout` filter; strip outputs | `notebooks/` < 1 MB | done (`8392757`, 16 MB → 440 KB); tag pushed |
| 0.5 | `icon/` → `third_party/icon` submodule (upstream if unmodified) | `git submodule status` lists it | done (`5fa64f2`): modified copy → `Jhko725/icon-baseline` @ `7ac3fb6` |
| 0.6a | `bias_hyperinit → bias-hyperinit` in 10 yamls; runtime check; default `bias-hyperinit` (decision D15); unit test | `grep bias_hyperinit` empty; test passes | done (`25c7494`) |
| 0.6b | Delete 3 `HyperFluxFNO` configs + `train_multiphysics_2d.py` | `grep HyperFluxFNO\b` empty | done (`25c7494`) |
| 0.6c | `data/__init__` exports | `from context_flux_no.data import TheWellDataSource` works | done (`25c7494`) |
| 0.6d | Delete/rename the misnamed `lskflux` config | a ConvNeXt config still exists | done (`25c7494`) |
| 0.6e | `scripts/check_configs.py`; census → `docs/config_census_phase0.txt` | no Import/Attribute/init-string failures | done (`25c7494`) |
| 0.7 | Bias-HyperInit audit of wandb runs → `findings.md` | conclusion recorded | script ready (`scripts/audit_bias_hyperinit.py`); run deferred — cluster down |
| 0.8 | Regression harness: cases, inputs, goldens (CPU), tests green twice | `pytest tests/regression` green ×2 | |
| 0.9 | Tooling: dev deps, ruff format, pre-commit, CUDA as optional extra, CPU CI, pytest markers | CI green | done (`fbeb7a8`, `2a65dfc`); `the-well`/`torch` dropped 2026-10-24 |
| 0.10 | README, `pyproject` metadata, hello-world removed, training README fixed | — | done (`7325817`) |
| 0.11 | `docs/` tree, `CLAUDE.md`, `scripts/check_docs.py` | `check_docs.py` passes | done (`0b0fd68`) |

Exit: `pytest -m "not regression"` green; regression tier green on CPU; `check_configs.py`
has no import/attribute/init-string failures; `train_multiphysics.py
model=hyperfluxno_2d_euler_v2 training.max_steps=20` runs and checkpoints.

## Phase 1 — Delete dead and legacy code  [C]  (branch `phase1-prune`; **landed 2026-10-24, awaiting the regression gate to merge**)

Change document: `changes/2026-10-04-phase1-prune.md` (landed). Dependency-ordered
deletion; `ruff check` (F401) and the fast tiers after each group; regression tier at the
end of each sub-phase.

**Why Phase 1 can proceed without goldens (2026-10-24).** Principle 2 guards *behaviour* of the
checkpointed models. Phase 1 deletes only symbols with zero references and legacy data paths; no
config, notebook or ablation (`ablate_encoders.ipynb` compares TRecViT vs ViT only) selects
`DPOTEncoder` or `LSKFluxNOTargetNetwork`, so no ICLR checkpoint's stored config names anything
Phase 1 removes, and `load_model` is unaffected. The one item that changes a forward path —
none; the 1.3 key-reuse fixes change *initialisation* only. Hence: the work can be done and
self-reviewed now; **merging `phase1-prune` waits for the regression tier to be green** on the
goldens, which is the mechanical confirmation of the argument above. If a golden fails after the
merge of Phase 1 work, the bisect is one branch long.

| # | Item | Done when | Status |
|---|---|---|---|
| 1.1 | 14 dead symbols (assessment §5 + D1: `DPOTEncoder`, `LSKFluxNOTargetNetwork`) | grep list empty; `import_all.py` clean | done (`359c773`, `0e3241e`) |
| 1.1 | Keep `AbstractTrainLoss`; wire into `Trainer` type hints | — | done (`9c16792`) |
| 1.2 | Legacy xarray path removed except `SegmentLoaderBackground` (+`make_segment_axis`), kept with DEPRECATED header until Phase 4 because seven evaluation notebooks still read the ICLR-era 1D datasets | `grep xarray src` → only `training/loader.py` | done (`9c16792`) |
| 1.2 | Legacy PDE classes return `(u, t, xs)`; `solution_to_dataset` gone | — | done (`24df75c`) |
| 1.3 | Correctness fixes: `test_models_disco.py:286` assert; DISCO `split(key, 4)`; `HyperNeuralOperator` key reuse; UNet `keys_bot` (F6, pulled forward; the `operatornet` copy parked 2026-10-26, F15); `jax.flatten_util` import (F9); key-hygiene test | regression unchanged | done (`134dfdb`, `518ee62`); `train_the_well.py` was already correct |
| 1.3 | Deferred correctness items recorded in `findings.md` (rfftn axes, `stack_grid`, UNet mask channel, DISCO double standardize, four face conventions) | — | done (F3, F4, F6 remainder, F2) |

Exit: `import_all.py` and `ruff check` clean ✔; fast tier green (33 tests) ✔; regression tier — pending goldens; `evaluate_cubic.ipynb` reaches its first metric cell — to be checked by Joon on the cluster.

| # | Item | Done when | Status |
|---|---|---|---|
| 1.0 | CI repair: `ruff` in the dev group; `scripts/import_all.py` | CI green on `phase1-prune` | done (`758600f`) |

## Phase 5 — Run identity  [C]  (branch `phase5-run-identity`, based on `phase1-prune`; **landed 2026-10-24**; P9, P10; ADR-0010)

| # | Item | Done when | Status |
|---|---|---|---|
| 5.1 | `RunRecord` (`metadata.json`): locally generated `run_id`, git SHA + dirty, package version, `x64`, `dataset_id`, index columns, model config, status, best/final step, resume notes | dataclass + Hypothesis round trip | done |
| 5.2 | `Run` directory = Hydra output dir; record written to `metadata.json`, orbax `custom_metadata` and wandb (`config` = index columns only; full config as file; `group`/`job_type`/`tags`); trainer loses wandb/orbax (injected `Logger`/`Checkpointer`), checkpoints whole `TrainerState`, final step forced | one training run shows all three; 28 CPU tests | done (wandb path untested against the service) |
| 5.3 | `runs/index.py`: `index(root) → DataFrame`, `select`, `load_run`, `load_state`, `from_wandb` | notebooks lose `CHECKPOINT_DICT` | code done; notebooks switch in Phase 6 |
| 5.4 | Backfill `metadata.json` for ICLR checkpoints (`scripts/backfill_run_records.py`, wandb match via the F1 audit CSV) | run once on the cluster | script + test done; **run pending** (with 0.7/0.8) |
| 5.5 | One line of design so `Trainer` does not preclude `jax.sharding` | — | done (`mesh` argument; loop never reads device values) |
| 5.6 | Exact data-stream continuation on resume (grain iterator state in the checkpoint) | resumed run bit-equal to an uninterrupted one | open; the stream restarts today (ADR-0010 §5) |
| 5.7 | `save_every` default: 1 (ICLR protocol, best over every step) or 100 (current) | Joon decides | open (F10) |

## Phase 2 — FV core, reference backends, classical fluxes, verification  [C] / [R]

Ordered so that external references exist before any in-house numerics is written.
Conventions: `design/fv-conventions.md`. Decisions: ADR-0001 (tiered coupling), ADR-0002
(N-D), ADR-0003 (BCs), ADR-0004 (WENO5 + reference backends), ADR-0005 (three-part flux).

| # | Track | Item | Done when |
|---|---|---|---|
| 2a-pyclaw | C | `simulations/pyclaw.py::PyClawBackend` (Classic + SharpClaw) behind `SolverBackend`; Classic bit-identical to the ICLR `solve()`; writes Well-schema stores for every family | V18a green, V25 green — **done 2026-10-25** (`phase2a-pyclaw-backend`) |
| 2a-trixi | C | `simulations/trixi.py::TrixiBackend` (juliacall, in-process; `trixi_spec` on six families; `InitialCondition` contract so each backend evaluates the IC itself) + `reference/trixi/ContextFluxTrixi/` (Julia package; layout per `design/python-julia-repo.md`) | V18 green — **done 2026-10-26** (`phase2a-trixi-backend`; Python side 2026-10-25, Julia package and V18 ratios 2026-10-26); CI Julia job (ledger 6) pending |
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
| 4.1 | M | tier-0 conformance: every family implements `flux(u, params, axis, x)`, `max_wave_speed`, `reflect`, conversions, flags on `AbstractPDEFamily` (the class exists since 2a; the numerics methods arrive with 2b–2e, ADR-0001 amendment) | `requires ⊆ capabilities` check passes for every config |
| 4.2 | M | `ic/InitialCondition.sample(key, grid, layout, bcs)`; one GRF, one RandomStep | property: valid for the PDE for any seed |
| 4.3 | M | `DatasetRecord`; backend-agnostic writer; `bcs` + backend in Well metadata | **done 2026-10-25** with 2a (`simulations/{record,writer}.py`; `dataset_id` from records) |
| 4.4 | M | Regen cubic 1D · sine · Burgers · SW · cubic 2D via reference backends (backend: ____ — `backend=sharpclaw` available now; cubic 2D needs Trixi or `JaxBackend`); Euler stays on Clawpack | datasets on disk with `dataset.json` |
| 4.5 | M | `TheWellDataSource` folded into base; `SegmentLoaderBackground` deleted; xarray/h5netcdf/netcdf4/dask gone from `src` | `grep xarray src` empty |
| 4.6 | M | `bench_loader.py`; GC/prefetch fixes into the loader factory (P3) | before/after numbers in change doc |
| 4.7 | R | `simulations/jax.py::JaxBackend`; second regen with verified WENO5 (new `DatasetRecord` version) | cross-code agreement |
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
| 7.2 | Dependencies pruned (`the-well` already gone; xarray/h5netcdf/netcdf4/dask after Phase 4; check `diffrax`, `tensorstore`, `fsspec`); README final | — |

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
- ~~`orbax-checkpoint==0.11.25` pin~~ — upgraded to `>=0.12.8` in Phase 5 (review round 1; typed PRNG keys restore natively). 0.12 reads the 0.11.25 ICLR checkpoints once `load_model` names the legacy checkpointable (`pytree`, renamed `state` in 0.12) — confirmed by the regression tier 2026-10-26; no re-save needed.
- Whether `TrixiBackend` remains a permanent generator after WENO5 is verified.
- DISCO `operatornet_scale` as checkpointed array leaves (F15), with the `operatornet.py` `keys_bot` fix and the hygiene xfail flip — Phase 3 (baselines), when legacy checkpoints get the leaf filled at load.

---

## Done

| When | Branch | What | Record |
|---|---|---|---|
| 2026-10-04 | `phase0-freeze` | 0.11: `docs/` tree (roadmap, architecture, index, verification, findings, glossary, assessment, ADR-0001…0009, `design/fv-conventions.md`, change documents for Phases 0–1), `CLAUDE.md`, `scripts/check_docs.py` | `changes/2026-10-04-phase0-freeze.md` |
| 2026-10-04 | `phase0-freeze` | 0.3: bytecode, log and `.h5` fixture untracked; `.gitignore` fixed | `changes/2026-10-04-phase0-freeze.md` |
| 2026-10-04 | `phase0-freeze` | 0.4: notebook outputs stripped (16 MB → 440 KB); `nbstripout` filter; `pre-nbstripout` tag (local) | `changes/2026-10-04-phase0-freeze.md` |
| 2026-10-04 | `phase0-freeze` | 0.9: dev deps (hypothesis, chex, ty, pre-commit, nbstripout); CUDA optional extra; pytest markers; ruff format + excludes; pre-commit; CPU CI; `style:` commit isolating the reformat | `changes/2026-10-04-phase0-freeze.md` |
| 2026-10-04 | `phase0-freeze` | 0.10: README; training README model names; hello-world stub removed | `changes/2026-10-04-phase0-freeze.md` |
| 2026-10-24 | `phase0-freeze` (via `main`) | 0.1/0.4 tags pushed; 0.2 breakages confirmed; 0.6a–e: `bias-hyperinit` fixed + guarded + default, dangling configs and `train_multiphysics_2d.py` removed, `lskflux`→`convnext` config, `data/__init__` exports, `check_configs.py`; 0.5: `icon/` → `third_party/icon` submodule (`icon-baseline`) | `changes/2026-10-04-phase0-freeze.md` |
| 2026-10-24 | `phase0-freeze` | 0.8 machinery: `tests/regression/` (cases, helpers, `make_inputs.py`, `make_goldens.py`, skip modes); 0.7 script `scripts/audit_bias_hyperinit.py`; `the-well` (and with it `torch`) removed; README build prerequisites | `changes/2026-10-04-phase0-freeze.md` |
| 2026-10-24 | `phase1-prune` | 1.0: CI repaired (`ruff` dev pin), `scripts/import_all.py`; hypernetwork test fixed; `jax.flatten_util` import (F9) | `changes/2026-10-04-phase1-prune.md` |
| 2026-10-24 | `phase1-prune` | 1.1: 14 dead symbols removed (incl. `DPOTEncoder`, `LSKFluxNOTargetNetwork`, D1); `AbstractTrainLoss` wired into `Trainer` | `changes/2026-10-04-phase1-prune.md` |
| 2026-10-24 | `phase1-prune` | 1.2: legacy xarray training path removed (loaders, script, 3 data configs, 5 notebooks); `SegmentLoaderBackground` DEPRECATED until Phase 4; PDE classes return `(u, t, xs)`; `solution_to_dataset` gone | `changes/2026-10-04-phase1-prune.md` |
| 2026-10-24 | `phase1-prune` | 1.3: key-reuse fixes (F5, F6) + `tests/unit/test_key_hygiene.py`; missing assert; `src/` 10,765 → 9,297 lines | `changes/2026-10-04-phase1-prune.md` |
| 2026-10-24 | `phase5-run-identity` | 5.2 (trainer side): `training/loggers.py`, `training/checkpoint.py`, `provenance.py`; `Trainer` without wandb/orbax, whole-`TrainerState` checkpoints, forced final save, non-finite stop | `changes/2026-10-24-phase5-run-identity.md` |
| 2026-10-24 | `phase5-run-identity` | 5.1, 5.3: `RunRecord`, `Run` directory object with refuse/resume/warm-start semantics, `runs.index/select/load_run/load_state/from_wandb`, `WellDatasetSourceBase.dataset_id`, `pandas` declared | `changes/2026-10-24-phase5-run-identity.md` |
| 2026-10-24 | `phase5-run-identity` | 5.2 (run side), 5.4 script, 5.5: `wandb_run.py` (mirror), `experiment.py`, Hydra `runs/<date>/<time>` run dir, `wandb=disabled|offline|online`, scripts on the new API, the_well optimizer chain in config, `backfill_run_records.py`; ADR-0010, `design/run-identity.md` | `changes/2026-10-24-phase5-run-identity.md` |
| 2026-10-24 | `phase5-run-identity` | review round 1: `orbax-checkpoint>=0.12.8` (typed-key wrapper gone); `runs/` package + `experiment.py` at package top, `training/` = building blocks; `RunLayout`; `block` removed | `changes/2026-10-24-phase5-run-identity.md` |
| 2026-10-25 | `phase5-run-identity` | review round 2: wandb mirror functions moved from `runs/wandb_mirror.py` into `experiment.py` (config-driven assembly); `runs/` = `record`, `run`, `index` | `changes/2026-10-24-phase5-run-identity.md` |
| 2026-10-25 | `phase2a-pyclaw-backend` | 2a-pyclaw: `simulations/{grid,backend,pyclaw}.py`, `PyClawSpec` on every PDE class (all seven now on the common base), Classic bit-identity goldens (V25), SharpClaw on every PyClaw family, backend-agreement tests (V18a; `tests/verification/` tier scaffold) | `changes/2026-10-25-phase2a-pyclaw-backend.md` |
| 2026-10-25 | `phase2a-pyclaw-backend` | 4.3: `DatasetRecord`, `WellWriter` (Well `boundary_conditions` group, `dataset.json`), `generate_dataset(backend)`, data-script `backend` group, `dataset_id` from records; `simulations/{dataset,utils,pdesolve,pyclaw_utils}.py` deleted; GRF defaults (F11) and `ViscousBurgers1D` (F8) fixed; ADR-0011, `design/simulation-backends.md` | `changes/2026-10-25-phase2a-pyclaw-backend.md` |
| 2026-10-25 | `phase2a-pyclaw-backend` | review round 1: Python floor 3.13 + `copy.replace` (TID251); `AbstractPDEFamily` (ADR-0001 amended: tier 0 grows in place, 4.1 = conformance); `utils/` package; `WellWriter` owns the layout and writes only; `SolverFailure`; `waveforms/` → `simulations/initial_conditions/`; agreement test in `tests/verification/`; Test-contracts and persistence-verb sections in coding-standards | `changes/2026-10-25-phase2a-pyclaw-backend.md` |
| 2026-10-25 | `phase2a-trixi-backend` | `InitialCondition`/`InitialConditionSampler` contract (ADR-0011 §Initial conditions): backends receive the function (`solve(pde, ic, …)`), samplers draw positions in the domain, `FromFunction`/`Stacked`/`Conserved`/`Multichannel`; Python side of 2a-trixi: `TrixiSpec` on six families, `simulations/backends/trixi.py` (juliacall, shared juliapkg project at `reference/trixi/`), `scripts/setup_julia.py`, V18 tests (skip without Julia), ADR-0004 amended (in-process transport), F12 | `changes/2026-10-25-phase2a-trixi-backend.md` |
| 2026-10-26 | `phase2a-trixi-backend` | Julia side of 2a-trixi: `reference/trixi/ContextFluxTrixi` 0.1.0 (cubic 1D/2D, sine, parametrized SW equations with the exact LF bound; `node_coordinates`/`solve`/`versions`; 72 Julia checks) + the committed `reference/trixi/{Project,Manifest}.toml` lock (Trixi 0.17.16, Julia 1.12.7); V18 green on six families (ratios in `verification.md`); `julia_package_version` in `describe()`; `CubicFlux2D.solve` time stepping fixed (F14); `SHARPCLAW_SSP33` in `tests/helpers.py`; F13 | `changes/2026-10-25-phase2a-trixi-backend.md` |
| 2026-10-26 | `main`, `phase1-prune` | 0.8: regression goldens from the ICLR checkpoints (7 cases; `translate_legacy_config` F1, `checkpoint_template` for DISCO's `DropPath.p` leaves); V22 green twice on `main`; on `phase1-prune` after parking the `operatornet` key fix (F15: DISCO's forward pass depends on un-checkpointed `operatornet_scale`) | `changes/2026-10-04-phase0-freeze.md`, `changes/2026-10-04-phase1-prune.md` |
