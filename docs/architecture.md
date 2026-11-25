---
type: architecture
status: current
updated: 2026-11-25
sources:
  - "code"
---

# Architecture

What the `context_flux_no` package is **now**, module by module, and what each part is
planned to become (marked *→ planned*, with the roadmap phase). Kept current with the code:
when a phase lands, its section here changes in the same commit. Decisions are in
`decisions/`; numerics conventions in `design/fv-conventions.md`; vocabulary in
`glossary.md`.

## Purpose

HFluxNO: in-context learning of conservation-law solvers. Given a short observed trajectory
of a PDE solution, a context encoder infers the unknown coefficients implicitly, a
conditioning mechanism (hypernetwork) specializes a target network, and the target network
— a finite-volume *flux* network — predicts the next state. Families: cubic, sine and
Burgers scalar laws, shallow water, 2D Euler. Baselines: DISCO, DPOT, FNO variants.

## Module map — after Phase 1b (`refactor-foundation`, 2026-11-25)

```
src/context_flux_no/
├── legacy/                      code that awaits its rewrite (ADR-0012): importable, otherwise untouched, outside the strict checks
│   ├── models/
│   │   ├── multiphysics/
│   │   │   ├── abstract.py          AbstractMultiphysicsOperator: __call__(u, args) -> (u_next, aux); rollout()
│   │   │   ├── hyperfluxfno/        the method (four generations side by side): hyperfluxfno.py, encoders/ (ViT, TRecViT),
│   │   │   │                        target_networks/ (FluxNO, NDFluxNO, UNet, FNO, ConvNeXtV2; each does its own FV update), utils.py
│   │   │   ├── disco/               DISCO baseline (own HyperNetworkHead, UNet blocks byte-identical to target_networks/unet.py)
│   │   │   ├── dpot.py              DPOT baseline
│   │   │   └── fno.py, fluxno.py    NaiveFNO · SpatiotemporalFNO · NaiveFluxNO (thin wrappers)
│   │   └── fno.py                   FNO, the one real FNO implementation
│   ├── nn/                      attention (fork of eqx), hypernetwork.HypernetworkHead, embedding, position_encoding,
│   │                            operators/fourier (SpectralConv, Fourier), ssm, convtranspose, misc
│   └── plotting/
├── training/                    building blocks only (ADR-0010; imports nothing from runs/ or experiment): trainer.py (pure loop, injected side effects),
│                                loggers.py, checkpoint.py (OrbaxCheckpointer, whole TrainerState), loss.py (OneStep/Denoising/Pushforward)
├── runs/                        run identity: record.py (RunRecord, index columns), run.py (Run directory, RunLayout), index.py (index/select/load_run/load_state/from_wandb)
├── experiment.py                config → objects: build_model/build_optimizer/state_template, the wandb mirror, and train(), which scripts/train.py calls
├── utils/                       arrays.py (is_floating_array, num_parameters), provenance.py (git_state, package_version),
│                                serialization.py (dumps: the one JSON encoder for records and store attributes)
├── data/                        sources.py: WellDatasetSourceBase (loss_args, dataset_id) · ZarrWellDatasetSource · TheWellDataSource;
│                                loader.py: make_loader (the one loader factory, training and evaluation order), gc_paused
├── evaluation/                  imports nothing from legacy/ (rule R12); a model reaches it through the Operator protocol
│                                metrics.py (norms, relative errors, total variation, spectra, conservation drift, rollout references and diagnostics,
│                                valid horizon, segment errors), rollout.py (Operator, evaluate_rollouts, evaluate_references, segment evaluation;
│                                accelerator side), summary.py (summaries, matching, binning; host side), cost.py (count_flops, measure),
│                                record.py (EvaluationRecord)
├── simulations/                 reference solvers + data generation (ADR-0011): grid.py (Grid), record.py (DatasetRecord), writer.py (WellWriter,
│                                Well boundary_conditions, store_channel_order), generate.py (generate_dataset); pde/{base (minimal interface until 4.1;
│                                positive_quantities, admissible), burgers, cubic, euler, shallow_water, sine_1d}: 5 PyClaw families + CubicFlux2D/ViscousBurgers1D own steppers
│   ├── backends/                base.py (SolverBackend, BC names, SolverFailure), pyclaw.py (PyClawSpec, PyClawBackend classic|sharpclaw),
│                                trixi.py (TrixiSpec, TrixiBackend via juliacall; Julia side in reference/trixi/ContextFluxTrixi/); jax.py at 4.7
│   └── initial_conditions/      base.py (InitialCondition.at_points, FromFunction, Stacked, Conserved; InitialConditionSampler.sample(key, x_spans),
│                                Multichannel), grf.py (GaussianRandomField via gstools), step.py (PeriodicStep1D/PeriodicVoronoi2D
│                                + their samplers; at least two jumps), fourier.py (TruncatedFourier1D)
└── custom_types.py
scripts/
├── train.py                  the one training entry point: the data config's `source:` node picks the source class
├── evaluate.py               named runs on a named test set → one HDF5 result file with its EvaluationRecord
├── bench_loader.py           times make_loader on a named dataset (numbers pending, roadmap 4.6)
├── training/configs/         model (22), data (3, each with a `source:` node), training (2), wandb (3) yamls; config.yaml
├── checks/                   check_docs · check_changes · check_md_math (+ markdown.py) · check_references · check_spelling ·
│                             check_containers · check_dependencies · check_docstrings; each reads text or syntax and never imports the package (rule R10)
├── docs/generate_references.py   writes docs/references.md from docs/references.bib
├── setup_julia.py            Julia ~1.12 + reference/trixi/ project via juliapkg, once per machine
└── data/                     generate_dataset.py (backend=default|classic|sharpclaw, bc=); configs/ (pde, initial_condition (sampler:), backend)
tests/        conftest.py (Hypothesis profiles; jaxtyping + beartype import hook on the clean paths), helpers.py (named tolerances, stub operators);
              unit/ (training, runs, loader, configs, imports, evaluation, admissibility, backends + goldens_simulations/);
              verification/ (backend agreement, cross-code; x64); legacy/ (the tests of legacy/, unchanged)
justfile, prek.toml           every command is a recipe; every hook runs in `just check`
```

The survey of the repository before the cleanup is `assessment-2026-09-26.md`, which was never committed to this repository.

## Data flow of one training step — as found

```
data.make_loader(source): shuffle/repeat/batch ─► host prefetch ─► device_put ─► thread prefetch
        batch u[B, T+1, C, *grid], loss_args = (dt, dx, …)
          │
          ▼
OneStepLoss / PushforwardOneStepLoss(model, batch, loss_args, key)
          │  model(u[:-1], (dt, dx)) ─► encoder(context) ─► hypernet ─► target_network(u_last) ─► u_next
          ▼                           (target network does u − dt·ΔF/dx itself, 8 copies with 4 face conventions)
TrainerState.take_step(grads) ─► optax; wandb.log (one step delayed); orbax save_pytree (BestN(1))
```

## Target layout — planned (Phases 2–7)

```
src/context_flux_no/
  fv/            Phase 2  pure numerics, N-D via `axis`: stencil · boundary · divergence · split · integrate ·
                          analytic · riemann · flux/{base (FluxPredictor, NumericalFlux), classical, neural}
  pde/           Phase 4  PDEFamily (tiered: flux, max_wave_speed, reflect, conversions | eigensystem… | riemann_flux | viscous_flux)
  simulations/initial_conditions/  2a-trixi ✓ InitialCondition contract (functions of position; backends discretize)
  simulations/   Phase 2a ✓ PyClawBackend · DatasetRecord · WellWriter · generate · TrixiBackend + ContextFluxTrixi (V18 green);  pending: JaxBackend (4.7) · PDEFamily (4.1)
  models/incontext/  Phase 3  InContextOperator ⊃ InContextFluxOperator; conditioners; encoders
  models/baselines/  DISCO · DPOT · NaiveFNO …
  training/      Phase 5 ✓ Trainer · losses · loggers · checkpoint   (building blocks)
  runs/          Phase 5 ✓ RunRecord · Run · index                  (run identity)
  experiment.py  Phase 5 ✓ config → objects incl. the wandb mirror; Phase 3/7 build() lands here
  evaluation/    Phase 1b ✓ metrics · rollout · summary · cost · record;  Phase 6.2: tables · figures
  configs/       Phase 3/7  dataclass configs + ConfigStore
reference/trixi/  Julia project (committed Project/Manifest lock; Trixi 0.17.16, Julia 1.12.7) + ContextFluxTrixi 0.1.0: equations.jl (cubic 1D/2D, sine, parametrized SW; exact LF bound), solve.jl (node_coordinates, solve, versions), test/runtests.jl (72 checks) — reference oracle, fallback generator
scripts/        generate.py · train.py ✓ · evaluate.py ✓ · bench_loader.py ✓ · checks/ ✓
tests/          unit/ · property/ · verification/
```

## Data flow of one training step — planned

```
batch ─► loss(model, batch)
          │  InContextFluxOperator(context, u, dt, dx, bcs):
          │    ctx = encoder(context)
          │    flux = conditioner(Scaled(NumericalFlux(reconstruction, base, dissipation)), ctx)
          │    u_next = integrator.step(RHS(hyperbolic = −Σ_axis div(flux(pad_ghost_cells(u, bcs))), source), u, dt)
          ▼
experiment.train: Run(dir) → metadata.json · orbax custom_metadata · wandb(index columns);  Trainer(logger, checkpointer, evaluate)
```

The same `fv/` objects — `pad_ghost_cells`, `Integrator`, `NumericalFlux` with a classical
reconstruction and base — drive `JaxBackend` for data generation, so the learned flux and
the classical flux are interchangeable plug-ins of one stepper (Principle 6).

## Tests

The fast tier holds `tests/unit`, the unmarked tests of `tests/verification` and
`tests/legacy`; the slow tier holds the tests marked `slow`. The property tier of
ADR-0008 (`tests/property`, Hypothesis and chex) arrives with the finite-volume core.
There is no regression tier (ADR-0008, amended 2026-11-25). `verification.md` is the
evidence that each numerics claim is tested, and `coding-standards.md` gives the recipes
and the contract of each tier.

## Where each decision is recorded

| Topic | Record |
|---|---|
| PDE ↔ solver coupling, tiers, `requires` | ADR-0001 |
| N-D stepping, no per-dimension classes | ADR-0002 |
| Boundary conditions as objects | ADR-0003 |
| Own WENO5 with PyClaw/Trixi as references | ADR-0004 |
| Three-part numerical flux | ADR-0005 |
| Operator hierarchy; scale wraps the conditioned module | ADR-0006 |
| Dataclass configs + ConfigStore | ADR-0007 |
| Test tiers (the regression tier removed 2026-11-25) | ADR-0008 |
| Tracks C/M/R | ADR-0009 |
| Run identity: run directory, `RunRecord`, injected side effects, wandb as mirror, resume | ADR-0010, `design/run-identity.md` |
| Reference solvers behind `SolverBackend`; `DatasetRecord`; agreement by mutual convergence | ADR-0011, `design/simulation-backends.md` |
| `legacy/` and the clean list that scopes the strict checks | ADR-0012 |
| Smaller decisions (D1–D4, D15, D16) | `roadmap.md` tables and change documents |
