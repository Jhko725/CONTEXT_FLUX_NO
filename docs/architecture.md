---
type: architecture
status: current
updated: 2026-10-24
sources: [code]
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
conditioning mechanism (hypernetwork) specialises a target network, and the target network
— a finite-volume *flux* network — predicts the next state. Families: cubic, sine and
Burgers scalar laws, shallow water, 2D Euler. Baselines: DISCO, DPOT, FNO variants.

## Module map — after Phase 1 (`phase1-prune`, 2026-10-24); struck-through = removed in Phase 1

```
src/context_flux_no/
├── models/
│   ├── multiphysics/
│   │   ├── abstract.py          AbstractMultiphysicsOperator: __call__(u, args) -> (u_next, aux); rollout()
│   │   ├── hyperfluxfno/        the method (four generations side by side)
│   │   │   ├── hyperfluxfno.py  HyperFluxFNOLocal · HyperNeuralOperator · HyperNeuralOperatorv2 ·
│   │   │   │                    ContextAppendedFluxNO; InterfaceReconstructor; FluxModel; ContextConditionedFluxModel
│   │   │   ├── encoders/        ViTEncoder · TRecViTEncoder · ~~DPOTEncoder~~
│   │   │   ├── target_networks/ FluxNO · NDFluxNO · UNet · FNO · ConvNeXtV2 · ~~LSKFluxNO~~; each does its own FV update
│   │   │   └── utils.py         make_encoder / make_target_network (string dispatch)
│   │   ├── disco/               DISCO baseline (own HyperNetworkHead, UNet blocks byte-identical to target_networks/unet.py)
│   │   ├── dpot.py              DPOT baseline
│   │   ├── fno.py, fluxno.py    NaiveFNO · SpatiotemporalFNO · NaiveFluxNO (thin wrappers)
│   │   └── ~~wrapper.py~~
│   └── fno.py                   FNO — the one real FNO implementation          (~~fluxfno.py~~)
├── nn/                          attention (fork of eqx), hypernetwork.HypernetworkHead, embedding, position_encoding,
│                                operators/fourier (SpectralConv, Fourier), ssm, convtranspose, misc, … (~~vit.py~~)
├── training/                    building blocks only (ADR-0010; imports nothing from runs/ or experiment): trainer.py (pure loop, injected side effects),
│                                loggers.py, checkpoint.py (OrbaxCheckpointer, whole TrainerState), loss.py (OneStep/Denoising/Pushforward),
│                                loader.py (DEPRECATED: SegmentLoaderBackground only, until Phase 4), io.py (legacy load_model)  (~~dataset.py~~, ~~batching.py~~)
├── runs/                        run identity: record.py (RunRecord, index columns), run.py (Run directory, RunLayout), index.py (index/select/load_run/load_state),
│                                wandb_mirror.py (wandb as a mirror)
├── experiment.py                config → objects: build_model/build_optimizer/state_template, what the training scripts share (Run in Hydra's dir, resume/warm_start)
├── provenance.py                git_state(), package_version()
├── data/                        sources.py: WellDatasetSourceBase · ZarrWellDatasetSource (current) · TheWellDataSource (re-implements base)  (~~transforms.py~~)
├── simulations/                 pde/{burgers, cubic, euler, shallow_water, sine_1d, base}: 2 classes on AbstractHyperbolicConservationLaw,
│                                5 plain eqx.Modules (~~Euler1D~~), all returning (u, t, xs); pdesolve.py (PyClaw wrapper); utils.generate_dataset;
│                                dataset.py ZarrWellDataset (writer); pyclaw_utils.py
├── waveforms/                   GaussianRandomField{,1D,2D}, PeriodicRandomStepFunction{1D,2D}, MultichannelWaveform, TruncatedFourier1D
├── plotting/, metrics.py, utils.py, custom_types.py
scripts/
├── training/  train_multiphysics.py (zarr) · train_the_well.py — data + loaders only, on experiment.train; configs/ (22 model, 3 data, 2 training, 3 wandb yamls; config.yaml + config_the_well.yaml)
├── backfill_run_records.py   legacy ICLR checkpoints → metadata.json
└── data/      generate_dataset.py; configs/
notebooks/    data_generation/ evaluation/ misc/ nn/ training/ — evaluation logic (compute_metrics ×7, metrics_long_rollout ×4) lives here; 5 legacy notebooks removed
tests/        5 example files + test_nn_hypernetwork.py; tests/unit/ (key hygiene; loggers, checkpoint, trainer, run_record, run, experiment, backfill — 37 tests); tests/regression/ machinery (goldens pending)
third_party/icon   submodule → Jhko725/icon-baseline (upstream ICON + Joon's HDF5 data path); not imported by src   [0.5 done]
```

Full evidence: `assessment-2026-09-26.md`.

## Data flow of one training step — as found

```
grain: ZarrWellDatasetSource ─► shuffle/repeat/batch ─► host prefetch ─► device_put ─► thread prefetch
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
  ic/            Phase 4  InitialCondition samplers over waveforms/
  sim/           Phase 2/4  SolverBackend: PyClawBackend · TrixiBackend · JaxBackend; dataset writer; DatasetRecord
  models/incontext/  Phase 3  InContextOperator ⊃ InContextFluxOperator; conditioners; encoders
  models/baselines/  DISCO · DPOT · NaiveFNO …
  training/      Phase 5 ✓ Trainer · losses · loggers · checkpoint   (building blocks)
  runs/          Phase 5 ✓ RunRecord · Run · index · wandb mirror     (run identity)
  experiment.py  Phase 5 ✓ config → objects; Phase 3/7 build() lands here
  evaluation/    Phase 6  rollout · metrics · tables · figures
  configs/       Phase 3/7  dataclass configs + ConfigStore
reference/trixi/  Julia project, one elixir per family (reference oracle, fallback generator)
scripts/        generate.py · train.py · evaluate.py · bench_loader.py · check_configs.py · check_docs.py
tests/          unit/ · property/ · regression/ · verification/
third_party/icon
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

As found: five example-test files on the baselines and nn utilities. Planned (ADR-0008):
four tiers — `unit/`, `property/` (Hypothesis + chex; invariants), `regression/` (goldens
from the ICLR checkpoints, CPU), `verification/` (oracle-based numerics, `x64`, slow). The
evidence that each numerics claim is tested is `verification.md`.

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
| Four-tier testing | ADR-0008 |
| Tracks C/M/R | ADR-0009 |
| Run identity: run directory, `RunRecord`, injected side effects, wandb as mirror, resume | ADR-0010, `design/run-identity.md` |
| Smaller decisions (D1–D4, D15, D16) | `roadmap.md` tables and change documents |
