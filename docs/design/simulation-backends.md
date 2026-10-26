---
type: design
status: agreed
updated: 2026-10-25
verified_by: joon (Phase 2a ledger, 2026-10-25)
sources: ADR-0004; ADR-0011; PyClaw 5.14.0 source; the_well/docs/data_format.md
---

# Simulation backends — how to run, add and verify a reference solver

The decision is ADR-0011; this is the working reference for `context_flux_no.simulations`.
Everything is **[code]** on `phase2a-pyclaw-backend` unless marked.

## The pieces

```
simulations/
  grid.py       Grid(x_spans, Nxs): centres (PyClaw arithmetic), dxs, axis_names, to_dict()
  backend.py    SolverBackend.solve(pde, u0, grid, t_out, bcs); BCName = periodic|extrap|wall; normalize_bcs
  pyclaw.py     PyClawSpec (per PDE class) · PyClawBackend(scheme="classic"|"sharpclaw", **solver_overrides)
  record.py     DatasetRecord (dataset.json + root attrs)
  writer.py     WellWriter.create / write_trajectory / finalize / open  (Well schema + boundary_conditions)
  generate.py   generate_dataset(..., backend=None, ic_description=...)
  pde/          the PDE classes; AbstractPDEFamily (ADR-0001 tier 0, data-facing part; numerics methods arrive with fv/)
```

A family needs `n_spatial_dims`, `field_rank_names` (Well layout, state order),
`parameters` (the Well scalars) and, if PyClaw solves it, `pyclaw_spec`. The writer splits a
`[time, n_eqn, *N]` solution into the Well rank groups with the layout it was created with
(`write_trajectory(idx, u, scalars)`). `solve(ic_factory,
x_spans, Nxs, t_span, Nt, bc, *, backend=None)` is inherited: it builds the `Grid`, evaluates
`ic_factory(grid.centers)`, and runs the backend (Classic PyClaw by default).

## Running

```python
from context_flux_no.simulations import PyClawBackend, Grid, normalize_bcs
from context_flux_no.simulations.pde import CubicFlux1D

pde = CubicFlux1D(1.0, -0.5, 0.25)
u, t, xs = pde.solve(ic, [(0, 1)], [256], (0, 0.2), 20, "periodic")                       # Classic
u, t, xs = pde.solve(ic, [(0, 1)], [256], (0, 0.2), 20, "periodic",
                     backend=PyClawBackend("sharpclaw", time_integrator="SSP33"))         # WENO5 + SSP33
u, t, xs = pde.solve(ic, [(0, 1)], [256], (0, 0.2), 20, [("extrap", "wall")])              # outflow left, wall right
```

`solver_overrides` are set on the PyClaw solver after the spec's per-scheme options; useful
keys: `cfl_desired`, `cfl_max`, `limiters` (`"MC" | "vanleer" | "minmod" | "superbee"`),
`weno_order` (5, 7, 9, 11, 13, 15, 17), `time_integrator` (`"SSP104"`, `"SSP33"`, `"Euler"`,
`"RK"`), `lim_type` (1 TVD / 2 WENO), `char_decomp`. PyClaw's defaults otherwise
(`cfl_desired` 0.9 Classic / 2.45 SharpClaw-SSP104). The spec's Classic options reproduce the
ICLR settings per family (MC limiter and CFL 0.5/0.9 for the Python-kernel families, van Leer
for Burgers, transverse waves for Euler).

Generating a store:

```bash
uv run python scripts/data/generate_dataset.py pde=cubic_1d initial_condition=grf             # backend=default → Classic
uv run python scripts/data/generate_dataset.py pde=shallow_water_1d initial_condition=grf_shallow backend=sharpclaw
uv run python scripts/data/generate_dataset.py pde=euler_2d initial_condition=grf_euler_2d dataset_type=test
```

writes `<savedir>/<pde.name>/data/<split>/<name>.zarr` and `<name>.dataset.json`. Python-kernel
families (cubic, sine, shallow water) run SharpClaw's WENO in pure Python and are roughly an
order of magnitude slower than Classic; Burgers and Euler use compiled kernels.

## PyClaw families and what each can do

| Class | Riemann solver | Kernel | Classic | SharpClaw | Notes |
|---|---|---|---|---|---|
| `CubicFlux1D` | `riemann_cubic_1D` (Roe speed, Python) | Python | ✓ (MC, CFL 0.5) | ✓ | |
| `SineFlux1D` | `riemann_sine_1D` (char. speed at the mean, Python) | Python | ✓ (MC, CFL 0.5) | ✓ | |
| `ParametrizedShallowWater1D` | Roe-type, Python | Python | ✓ (MC, CFL 0.5) | ✓ | `h_floor` is solver data, not a scalar |
| `Burgers1D` | `clawpack.riemann.burgers_1D` | Fortran | ✓ (van Leer) | ✓ | |
| `Euler2D` | `clawpack.riemann.euler_4wave_2D` | Fortran | ✓ (transverse 2) | ✓ | `use_rho_v_p_ics` → `to_conserved` |
| `CubicFlux2D` | — | — | own JAX Rusanov (F8) | — | PyClaw has no 2-D Python kernels; Phase 4.4/4.7 |
| `ViscousBurgers1D` | — | — | own JAX Rusanov + FD diffusion | — | parabolic term; Phase 4 |

Adding a PyClaw family = a class with the four attributes above and a `pyclaw_spec`; adding a
backend = a `SolverBackend` subclass with `solve` and `describe` (Trixi: ADR-0004,
`design/python-julia-repo.md`).

## Boundary conditions in the store

Per axis `d` with name `x`: periodic → one group `x_periodic` (`bc_type="periodic"`, mask True
at both ends); otherwise `x_<lower>_lower` and `x_<upper>_upper` with `bc_type` `open` (our
`extrap`) or `wall`, mask True at index 0 / −1. `associated_dims=[x]`, `associated_fields=[]`,
`sample_varying=time_varying=False`. This is what The Well's `WellDataset._padding_bcs`
reconstructs per side [cited: `the_well/the_well/data/datasets.py`].

## `DatasetRecord`

| Field | Content |
|---|---|
| `id` | `<dataset>-<split>-<UTC stamp>-<8 hex>`, local |
| `family`, `dataset_name`, `split`, `created_at`, `status` (`writing` → `complete`), `seed` | |
| `n_trajectories`, `n_failed` | written count after `finalize`; failures are logged with their parameters |
| `grid`, `time`, `bcs`, `field_rank_names` | `Grid.to_dict()`, `{t_span, Nt, dt}`, per-axis `[lower, upper]`, Well layout |
| `params` | names, ranges, sampling strategy, `n_coeffs`, `n_ics_per_coeff` |
| `ic` | `{name, config}` from the Hydra IC config |
| `backend` | `describe()`: name, scheme, Clawpack version, solver overrides |
| `git`, `package_version`, `extra.solve_kwargs`, `schema_version` | provenance |

`WellDatasetSourceBase.dataset_record_ids` lists the ids of the files a source reads;
`dataset_id` hashes them (with the window parameters) — or falls back to the provisional
metadata hash for ICLR-era stores without a record.

## Verification protocol (V18a, V18, V19)

Two consistent schemes converge to the same entropy solution, so on a smooth horizon the L1
distance between them on a shared grid must shrink under refinement; with a second-order and a
fifth-order scheme the ratio per halving tends to 4. The test requires ≥ 1.5 over N = 64, 128,
256 (1-D, fast tier) and 16, 32, 64 (2-D Euler, `-m slow`). Measured 2026-10-25 (Classic vs
SharpClaw, L1 ratios per halving): cubic 3.0 / 4.2, sine 4.1 / 4.3, Burgers 3.5 / 4.1,
shallow water 4.7 / 5.0, Euler 2-D 8.3 / 3.5. Horizons: 0.05 (scalar), 0.03 (SW), 0.02
(Euler) — before any shock forms for the test ICs.

## Regenerating the bit-identity goldens

`tests/unit/goldens_simulations/*.npz` pin `PyClawBackend("classic")` to the pre-2a output.
Regenerate (`uv run python tests/unit/make_goldens_simulations.py`) **only** with a change
document entry naming the commit that changed PyClaw's output and why (a Clawpack upgrade, a
deliberate limiter change).
