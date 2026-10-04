---
type: verification
status: current
updated: 2026-10-04
sources: [claude]
---

# Verification matrix

What proves each numerics claim. One row per claim: the test that carries it, the oracle it is
compared against, the tolerance, and the status. Rows are added when a claim is made in code or in
a design document and flipped to *green* (with the commit) when the test passes. A claim with no
row is unverified. Tiers: `property` (fast, every commit) or `verification` (`-m slow`, `x64`).

| # | Claim | Test | Oracle | Tolerance | Tier | Status |
|---|---|---|---|---|---|---|
| V1 | `flux_divergence` with periodic BC conserves `∑u Δx` to round-off | `tests/property/test_fv_core.py::test_conservation_periodic` | identity | `atol 1e-12` (x64) / `1e-6` (fp32) | property | planned |
| V2 | With non-periodic BC, `d/dt ∑u Δx = F_in − F_out` | `…::test_conservation_boundary_flux` | identity | as V1 | property | planned |
| V3 | Every classical `NumericalFlux` is consistent: `F̂(u,u) = f(u)` | `…::test_flux_consistency` | `PDEFamily.flux` | `rtol 1e-12` | property | planned |
| V4 | `Rusanov`, `GodunovScalar` are monotone (nondecreasing in `u_L`, nonincreasing in `u_R`) | `…::test_monotone_flux` | sign of finite differences | — | property | planned |
| V5 | `pad_ghost_cells`: interior unchanged; `Periodic` == wrap; `Reflective ∘ Reflective` == id | `tests/property/test_boundary.py` | identity | exact | property | planned |
| V6 | N-D stepping equals 1D stepping on a field constant along transverse axes | `tests/property/test_nd.py` | 1D path | `rtol 1e-12` | property | planned |
| V7 | `rotate_to_axis ∘ rotate_from_axis == id`; Euler `f_y` == rotated `f_x` | `tests/property/test_nd.py` | identity | exact | property | planned |
| V8 | SSPRK3 on `u' = λu` matches the RK3 amplification factor; `dt=0` is identity | `tests/property/test_integrate.py` | closed form | `rtol 1e-12` | property | planned |
| V9 | WENO5 weights sum to 1; exact on polynomials of degree ≤ 4 with weights → linear optimal | `tests/property/test_weno.py` | closed form | `rtol 1e-10` | property | planned [R] |
| V10 | WENO5 + SSPRK3 is fifth order in space on smooth advection and pre-shock Burgers | `tests/verification/test_eoc.py` | `AnalyticSolution` | EOC ≥ 4.8 over 4 grids | verification | planned [R] |
| V11 | SSPRK3 is third order in time | `tests/verification/test_eoc.py::test_temporal_order` | `AnalyticSolution` | EOC ≥ 2.9 | verification | planned |
| V12 | Rusanov is first order | `tests/verification/test_eoc.py::test_rusanov_order` | `AnalyticSolution` | 0.9 ≤ EOC ≤ 1.1 | verification | planned |
| V13 | MMS: any `f(u)` + implied source reproduces the manufactured solution at design order | `tests/verification/test_mms.py` | `AnalyticSolution` + source | as V10 | verification | planned |
| V14 | Burgers shock speed (Rankine–Hugoniot) and rarefaction (entropy solution, no expansion shock) | `tests/verification/test_riemann.py` | `fv/riemann.py` exact | `L1 ≤ C·Δx` | verification | planned |
| V15 | Cubic and sine flux Riemann problems produce the correct composite waves | `tests/verification/test_riemann.py::test_nonconvex` | Osher/convex-hull exact | as V14 | verification | planned |
| V16 | Sod and Lax tubes (Euler); SW dam-break | `tests/verification/test_riemann.py` | Toro exact / SW exact | as V14 | verification | planned |
| V17 | Non-periodic: inflow/outflow advection; Sod with extrapolation; wall shock reflection | `tests/verification/test_bc.py` | exact | as V14 | verification | planned |
| V18 | Backend agreement: `PyClawBackend` vs `TrixiBackend` on every family | `tests/verification/test_crosscode.py::test_reference_backends` | each other | from mutual convergence study | verification | planned [C] |
| V19 | Three-way: own WENO5 vs SharpClaw vs Trixi | `…::test_three_way` | as V18 | as V18 | verification | planned [R] |
| V20 | 2D Euler vs JAX-Fluids on one Lax–Liu configuration | `…::test_jaxfluids_euler` | JAX-Fluids | as V18 | verification | planned [R] |
| V21 | `vmap` == Python loop; `jit` == eager, for every `FluxPredictor` | `tests/property/test_transforms.py` | each other | `rtol 1e-6` | property | planned |
| V22 | ICLR checkpoints reproduce their recorded forward pass and 5-step rollout | `tests/regression/test_regression.py` | goldens | `rtol 1e-5, atol 1e-6` (CPU) | regression | planned (Phase 0.8) |
| V23 | Mandatory statement: all of V1–V21 hold at N = 1, 2 and (tiny) 3 | parametrised | — | — | both | planned |

## Tolerance notes

- fp32 cannot demonstrate fifth order beyond ~2 refinements (error floor ~1e-7); verification uses `jax_enable_x64`.
- Cross-code tolerances are set from a mutual convergence study (DG and FV dissipate differently near shocks), recorded here when measured.
- Regression goldens are generated and compared on CPU (`JAX_PLATFORMS=cpu`) for determinism.
