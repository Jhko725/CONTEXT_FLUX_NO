---
type: design
status: agreed
updated: 2026-10-04
verified_by: joon (discussions 2026-09-26/27)
sources: [LeVeque 2002; Shu 1998/2009; Jiang & Shu 1996; Clawpack docs; Trixi.jl docs; Pareschi & Russo 2005; Audusse et al. 2004]
---

# Finite-volume conventions for `fv/`

The conventions every module in `context_flux_no.fv` (Phase 2) follows, and the reasoning behind
them. Decisions: ADR-0001 (tiers), ADR-0002 (N-D), ADR-0003 (BCs), ADR-0005 (flux parts). Code
does not restate these; it links here. Provenance: **[cited]** from the named source, **[code]**
verified in the tree at `iclr2027-submission`, **[deduced]** inference.

## 1. Grid, cells, faces

- Uniform Cartesian grid, cell-centred values `u_i` with `i = 0 … N−1` along each axis; `dx` per axis.
- Faces along `axis` are indexed `i + ½`, `i = −1 … N−1`, so a face array has **`N + 1` entries**
  along that axis; entry `k` is face `k − ½` (between cells `k−1` and `k`).
- `flux_divergence(F, dx, axis) = jnp.diff(F, axis=axis) / dx` returns `N` entries: `(F_{i+½} − F_{i−½})/dx`.
- The semi-discrete update is `du_i/dt = −Σ_axis flux_divergence(F_axis, dx_axis)_i + s_i`.

## 2. Stencils and ghost cells

- A kernel `(a, b)` means face `i + ½` sees cells `i − a, …, i + b`. A symmetric 5-point WENO5
  stencil for face `i+½` is `(a, b) = (2, 3)` for the left-biased and `(3, 2)` for the right-biased
  reconstruction; `NumericalFlux.stencil` is the union.
- `pad_ghost_cells(u, axis, (a, b), bcs, *, pde)` pads `(a + 1, b)` cells along `axis` (so that the
  `N + 1` faces including the two boundary faces all have full stencils) and a symmetric transverse
  window on the other axes when the flux is genuinely multidimensional. No `jnp.roll` anywhere: the
  padding is the single place boundary conditions enter. **[code]** This is the convention the
  `InterfaceReconstructor` already used.
- Ghost values are produced by `BoundaryCondition.fill_ghost` per (axis, side) (ADR-0003):
  `Periodic` copies the opposite end; `Extrapolate(0)` repeats the boundary cell, `Extrapolate(1)`
  continues linearly; `Reflective` mirrors and applies `pde.reflect(u, axis)` (sign flip of the
  normal velocity/momentum, identity for scalars); `Dirichlet(value | fn(x, t))` sets the ghost
  cells to the prescribed state (**[cited]** LeVeque Ch. 7 for the mirror and extrapolation rules);
  `Inflow(primitive_state)` is `Dirichlet` after `pde.to_conserved`.
- Near-boundary order: `Extrapolate(0)` is first order at the boundary; inverse Lax–Wendroff is a
  later `fill_ghost` implementation (**[cited]** Tan & Shu, JCP 229, 2010).

## 3. Dimensions (ADR-0002)

- Every `fv/` function takes `axis`. The stepper evaluates all directional fluxes on the *same*
  state and sums the divergences (unsplit, method of lines). Transverse axes are batch axes.
- Formulation is finite-*difference* WENO (reconstruct the split *flux* along a line) because on
  uniform grids it keeps fifth order in multi-D with dimension-by-dimension application; FV-WENO
  would need face quadrature for order > 2 (**[cited]** Shu, SIAM Review 51, 2009; Titarev & Toro,
  JCP 201, 2004). Consequence: FD-WENO requires a *uniform* grid (not periodicity).
- Systems with rotational invariance implement one normal-direction flux and `rotate_to_axis`
  (permute vector components); scalar laws implement `f` once or per axis.
- Not in scope: sequential Strang/Godunov splitting; genuinely multidimensional schemes (CTU).

## 4. Numerical flux (ADR-0001, ADR-0005)

```
NumericalFlux(reconstruction, base, dissipation):   F̂ = base(rec(u)) − dissipation(rec(u))
```
- `reconstruction`: stencil → face states `(u_L, u_R)` (or, for `Fused`, stencil features for a learned flux).
  Classical: `Constant` (first order), `WENO5(JS | Z)`; later `WellBalanced(...)`.
- `base`: `(u_L, u_R) → F`. Classical: `Central`, `Upwind`, `GodunovScalar` (Osher's exact flux for
  any scalar `f`: `min_{u∈[u_L,u_R]} f` if `u_L ≤ u_R`, else `max`), later `HLL`/`HLLC`/`Roe`;
  entropy-conservative fluxes if needed. Learned: an MLP over stencil features.
- `dissipation`: `(u_L, u_R) → D`. `LocalLaxFriedrichs(α = max_wave_speed)`, `GlobalLaxFriedrichs`,
  or learned, or `None`.
- `requires` is the union of the parts' requirements and is checked against the `PDEFamily`
  capabilities at build time. Learned parts require nothing.
- Consistency: `F̂(u, u) = f(u)` for every classical configuration — a property test.
- FD-WENO5 as a `NumericalFlux`: `WENO5` reconstruction applied to the Lax–Friedrichs split fluxes
  `f± = ½(f(u) ± α u)` component-wise (**[cited]** Shu 1998; Jiang & Shu 1996); in the three-part form
  this is `reconstruction = WENO5`, `base = Central`, `dissipation = LocalLaxFriedrichs`, applied to
  the split fluxes rather than to `u`. `design/weno5.md` (Phase 2) derives the weights.

## 5. Time integration

- `Integrator.step(rhs: RHS, u, dt)` with `RHS(hyperbolic, parabolic=None, source=None)` so that
  split and IMEX integrators can treat the parts differently (**[cited]** LeVeque 2002 Ch. 17 for
  fractional-step sources; Pareschi & Russo, J. Sci. Comput. 25, 2005 for IMEX-RK).
- `ForwardEuler`, `SSPRK2`, `SSPRK3` (Shu–Osher form). `CFLAdaptive(inner, cfl, max_substeps)`:
  `Δt ≤ CFL / Σ_axis α_axis/dx_axis`, bounded substeps under `lax.scan`/`while_loop`; implemented
  when first needed.
- The flux is stateless: no `dt` inside a `FluxPredictor`. Data `Δt` vs solver `Δt` is the
  integrator's concern.
- Scale normalisation (ADR-0006) wraps the flux, never the integrator: the integrator always sees
  physical `u`, so discrete conservation holds exactly.

## 6. PDE family interface (ADR-0001)

```
tier 0 (required)  field_rank_names · parameters(+ranges) · n_spatial_dims · supported_bcs
                   flux(u, params, axis, x=None) · max_wave_speed(u, params, axis)
                   to_conserved / to_primitive · reflect(u, axis)
                   has_source · has_nonconservative_terms
tier 1 (optional)  wave_speed_estimates(uL, uR, params, axis) · eigensystem(u, params, axis)
tier 2 (optional)  riemann_flux(uL, uR, params, axis)
tier 3 (optional)  viscous_flux(u, grad_u, params, axis)
```

## 7. Analytic solutions and verification

- `AnalyticSolution(x, t, params)` serves as initial condition (`t = 0`), as Dirichlet/inflow
  boundary data, and as the error reference (**[cited]** Trixi's `initial_condition(x, t)` /
  `AnalysisCallback` pattern). MMS = an `AnalyticSolution` plus the source it implies
  (**[cited]** Roache, J. Fluids Eng. 124, 2002).
- Verification claims and the tests that carry them are tabulated in `../verification.md`.
  Verification runs with `jax_enable_x64`: fifth-order EOC hits the fp32 floor after two refinements.

## 8. Extensibility (assessed 2026-09-27)

| Extension | Fits? | What it needs |
|---|---|---|
| Non-stiff source terms | yes | the `source` slot |
| Stiff sources | yes, via integrator | Strang-split or IMEX `Integrator` |
| Balance laws (SW + bathymetry) | yes | well-balanced `Reconstruction` (**[cited]** Audusse et al. 2004); "lake at rest" test |
| Spatially varying `f(u, x)` | yes | `x` in `flux` (done) |
| Nonconservative products | **no** | path-conservative schemes (Parés 2006) — out of scope; flag exists so `requires` can refuse |
| Compressible Navier–Stokes | yes | tier-3 `viscous_flux`, a `DiffusiveFlux` predictor (symmetric stencil), IMEX, wall BCs |
| Incompressible Navier–Stokes | **no** | elliptic pressure constraint; a different PDE kind with its own stepper |

## Sources

[cited] LeVeque, *Finite Volume Methods for Hyperbolic Problems*, CUP 2002 (Ch. 4, 7, 12, 15, 17) ·
Shu, *Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic
conservation laws*, ICASE 97-65 (1998) · Shu, SIAM Review 51(1) 2009 · Jiang & Shu, JCP 126 (1996) ·
Titarev & Toro, JCP 201 (2004) · Tan & Shu, JCP 229 (2010) · Pareschi & Russo, J. Sci. Comput. 25
(2005) · Audusse, Bouchut, Bristeau, Klein, Perthame, SIAM J. Sci. Comput. 25 (2004) · Parés, SIAM J.
Numer. Anal. 44 (2006) · Roache, J. Fluids Eng. 124 (2002) · Clawpack Riemann-solver docs
(clawpack.org/riemann.html) · Trixi.jl documentation (trixi-framework.org).
[deduced] The three-part decomposition mapping of existing model classes; the tiering as the
structural form of PDE-agnosticism; the `Scaled` placement argument.
