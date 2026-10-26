---
type: design
status: tentative
updated: 2026-10-25
verified_by: — (written by Claude for Joon to implement Phase 2b/2c; becomes `agreed` once Joon has read and amended it)
sources: [LeVeque 2002; Shu 1998 (ICASE 97-65); Shu 2009; Gottlieb, Shu & Tadmor 2001; Shu & Osher 1988; Ketcheson 2008; Hesthaven 2018; Clawpack docs; Ketcheson, LeVeque & del Razo 2020; JAX docs; Hypothesis docs]
---

# Phase 2b/2c — the `fv/` core: what to read, what to build, what proves it

Phase 2b is `fv/stencil.py`, `fv/boundary.py`, `fv/divergence.py`, `fv/split.py`; 2c is
`fv/integrate.py` and `fv/analytic.py`. The conventions are fixed in `fv-conventions.md` and
ADR-0002/0003/0005; this document is the *working plan*: a reading path that targets exactly the
decisions these modules embody, then a module-by-module specification (signatures, shapes,
invariants, tests), then the order to build in. Provenance labels: **[cited]** from the named
source, **[code]** from the tree, **[deduced]** Claude's inference.

The guiding picture: a finite-volume step is

```
u ──pad_ghost_cells──► ũ ──stencil_windows──► W ──FluxPredictor──► F̂ (N+1 faces) ──flux_divergence──► du/dt ──Integrator──► u⁺
      (boundary.py)        (stencil.py)          (Phase 2d/2e)         (divergence.py)                 (integrate.py)
```

2b builds the arrows around the flux; 2c builds the last arrow and the analytic references that
let 2f measure orders of accuracy. The flux itself (2d/2e) plugs into the middle.

---

## Part I — Reading path (≈ 2 days of focused reading; each item says what to extract)

Ordered so that each item answers a question the next module raises. Chapter numbers are for the
editions named in Sources.

### I.1 The finite-volume frame and ghost cells — for `divergence.py`, `boundary.py`

1. **LeVeque, *Finite Volume Methods for Hyperbolic Problems* (2002), Ch. 4 (§4.1–4.4, 4.10–4.11)**
   — the REA (reconstruct–evolve–average) view, the conservative update
   `Q_i^{n+1} = Q_i^n − (Δt/Δx)(F_{i+½} − F_{i−½})`, why conservation form matters (Lax–Wendroff
   theorem, §12.10), the CFL condition (§4.4). *Extract*: the face/cell indexing that §1 of
   `fv-conventions.md` fixes, and the statement of discrete conservation that V1/V2 test.
   **[cited]**
2. **LeVeque Ch. 7 (Boundary Conditions and Ghost Cells), all of it** — periodic (§7.1),
   outflow/extrapolation and why zero-order extrapolation is first order but usually fine (§7.3.1),
   solid walls by reflection with the normal-velocity sign flip (§7.3.3), inflow via prescribed
   ghost data (§7.2); the discussion of how many ghost cells a stencil needs. *Extract*: the exact
   fill rules for `Periodic`, `Extrapolate(0|1)`, `Reflective`, `Dirichlet`/`Inflow`, and the
   `a + 1`/`b` padding count argument. This chapter *is* `boundary.py`. **[cited]**
3. **Clawpack's boundary-condition page** (<https://www.clawpack.org/bc.html>) and PyClaw's
   `pyclaw.BC` enum + `solver.num_ghost` — how a mature code exposes the same four kinds plus a
   user callback, and that `num_ghost` is set by the *solver* (2 for Classic, 3 for SharpClaw).
   *Extract*: confirmation that "BC = function filling `num_ghost` cells per (axis, side)" is the
   standard shape, and the `aux` hook we do not need yet. **[cited]**
4. **Hesthaven, *Numerical Methods for Conservation Laws: From Analysis to Algorithms* (SIAM
   2018), Ch. 5–6** — the same material written for implementers, with MATLAB that maps almost
   line-for-line onto `jnp` (the `extend`/ghost-cell helper in Ch. 5 is our `pad_ghost_cells`).
   *Extract*: his `extend(u, h, m, BCl, ul, BCr, ur)` signature — compare with ours below. Code:
   <https://www.mathworks.com/academia/books/numerical-methods-for-conservation-laws-hesthaven.html>.
   **[cited]**

### I.2 Why FD-WENO and what a stencil window is — for `stencil.py`, `split.py`

5. **Shu, *Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for
   Hyperbolic Conservation Laws*, ICASE 97-65 (1998), §2.1–2.3 and §4.1–4.2**
   (<https://ntrs.nasa.gov/citations/19980007543>; mirror
   <https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf>) — reconstruction from cell
   averages vs from point values (§2.1–2.2: the *finite-volume* and *finite-difference* frameworks
   differ in what the stencil holds), the one-dimensional conservative FD scheme where the
   numerical flux is reconstructed from the *flux values* `f(u_j)` (§4.2), Lax–Friedrichs flux
   splitting `f± = ½(f ± αu)` (§4.2, eq. 4.19–4.21) and why each half is reconstructed with its own
   bias. *Extract*: (i) the stencil `(a, b)` for the left-biased `f+` and right-biased `f−`
   reconstructions at face `i+½` (`fv-conventions.md` §2 quotes (2, 3)/(3, 2) for WENO5);
   (ii) the identity `f+ + f− = f` and the monotonicity `∂f±/∂u ≷ 0` that `split.py` tests.
   **[cited]**
6. **Shu, "High order weighted essentially nonoscillatory schemes for convection dominated
   problems", SIAM Review 51(1), 2009, §2–3** — the multi-D argument: FD-WENO applies
   dimension by dimension on uniform grids at full order; FV-WENO needs face quadrature beyond
   second order. *Extract*: the reason ADR-0002 chose FD, and the uniform-grid caveat.
   **[cited]**
7. **Jiang & Shu, JCP 126 (1996)** — only §2 for now (the WENO5 weights); the rest when 2e(R)
   starts. Skim so that `stencil_windows` returns what the weights need (a window of 5 values per
   face, per split flux). **[cited]**

### I.3 Time integration — for `integrate.py`

8. **Gottlieb, Shu & Tadmor, "Strong stability-preserving high-order time discretization
   methods", SIAM Review 43(1), 2001, §1–3** (<https://www.cfm.brown.edu/people/sg/SSPsiamreview.pdf>)
   — the SSP idea (a convex combination of forward-Euler steps inherits any convex-stable property
   of Euler, e.g. TVD), the optimal SSP(2,2) and SSP(3,3) Shu–Osher forms, the SSP coefficient and
   the resulting CFL restriction `Δt ≤ c·Δt_FE`, and why classical RK4 is *not* SSP. *Extract*:
   the exact Shu–Osher coefficients for `SSPRK2`/`SSPRK3`, and the TVD test idea (V-row below).
   **[cited]** The original: Shu & Osher, JCP 77 (1988), §2. **[cited]**
9. **Ketcheson, SIAM J. Sci. Comput. 30(4), 2008** ("Highly efficient SSP RK methods with
   low-storage implementations") — SSP(10,4), PyClaw/SharpClaw's default `SSP104`. Optional now;
   relevant when matching SharpClaw's time accuracy in V19. **[cited]**
10. **LeVeque Ch. 17 (§17.1–17.3)** — fractional-step/Strang splitting for source terms; why the
    `RHS(hyperbolic, parabolic, source)` split exists even though 2c only sums the parts.
    **[cited]** Pareschi & Russo (2005) for IMEX is parked until a stiff source appears.

### I.4 Analytic references — for `analytic.py`

11. **LeVeque Ch. 11 (§11.1–11.3) and Hesthaven Ch. 2** — the method of characteristics for
    scalar laws; Burgers' pre-shock solution `u(x,t) = u₀(x − u t)` solved implicitly (Newton), the
    breaking time `t_b = −1/min u₀'`. *Extract*: the two or three closed forms `analytic.py` ships:
    advected sine, pre-shock Burgers from a sine, a translating step. **[cited]**
12. **Ketcheson, LeVeque & del Razo, *Riemann Problems and Jupyter Solutions* (SIAM 2020)**,
    free online (<https://nbviewer.org/github/clawpack/riemann_book/blob/master/Preface.ipynb>)
    — the Burgers and traffic chapters give exact Riemann solutions with runnable code; the
    nonconvex-flux chapter is the reference for cubic/sine (V15). Read for pleasure now, use in 2e.
    **[cited]**
13. **Roache, J. Fluids Eng. 124 (2002)** — the method of manufactured solutions, for V13; one
    reading, no code yet. **[cited]**

### I.5 Tooling — JAX and property tests

14. `jax.numpy.pad` (per-axis `pad_width`, `mode="wrap" | "edge" | "reflect" | "symmetric"`;
    note `"reflect"` excludes the edge sample, `"symmetric"` includes it — `Reflective` for cell
    averages wants **`symmetric`** [deduced from LeVeque §7.3.3: ghost cell `i=−1` mirrors cell
    `i=0`]); `jax.lax.slice_in_dim` / `jax.lax.dynamic_slice_in_dim`; `jnp.moveaxis` for
    axis-generic code; `jax.vmap` with `in_axes` for the transverse batch axes.
    <https://docs.jax.dev/en/latest/jax.numpy.html>. **[cited]**
15. **Hypothesis** (<https://hypothesis.readthedocs.io>) — `@given`, `strategies.integers/floats`,
    `hypothesis.extra.numpy.arrays`, `settings(deadline=None)`, `assume`; and **chex**
    (`chex.assert_trees_all_close`, `chex.variants` for jit/vmap parity — V21). The repo profiles
    are in `pyproject.toml` (`dev` 10 examples, default 50, `ci` 200). **[cited]**

---

## Part II — Module specifications

All functions take `axis: int` (ADR-0002), are pure, operate on `jnp` arrays whose spatial axes
are the trailing `n_spatial_dims` axes after a leading `n_eqn` (channel) axis:
`u: Float[Array, "n_eqn *N"]`. Batch dimensions, if any, are added by `vmap` outside `fv/`
(V21). Everything is written to run under `jit`; shapes are static.

### II.1 `fv/stencil.py`

```python
@dataclass(frozen=True)
class Stencil:
    """Face i+½ sees cells i−a … i+b (fv-conventions §2)."""
    a: int
    b: int
    @property
    def width(self) -> int: return self.a + self.b + 1
    def __or__(self, other) -> "Stencil": return Stencil(max(self.a, other.a), max(self.b, other.b))
    @property
    def pad(self) -> tuple[int, int]: return (self.a + 1, self.b)        # cells to pad per side

def stencil_windows(u_padded: Array, stencil: Stencil, axis: int) -> Array:
    """[..., N + a + 1 + b, ...] along `axis`  →  [..., N + 1, ..., width]: window k holds the
    cells face k−½ sees, oldest-left. Implemented as a stack of `width` shifted slices
    (`lax.slice_in_dim`), which XLA fuses; no gather."""
```

Invariants (property tests, `tests/property/test_stencil.py`):
- `stencil_windows(pad(u), s, axis).shape[axis] == N + 1` and the last axis has `s.width` entries.
- For `s = Stencil(0, 0)` and periodic padding, window `k` is exactly `u[k−1]` (one cell per face).
- Window `k`, entry `j`, equals `u_padded[k + j]` along `axis` (the definition) — test on random
  shapes with `N ∈ [1, 8]`, `n_spatial_dims ∈ {1, 2, 3}`, `axis` any.
- `jit`/`vmap` parity (V21 style).

Design notes: the window axis is appended *last* so a `FluxPredictor` (2d) can treat it as the
feature axis of an MLP or the input of WENO weights without transposes **[deduced]**. A
`(2, 3)` ∪ `(3, 2)` union gives `Stencil(3, 3)`, `pad = (4, 3)`; WENO5 then slices its two
biased sub-windows out of the 7-wide union window rather than padding twice **[deduced]**.

### II.2 `fv/boundary.py`

```python
class BoundaryCondition(eqx.Module):          # ADR-0003
    def fill_ghost(self, u: Array, axis: int, side: Literal["lower", "upper"], n: int,
                   *, pde=None, x: Array | None = None, t: float | None = None) -> Array:
        """Return the n ghost cells for `side` of `axis`, shaped like u with `n` along axis."""

class Periodic(BoundaryCondition): ...         # opposite end
class Extrapolate(BoundaryCondition):          # order ∈ {0, 1}
class Reflective(BoundaryCondition):           # mirror then pde.reflect(u, axis)
class Dirichlet(BoundaryCondition):            # value: Array | Callable[[x, t], Array]
class Inflow(Dirichlet):                       # primitive state → pde.to_conserved

BCs = tuple[tuple[BoundaryCondition, BoundaryCondition], ...]   # per axis (lower, upper)

def pad_ghost_cells(u: Array, axis: int, stencil: Stencil, bcs: BCs, *, pde=None,
                    x=None, t=None, transverse: int = 0) -> Array:
    """Pad (a+1, b) along `axis` using bcs[axis], and `transverse` cells on every other spatial
    axis using that axis's bcs (0 unless the flux is genuinely multidimensional)."""
```

Fill rules (LeVeque Ch. 7 **[cited]**; `x`, `t` only used by `Dirichlet`/`Inflow`):
- `Periodic`: lower ghosts = last `n` cells, upper ghosts = first `n` cells (`jnp.pad(mode="wrap")`).
- `Extrapolate(0)`: repeat the boundary cell (`mode="edge"`); `Extrapolate(1)`: `u_0 + k (u_0 − u_1)`.
- `Reflective`: `mode="symmetric"` then `pde.reflect(·, axis)` on the ghost block (identity for
  scalars; sign flip of the normal momentum/velocity for systems — `reflect` is Phase 4.1's
  `PDEFamily` method; until then pass a callable or `None` = identity).
- `Dirichlet(value)`: broadcast the constant, or evaluate `value(x_ghost, t)` at ghost centres
  `x_ghost = x_0 − (k + ½) dx` **[deduced]**.
- Periodic on one side requires periodic on the other: validate in `pad_ghost_cells`.

Invariants (`tests/property/test_boundary.py`, V5):
- Interior unchanged: `padded[..., a+1 : a+1+N, ...] == u` exactly, for all BCs.
- `Periodic`: `padded == jnp.pad(u, mode="wrap")` exactly; and equals the `jnp.roll` construction
  the old `InterfaceReconstructor` used **[code]**.
- `Reflective ∘ Reflective == id` on the ghost block; with a scalar `reflect`, the ghost block equals
  `mode="symmetric"`.
- `Extrapolate(0)` leaves a constant field constant; `Extrapolate(1)` leaves a linear field linear
  (exact).
- `Dirichlet(c)`: ghost block `== c`.
- Shape: `N + a + 1 + b` along `axis`, unchanged elsewhere unless `transverse > 0`.
- Shapes hold for `N = 1` (the smallest grid; V23).

### II.3 `fv/divergence.py`

```python
def flux_divergence(F: Array, dx: float, axis: int) -> Array:
    """F has N+1 faces along axis → (F_{i+½} − F_{i−½}) / dx, N cells. = jnp.diff(F, axis)/dx."""

def total_divergence(Fs: Sequence[Array], dxs: Sequence[float]) -> Array:
    """Σ_axis flux_divergence(F_axis, dx_axis, axis) — the unsplit, method-of-lines RHS."""
```

Invariants (`tests/property/test_fv_core.py`, V1/V2):
- Conservation: `Σ_i flux_divergence(F)_i · dx == F[N] − F[0]` (telescoping; exact up to
  round-off: `atol 1e-12` x64 / `1e-6` fp32). With periodic BCs `F[0] == F[N]` so the sum is 0 (V1);
  otherwise it is `F_in − F_out` (V2).
- Linear advection, upwind flux `F_{i+½} = a u_i` (a > 0), CFL = 1, forward Euler: the update is the
  exact shift by one cell **[cited: LeVeque §4.9]** — a two-line integration test that exercises
  `pad → windows → flux → divergence → step` end to end before any WENO exists.

### II.4 `fv/split.py`

```python
def lax_friedrichs_split(f: Array, u: Array, alpha: Array | float) -> tuple[Array, Array]:
    """f± = ½ (f ± α u), α ≥ max |f'(u)| over the relevant set (Shu 1998 §4.2)."""
def global_alpha(pde, u, params, axis) -> Array: ...   # max over the whole field
def local_alpha(pde, windows, params, axis) -> Array: ...   # per-face max over the window (LLF)
```

Invariants: `f+ + f− == f` (exact); for a scalar law with `α ≥ max|f'|`, `f+` is nondecreasing
and `f−` nonincreasing in `u` on a sampled grid of `u` values (sign of finite differences; this is
the monotonicity WENO's upwinding relies on **[cited: Shu 1998 §4.2]**).

### II.5 `fv/integrate.py`

```python
class RHS(NamedTuple):
    hyperbolic: Callable[[Array, float], Array]          # (u, t) → −Σ div F
    parabolic:  Callable[[Array, float], Array] | None = None
    source:     Callable[[Array, float], Array] | None = None
    def __call__(self, u, t): return sum of the non-None parts

class Integrator(eqx.Module):
    def step(self, rhs: RHS, u: Array, t: float, dt: float) -> Array: ...

class ForwardEuler(Integrator): ...
class SSPRK2(Integrator):   # Shu–Osher: u1 = u + dt L(u);  u⁺ = ½u + ½(u1 + dt L(u1))
class SSPRK3(Integrator):   # u1 = u + dt L(u); u2 = ¾u + ¼(u1 + dt L(u1)); u⁺ = ⅓u + ⅔(u2 + dt L(u2))
class CFLAdaptive(Integrator):  # stub: raises NotImplementedError; signature fixed:
    inner: Integrator; cfl: float; max_substeps: int
```

Coefficients from Gottlieb–Shu–Tadmor 2001, eqs. (2.8)–(2.9) / Shu–Osher 1988 **[cited]**. Stage
times `t + c_k dt` are passed through for time-dependent sources/BCs (`c = (0, 1)` for SSPRK2,
`(0, 1, ½)` for SSPRK3 **[cited]**).

Invariants (`tests/property/test_integrate.py`, V8/V11 precursor):
- `dt = 0` is the identity for every integrator.
- On `u' = λu` one step equals the truncated exponential: `1 + z`, `1 + z + z²/2`,
  `1 + z + z²/2 + z³/6` with `z = λ dt` (`rtol 1e-12`, x64) — this is V8 and pins the
  coefficients.
- Temporal EOC on `u' = λu` over 4 halvings: 1, 2, 3 (`± 0.1`) — the cheap version of V11.
- SSP/TVD: linear advection with upwind flux, step IC, `CFL ≤ 1`: total variation
  `Σ|u_{i+1} − u_i|` is non-increasing for every integrator (Euler is TVD here; SSP inherits it
  **[cited: GST 2001 §2]**). Also check it *fails* for a non-SSP RK3 (e.g. classical Heun-3) at
  CFL just above its SSP radius — optional but the most instructive test in the phase.
- The integrator never sees `dx`, `α` or any flux object: discrete conservation is the
  divergence's job (`fv-conventions.md` §5).

### II.6 `fv/analytic.py`

```python
class AnalyticSolution(Protocol):
    def __call__(self, x: Array, t: float, params: Mapping[str, float]) -> Array: ...
    def source(self, x, t, params) -> Array | None   # None unless MMS

advected_sine(a, k)             # u = sin(k(x − a t))
burgers_presmooth(u0, t_break)  # implicit u = u0(x − u t) by Newton, valid for t < t_b
translating_step(a, x0, uL, uR) # for TVD / Riemann smoke tests
manufactured(f, u_exact)        # source = ∂_t u_exact + ∂_x f(u_exact) via jax.grad — MMS (V13)
```

The Newton solve for Burgers is a fixed small number of iterations under `lax.fori_loop`
(converges fast before breaking) **[cited: Hesthaven Ch. 2 code]**.

---

## Part III — Build order and checkpoints

1. `stencil.py` + tests (half a day). You will find the `(a+1, b)` padding count by writing the
   `N = 1`, `Stencil(0, 0)` case by hand first.
2. `divergence.py` + the conservation tests (an hour). Then the **upwind-advection end-to-end**
   test with a hand-written `F = a·u_left` flux: this is the first time the pipeline runs.
3. `boundary.py`: `Periodic` and `Extrapolate` first (both are `jnp.pad` modes), then
   `Reflective` with an identity `reflect`, then `Dirichlet`. Re-run the advection test with
   `Dirichlet` inflow / `Extrapolate` outflow — V17's smoke version.
4. `integrate.py`: Euler → SSPRK2 → SSPRK3 with the amplification-factor test after each; the
   TVD test last.
5. `split.py` (an hour) — it is only needed by 2e, but writing it now, with its monotonicity test,
   is the moment the FD-WENO picture clicks.
6. `analytic.py`: advected sine first (needed by the temporal EOC test), Burgers when 2e starts.
7. Flip `verification.md` rows V1, V2, V5, V8 to green with the commit; add V23's `N = 1, 2, 3`
   parametrisation to every property test from the start rather than retrofitting.
8. Change document `docs/changes/<date>-phase2b-fv-core.md` from day one (`status: planned` →
   `in-progress`), Review-notes rows for Claude to fill *Thoughts* on this time (role reversal:
   Claude reviews Joon's numerics against this plan and the conventions).

What is deliberately *not* in 2b/2c: any flux (2d/2e), `PDEFamily` (4.1 — pass `pde=None` or a
tiny stub exposing `reflect`/`flux`/`max_wave_speed` for tests), `CFLAdaptive` beyond the stub,
nonuniform grids, and anything about models.

## Open questions for Joon to settle while reading

- `Stencil` as a frozen dataclass vs a plain `(a, b)` tuple: the dataclass carries `pad`/`width`
  and the union; the tuple is what `fv-conventions.md` writes. Claude leans dataclass **[deduced]**.
- Whether `pad_ghost_cells` pads *all* spatial axes at once (one call per step, returning one
  padded array that every directional flux slices) or one axis per call as specified above. Per
  axis is simpler and matches ADR-0002's "transverse axes are batch axes"; all-at-once saves
  re-padding when `transverse > 0`. Start per axis.
- `x`/`t` plumbing for `Dirichlet`: pass through `pad_ghost_cells` (as above) or bind at
  construction (`Dirichlet.at(x, t)`)? The former keeps BCs stateless under `jit`.

## Sources

[cited] LeVeque, *Finite Volume Methods for Hyperbolic Problems*, CUP 2002 (Ch. 4, 7, 11, 12, 17;
book page <https://faculty.washington.edu/rjl/book.html>) · Shu, ICASE 97-65 (1998),
<https://ntrs.nasa.gov/citations/19980007543> · Shu, SIAM Review 51(1), 82–126 (2009) · Jiang & Shu,
JCP 126, 202–228 (1996) · Gottlieb, Shu & Tadmor, SIAM Review 43(1), 89–112 (2001),
<https://www.cfm.brown.edu/people/sg/SSPsiamreview.pdf> · Shu & Osher, JCP 77, 439–471 (1988) ·
Ketcheson, SIAM J. Sci. Comput. 30(4), 2113–2136 (2008) · Hesthaven, *Numerical Methods for
Conservation Laws: From Analysis to Algorithms*, SIAM 2018 (code via MathWorks book page) ·
Ketcheson, LeVeque & del Razo, *Riemann Problems and Jupyter Solutions*, SIAM 2020
(<https://nbviewer.org/github/clawpack/riemann_book/blob/master/Preface.ipynb>) · Roache, J. Fluids
Eng. 124, 4–10 (2002) · Clawpack BC docs <https://www.clawpack.org/bc.html> · JAX docs
<https://docs.jax.dev> · Hypothesis docs <https://hypothesis.readthedocs.io>.
[code] the `InterfaceReconstructor` roll-based padding at `iclr2027-submission`.
[deduced] window-axis-last layout; union-stencil slicing; `symmetric` vs `reflect` for cell
averages; ghost-centre formula; the dataclass-vs-tuple and per-axis-padding leanings.
