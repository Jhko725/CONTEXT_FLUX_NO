---
type: change
status: landed
updated: 2026-10-26
branch: phase1-prune
roadmap: Phase 1
---

# 2026-10-04 — `phase1-prune`

Phase 1 of `docs/roadmap.md`: delete dead symbols and the legacy xarray slice in dependency order,
behind the regression tier built in Phase 0. Executed by Claude on 2026-10-24 from the ledger Joon
confirmed; the **Steps** section below is the original plan of record and is kept for reference.
`src/` goes from 10,765 to 9,297 lines.

**Merge gate.** This branch is not merged until `tests/regression` is green on the ICLR goldens
(Phase 0.8, deferred while the cluster is down). ~~Nothing here changes the forward pass of loaded
weights~~ — **the gate found one** (2026-10-26, §Findings F15): DISCO's forward pass depends on its
construction-time initialization through the un-checkpointed `operatornet_scale`, so the F6 key fix
in `operatornet.py` moved the two DISCO goldens by ≤ 2e-5. That fix is parked (Files, commit 6); the
five other cases were bit-green.

## Summary

Removes 14 dead symbols (assessment §5 plus `DPOTEncoder` and `LSKFluxNOTargetNetwork` per D1),
the legacy xarray training path (loaders, one script, three data configs, five notebooks; keeping
`SegmentLoaderBackground` under a DEPRECATED header until Phase 4), switches the legacy PDE classes
to the `(u, t, xs)` return and deletes `solution_to_dataset`, fixes four PRNG-key-reuse bugs (F5,
F6) and guards against their recurrence with a key-hygiene test, fixes a `jax.flatten_util` import
that broke model construction on JAX 0.9, repairs CI (ruff was not a dependency) and the Phase 0
hypernetwork test.

## Files

Commit 0 — CI and test repair:
- `pyproject.toml`, `uv.lock` — `ruff==0.15.11` in the dev group (CI's `uv run ruff` had nothing to run).
- `.pre-commit-config.yaml` — ruff rev aligned to `v0.15.11`.
- `scripts/import_all.py` — new: imports every package module; missing `clawpack` reported as skipped unless `--strict`.
- `tests/test_nn_hypernetwork.py` — fixed keyword names (`in_size`, `target_network`, …); now also checks both valid initialisations build.
- `src/context_flux_no/nn/hypernetwork.py`, `src/context_flux_no/models/multiphysics/disco/disco.py` — `from jax import flatten_util` and `flatten_util.ravel_pytree(...)` (see Bugs fixed; review round 1 replaced the initial `import jax.flatten_util  # noqa` form).

Commit 1 — dead symbols (ledger 1–11):
- `src/context_flux_no/models/fluxfno.py` — deleted (`FluxFNO1D`).
- `src/context_flux_no/models/multiphysics/wrapper.py` — deleted (`MultiphysicsWrapper`); `models/multiphysics/__init__.py` re-export removed.
- `src/context_flux_no/nn/vit.py` — deleted (`VisionTransformer`).
- `src/context_flux_no/models/multiphysics/disco/disco.py` — `LinearVariableInFeatures` removed.
- `src/context_flux_no/nn/operators/fourier.py` — `SpectralConv1D/2D/3D`, `Fourier1D/2D/3D` removed; `nn/operators/__init__.py` exports `Fourier` only.
- `src/context_flux_no/nn/position_encoding/relative.py` — `rel_pos_to_bucket_bin` removed.
- `src/context_flux_no/training/batching.py` — deleted; `training/__init__.py` updated.
- `src/context_flux_no/data/transforms.py` — deleted (`SpatialDownsample`); `data/__init__.py` updated.
- `src/context_flux_no/simulations/pde/euler.py` — `Euler1D`, `riemann_euler_hll_1D`, `_pressure_flux_1d` removed.
- `src/context_flux_no/simulations/utils.py` — `sample_coefficients_uniform`, `BASE_KEY_DICT` removed.
- `src/context_flux_no/waveforms/grf.py` — `ExponentialCov` removed.

Commit 2 — D1 (ledger 12–13):
- `src/context_flux_no/models/multiphysics/hyperfluxfno/encoders/dpot_encoder.py` — deleted; `encoders/__init__.py` updated.
- `src/context_flux_no/models/multiphysics/hyperfluxfno/target_networks/lskflux.py` — deleted; `target_networks/__init__.py` updated.
- `src/context_flux_no/models/multiphysics/hyperfluxfno/utils.py` — `make_encoder` loses the `DPOT` branch and `Literal`; `TARGET_NETWORK_DICT` loses `LSKFluxNO`.

Commit 3 — legacy training path (ledger 14–17):
- `src/context_flux_no/training/trainer.py` — `loss_fn: AbstractTrainLoss` (kept on purpose).
- `src/context_flux_no/training/loader.py` — `SegmentLoaderNaive`, `SegmentLoader`, `ContextSegmentLoader` removed; DEPRECATED module docstring naming Phase 4; `SegmentLoaderBackground` + `make_segment_axis` kept.
- `src/context_flux_no/training/dataset.py` — deleted (`PDEDataset`); `training/__init__.py` exports `SegmentLoaderBackground` only.
- `scripts/training/train_shallow_water.py`; `scripts/training/configs/data/{burgers_1d,shallow_water_1d,2d_scalar}.yaml` — deleted.
- `notebooks/training/{fluxfno_burgers,shallow_water_1d,multiphysics_cubic_2d,viscous_burgers_1d}.ipynb`, `notebooks/misc/prototype_dataloading.ipynb`, `notebooks/misc/debug_dataloader.py` — deleted (recoverable from the `pre-nbstripout` tag).

Commit 4 — PDE returns (ledger 18):
- `src/context_flux_no/simulations/pde/{burgers,cubic,sine_1d,shallow_water}.py` — `solve` returns `(u, t, xs)`; `solution_to_dataset` import and `xarray` annotation removed.
- `src/context_flux_no/simulations/pdesolve.py` — `solution_to_dataset` and `import xarray` removed.

Commit 5 — correctness (ledger 19–21, 25):
- `src/context_flux_no/models/multiphysics/disco/disco.py` — `split(key, 4)`.
- `src/context_flux_no/models/multiphysics/hyperfluxfno/hyperfluxfno.py` — `HyperNeuralOperator`: `split(key, 6)`, head on `keys[5]`.
- `src/context_flux_no/models/multiphysics/disco/operatornet.py`, `src/context_flux_no/models/multiphysics/hyperfluxfno/target_networks/unet.py` — second bottleneck conv on `keys_bot[1]`.
- `tests/test_models_disco.py` — missing `assert` at the 3D check.
- `tests/unit/__init__.py`, `tests/unit/test_key_hygiene.py` — new.

Commit 6 — bookkeeping: this document, `docs/roadmap.md`, `docs/architecture.md`, `docs/findings.md`, `docs/verification.md`, `docs/index.md`; `scripts/check_docs.py` accepts path suffixes in Files entries.

Commit 6 — goldens gate, 2026-10-26 (after merging `main` with the goldens):
- `src/context_flux_no/models/multiphysics/disco/operatornet.py` — the second bottleneck conv back on `keys_bot[0]` (F15; comment at the site). `target_networks/unet.py` keeps `keys_bot[1]`.
- `tests/unit/test_key_hygiene.py` — DISCO case `xfail(strict=True, reason=F15)`, so the fix's return flips the test.
- `docs/findings.md` — F15; F6 action amended. `docs/verification.md` — V22 green, V24 note. `docs/roadmap.md` — 1.3 row, deferred item, Done row.

## Design

- **Keep `SegmentLoaderBackground`** (contrary to the assessment's dead list): seven evaluation
  notebooks still read the ICLR-era 1D datasets through xarray; it carries a DEPRECATED header naming
  Phase 4 as its removal. Rejected: a `legacy/` folder (Principle 3).
- **Keep `AbstractTrainLoss`** and type `Trainer.loss_fn` with it: flux-space losses (Phase 6) need
  the interface (plan §10.8).
- **Base-class / `coeffs` naming of the legacy PDE classes is not touched** — Phase 4's
  `PDEFamily`; doing it now means doing it twice. Only the return type is unified so that
  `solution_to_dataset` can go.
- **Key-hygiene test as the guard for F5/F6** rather than seed-parity goldens: bit-parity of fresh
  trainings with the ICLR runs stopped being a property of this repo at D15, and GPU training is not
  bitwise reproducible anyway. The test records every key passed to `jax.random.{split, fold_in,
  samplers}` during `__init__` and fails on any repeat; traced keys (under `vmap`) are skipped because
  they cannot be compared. It was verified to fail with the `keys_bot[0]` bug reinstated. Models are
  built from their Hydra configs on 16-cell grids so the real constructors are exercised; this also
  makes it the first test that covers the method's model classes at all.
- **Park the `operatornet` half of F6 rather than regenerate the DISCO goldens** (2026-10-26). The tier's claim is "the refactor reproduces the submitted behavior"; regenerating goldens on the branch would have made them pin a model that is not the ICLR DISCO, for a two-line fix. Rejected: regenerating with a change-document entry (allowed by policy, but it retires the DISCO baselines from V22); loosening the DISCO tolerance (0.4% relative is not a tolerance); restoring the training-time scales by re-running the buggy init behind a legacy flag (a second route to one result). The fix returns with checkpointed `operatornet_scale` (Phase 3).
- **Fix F6 now** (the two UNet copies) although the plan deferred it to Phase 2g: the hygiene test
  found it, the fix is one index, and leaving a known-failing test behind is worse than a one-line
  deviation from the ledger.
- **ruff pinned in the dev group**: `uv run ruff` in CI needs ruff to be a project dependency;
  `uvx ruff` would have been the alternative but pins nothing. `ruff format` on the ledger's files
  used 0.15.11, so the pre-commit rev follows.

## Bugs fixed

- **`jax.flatten_util` never imported.** `HypernetworkHead` and DISCO's `HyperNetworkHead` access
  `jax.flatten_util.ravel_pytree`, but on JAX 0.9 `flatten_util` is not an attribute of `jax` until
  the submodule is imported. Model construction raised `AttributeError` in isolation; in training
  scripts optax/equinox had imported it first, which is why it never surfaced. Found by the corrected
  hypernetwork test.
- **PRNG key reuse** (F5, F6): DISCO `keys[3]` out of range (clamped to `keys[2]`); `HyperNeuralOperator`
  `keys[3]` shared by `project_operator` and `hypernetwork_head`; both UNet copies `keys_bot[0]` for both
  bottleneck convolutions.
- `tests/test_models_disco.py:286` had no `assert`; the 3D `OperatorNetwork` shape check was a no-op.
- `tests/test_nn_hypernetwork.py` (Phase 0) raised `TypeError` from wrong keyword names and never
  exercised the guard.
- Ledger 22 (`train_the_well.py` valid sampler length) was already correct; no change.

## Findings

- F5 and F6 are fixed (commit 5); appended to `docs/findings.md` with the commit. F6's `operatornet` half is parked again in commit 6 (below).
- **F15** (2026-10-26): DISCO's forward pass depends on `operatornet_scale`, floats computed from the fresh init and never checkpointed; the ICLR DISCO checkpoints are reproducible only with the original init code and key. Caught by the regression tier: `disco_cubic1d` 47/100 points off by ≤ 1.9e-5 (0.4% relative), `disco_euler2d` 622/65536 by ≤ 4.9e-6; the five other cases bit-identical. Bears on every DISCO baseline number's reproducibility and on any later change to `OperatorNetwork`'s init.
- New: the `jax.flatten_util` import (F9) — a latent breakage, not a result-affecting one, recorded
  because any standalone model construction (e.g. the regression tier's `load_model`) would have hit it.

## Tests

- Fast tier: `JAX_PLATFORMS=cpu uv run pytest` → 33 passed (was 21 + 1 failing at the start of the
  branch). Runs on CPU in ~3.5 min because the DISCO/DPOT shape tests build large models.
- `uv run python scripts/import_all.py` → 0 failed (5 skipped here for missing `clawpack`; CI has
  gfortran and runs `--strict` implicitly through `uv sync`).
- `tests/unit/test_key_hygiene.py` — 6 models × no-reuse, 3 models × seed determinism.
- Regression tier — skips until the goldens exist (merge gate).

## Open issues

- Burgers / shallow-water / 2D-cubic *training* is unavailable between Phase 1 and the Phase 4 regen.
- `SegmentLoaderBackground` and the `xarray`/`h5netcdf`/`netcdf4`/`dask` dependencies remain until Phase 4.
- F3 (`rfftn` axes), F4 (`stack_grid` ignored), DISCO's double `standardize`, and the four face
  conventions stay deferred to Phase 2g/3 as recorded in `findings.md`.
- DISCO's hygiene case is a strict `xfail` until `operatornet_scale` is checkpointed and the `keys_bot` fix returns (F15).
- The hygiene test builds DPOT with `vmap`-initialised blocks whose keys are traced and therefore
  not checked; a reuse inside those blocks would not be caught.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| **Commit 6** `disco/operatornet.py` `keys_bot[0]` restored (F15); `test_key_hygiene.py` DISCO strict xfail; F15, V22/V24, roadmap | | |
| Commit 0 — `ruff` dev pin, pre-commit rev, `scripts/import_all.py`, hypernetwork test fix, `jax.flatten_util` imports | Mostly looks good. One thing - meybe to an `from jax import flatten_util` and replace `jax.flatten_util` with `flatten_util`? This would remove the need for #noqas. | Deferred to discussions with Claude. |
| Commit 1 — dead symbols (ledger 1–11) | Confirmed. | None |
| Commit 2 — `DPOTEncoder`, `LSKFluxNOTargetNetwork`, `make_encoder`/`TARGET_NETWORK_DICT` | Confirmed. | None |
| Commit 3 — legacy loaders/script/configs/notebooks; `SegmentLoaderBackground` DEPRECATED; `Trainer.loss_fn: AbstractTrainLoss` | Confirmed. | None |
| Commit 4 — PDE `solve` returns `(u, t, xs)`; `solution_to_dataset` removed | Confirmed. | None |
| Commit 5 — key-reuse fixes (DISCO, HNO, both UNets); `tests/unit/test_key_hygiene.py`; missing assert | Looks good. | None |
| Commit 6 — docs bookkeeping | Looks good. | None |
| Review round 1 (`a`) — `from jax import flatten_util`, call sites `flatten_util.ravel_pytree`; no `noqa` | | |

---

## Steps (plan of record)

Principle: delete in dependency order (leaves first), run `ruff check` (catches now-unused imports via F401) and `pytest -m "not regression"` after each group, and the regression tier at the end of each sub-phase. One commit per group so a mistaken deletion is a one-commit revert.

A useful helper for the whole phase — `scripts/import_all.py`:
```python
"""Import every module in the package; fails loudly on any broken import."""
import importlib, pkgutil, sys
import context_flux_no as pkg
bad = []
for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
    try: importlib.import_module(m.name)
    except Exception as e: bad.append((m.name, repr(e)))
for name, err in bad: print("FAIL", name, err)
sys.exit(1 if bad else 0)
```

### 1.1 Dead symbols (½ day)

Work through the table top to bottom. "Refs to fix" are the only places that mention the symbol outside its own file [code, §5 and the import graph checked on 09-27].

| # | Delete | Refs to fix | Notes |
|---|---|---|---|
| 1 | `src/context_flux_no/models/fluxfno.py` (whole file) | none | `FluxFNO1D`; only ref is a commented-out import in a notebook being deleted in 1.2 |
| 2 | `models/multiphysics/wrapper.py` | `models/multiphysics/__init__.py:13` | `MultiphysicsWrapper` |
| 3 | `nn/vit.py` | none (`nn/__init__` doesn't export it) | `VisionTransformer` is unfinished (`__call__` raises) |
| 4 | `disco/disco.py:18–20` `class LinearVariableInFeatures` | none | empty class |
| 5 | `nn/operators/fourier.py`: classes `SpectralConv1D/2D/3D` (≈123–188) and `Fourier1D/2D/3D` (≈274–343) | `nn/operators/__init__.py:2` — change to `from .fourier import Fourier as Fourier` | keep `SpectralConv` and `Fourier`; leave the `rfftn` TODO at :97 alone — it is a Phase 2 finding, not a deletion |
| 6 | `nn/position_encoding/relative.py:119` `rel_pos_to_bucket_bin` | none | duplicate of the method at :66 |
| 7 | `training/batching.py` | `training/__init__.py:1` | |
| 8 | `data/transforms.py` | `data/__init__.py` (remove the line added in 0.6c) | `SpatialDownsample`; `TheWellDataSource.downsample_spatial` never used it — leave that flag alone for now, it goes with the source in Phase 4 |
| 9 | `simulations/pde/euler.py`: `_pressure_flux_1d` (:36), `riemann_euler_hll_1D` (:65), `class Euler1D` (:108–175) | none | keep `Euler2D` |
| 10 | `simulations/utils.py`: `sample_coefficients_uniform` (:46), `BASE_KEY_DICT` (:63) | none | `generate_dataset` inlines its own dict at :151 |
| 11 | `waveforms/grf.py:21` `ExponentialCov` | none | |
| 12 | `src/context_flux_no/__init__.py::main` | `pyproject [project.scripts]` (done in 0.10) | |
| 13 | `models/multiphysics/hyperfluxfno/encoders/dpot_encoder.py` | `encoders/__init__.py:2`; `hyperfluxfno/utils.py:7` (import), `:30` (`Literal`), `:60–70` (`case "DPOT"` branch) | D1 |
| 14 | `models/multiphysics/hyperfluxfno/target_networks/lskflux.py` | `target_networks/__init__.py:8`; `utils.py:13` (import), `:95` (`TARGET_NETWORK_DICT`), `:102` (`Literal`) | D1 |

**Keep**, contrary to the assessment's dead list: `training/loss.py::AbstractTrainLoss` — it is the interface the trainer should be typed against (plan §10.8); wire it in: `Trainer.__init__(..., loss_fn: AbstractTrainLoss[M, Batch], ...)`.

After each row: `uv run ruff check --fix . && uv run python scripts/import_all.py && uv run pytest -m "not regression" -q`. For rows 5, 9 and 13–14, also `grep -rn "<Name>" src scripts tests notebooks` must be empty (notebook hits for `Fourier1D` are printed reprs and can be ignored). Commit per row or per 3–4 related rows: `refactor: remove dead <symbols>`.

Row 13 detail, `utils.py::make_encoder` after the change:
```python
def make_encoder(
    encoder_type: Literal["ViT", "TRecViT"], ...
) -> AbstractEncoder:
    match encoder_type:
        case "ViT": ...
        case "TRecViT": ...
        case _:
            raise ValueError(f"Unknown encoder_type {encoder_type!r}")
```
and run `scripts/check_configs.py` again — no config selects `DPOT` or `LSKFluxNO` after 0.6d, so the census must be unchanged.

End of 1.1: `JAX_PLATFORMS=cpu uv run pytest tests/regression --checkpoints-root ./checkpoints` — must be green. If a golden's `load_model` fails, you deleted something a checkpoint's stored config still names; restore it (`git checkout iclr2027-submission -- <file>`) and add the case to the Phase 3 shim list instead.

### 1.2 Legacy xarray path (½–1 day)

**Read this before deleting.** Six notebooks (`evaluate_cubic`, `evaluate_burgers`, `evaluate_shallow`, `evalaute_cubic_long_term`, `long_term_rollouts`, three `ablate_*`) still evaluate the *paper's* 1D results by `xr.open_dataset` on legacy HDF5 test sets, and `evaluate_2d_Scalar.ipynb` + `multiphysics_cubic.ipynb` use `SegmentLoaderBackground`. Those datasets only get regenerated in Phase 4. So the honest Phase 1 cut is:

**Delete now:**
| Item | Refs |
|---|---|
| `training/loader.py`: `SegmentLoaderNaive` (:32–71), `SegmentLoader` (:135–216), `ContextSegmentLoader` (:217–end) | `training/__init__.py:3` — change to export `SegmentLoaderBackground` only |
| `training/dataset.py` (`PDEDataset`) | `training/__init__.py:2` |
| `scripts/training/train_shallow_water.py` | none |
| `scripts/training/configs/data/{burgers_1d,shallow_water_1d,2d_scalar}.yaml` | none — they only work with the deleted scripts |
| notebooks: `training/fluxfno_burgers`, `training/shallow_water_1d`, `training/multiphysics_cubic_2d`, `training/viscous_burgers_1d`, `misc/prototype_dataloading`, `misc/debug_dataloader.py` | none |
| `simulations/pdesolve.py::solution_to_dataset` (:63–end) and `import xarray` (:6) | the six legacy PDE classes below |

**Keep until Phase 4, with a deprecation header:** `training/loader.py::{make_segment_axis, SegmentLoaderBackground}` — add at the top of the file:
```python
"""DEPRECATED — legacy xarray/HDF5 segment loader kept only for evaluating the ICLR-era
1D datasets from notebooks. Removed in Phase 4 once those datasets are regenerated in
Well/zarr form. Do not use in new code."""
```
and the same one-line note in the evaluation notebooks' first cell. `xarray`, `h5netcdf`, `netcdf4`, `dask` stay in `pyproject` for the same reason; they are pruned in Phase 7.

**Legacy PDE classes — switch to the tuple return now** so `solution_to_dataset` can go. In each of `burgers.py` (`Burgers1D.solve` :49, `ViscousBurgers1D.solve` :199), `cubic.py` (`CubicFlux2D.solve` :212), `sine_1d.py` (:86), `euler.py` (`Euler1D` is already deleted), `shallow_water.py` (:200): replace `return solution_to_dataset(u, t, xs, self.coeffs)` with `return u, t, xs` where `xs` is a tuple of coordinate arrays (`(x_grid,)` in 1D, `(x, y)` in 2D — match what `CubicFlux1D`/`Euler2D` already return so `generate_dataset` can consume all of them later). Remove the `-> xr.DataArray` annotation and `import xarray` in `burgers.py`. Don't touch the inconsistent base classes/`coeffs` naming yet — that is Phase 4's `PDEFamily`; doing it now means doing it twice.

Then `grep -rn "xarray\|xr\." src` → only `training/loader.py` (deprecated) should remain.

Verify: `import_all.py`, `ruff check`, `pytest -m "not regression"`, the regression tier, and — because the point of keeping `SegmentLoaderBackground` is the notebooks — open `evaluate_cubic.ipynb` and run the cells up to the first metric on one checkpoint. Commit: `refactor: remove legacy loaders, scripts and notebooks; PDE solves return arrays`.

### 1.3 Small correctness items surfaced by the assessment that are safe to fix now (1–2 h)

Not deletions, but cheap, isolated, and each protected by a test. Do them as separate commits so the regression tier tells you exactly which one changes a golden (it shouldn't — none affects a trained forward pass — but that is the check).

1. `tests/test_models_disco.py:286` — add the missing `assert`.
2. `disco/disco.py:121` `keys = jax.random.split(key, 3)` with `keys[3]` used at :167 → `split(key, 4)`. **This changes DISCO initialisation**, not its forward pass with loaded weights, so goldens stay green; note it in the commit because future DISCO trainings will differ from the ICLR ones by RNG.
3. `hyperfluxfno.py` `HyperNeuralOperator.__init__`: `keys[3]` used for both `project_operator` (:395) and `hypernetwork_head` (:424) → split one more key. Same remark.
4. `ContextSegmentLoader` `start_indices[0]` bug — moot, deleted.
5. `train_multiphysics.py`: the valid loader seeds are fine (`.seed(1)`); the bug noted in the project memory (`IndexSampler(len(source_train))` for the valid loader) lives in `train_the_well.py` — fix it there: `len(source_valid)`.

Leave for Phase 2, and record in `docs/phase2_findings.md` so they are not lost: the `rfftn` axes TODO (`fourier.py:97`), `stack_grid` ignored in two model classes, the UNet `keys_bot[0]` reuse and unused boundary-mask channel, the two `standardize` calls in DISCO, the four face conventions.

### Exit checklist

- [ ] `uv run python scripts/import_all.py` clean; `uv run ruff check .` clean
- [ ] `grep -rn "FluxFNO1D\|MultiphysicsWrapper\|VisionTransformer\|LinearVariableInFeatures\|Fourier1D\|SpectralConv1D\|rel_pos_to_bucket_bin\|RandomMiniBatching\|SpatialDownsample\|Euler1D\|sample_coefficients_uniform\|ExponentialCov\|DPOTEncoder\|LSKFluxNO\|PDEDataset\|SegmentLoaderNaive\|ContextSegmentLoader\|solution_to_dataset" src scripts tests` → empty
- [ ] `pytest -m "not regression"` green; regression tier green (CPU)
- [ ] `scripts/check_configs.py` census unchanged except for the deleted data yamls
- [ ] `evaluate_cubic.ipynb` still runs its first metric cell
- [ ] `docs/phase2_findings.md` lists the deferred correctness items
- [ ] Line count check for your own satisfaction: `find src -name '*.py' | xargs wc -l | tail -1` (expect roughly 9.5k → 7.5k)
