---
type: change
status: planned
updated: 2026-10-04
branch: phase1-prune
roadmap: Phase 1
---

# 2026-10-04 — `phase1-prune` (planned)

Phase 1 of `docs/roadmap.md`: delete dead symbols and the legacy xarray slice in dependency
order, behind the regression tier built in Phase 0. Written before the work as the plan of record;
**Files** below is the planned list and will be rewritten to match `git diff --stat main...HEAD`
as the work lands.

## Summary

Remove 14 dead symbols (assessment §5 plus `DPOTEncoder`, `LSKFluxNOTargetNetwork` per D1), the
legacy xarray loaders/scripts/notebooks (keeping `SegmentLoaderBackground` under a DEPRECATED
header until Phase 4 regenerates the 1D datasets), switch the legacy PDE classes to the
`(u, t, xs)` return, and apply four isolated correctness fixes. Target: ~9.5k → ~7.5k LOC in `src`.

## Files *(planned)*

See the Steps tables below; every row names its file. The final list is produced by
`scripts/check_docs.py --files` at landing.

## Design

- **Keep `SegmentLoaderBackground`** (contrary to the assessment's dead list) because seven
  evaluation notebooks still read the ICLR-era 1D datasets through xarray; it carries a DEPRECATED
  header naming Phase 4 as its removal.
- **Keep `AbstractTrainLoss`** and wire it into `Trainer` type hints; flux-space losses (Phase 6)
  need the interface.
- **Base-class/`coeffs` naming of the legacy PDE classes is not touched** — that is Phase 4's
  `PDEFamily`; doing it now means doing it twice.

## Bugs fixed *(planned)*

`test_models_disco.py:286` missing assert; DISCO `split(key, 3)`/`keys[3]`; `HyperNeuralOperator`
key reuse; `train_the_well.py` valid sampler length (F5 and assessment §6).

## Findings

Deferred correctness items are recorded in `docs/findings.md` (F3, F4, F6) rather than fixed here.

## Tests

`scripts/import_all.py` (new) must be clean; fast tiers after every group; regression tier at the
end of 1.1 and 1.2.

## Open issues

- Burgers / shallow-water / 2D-cubic *training* is unavailable between Phase 1 and the Phase 4 regen.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| *(rows added as files land)* | | |

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
