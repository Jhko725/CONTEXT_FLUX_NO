---
type: change
status: in-progress
updated: 2026-10-04
branch: phase0-freeze
roadmap: Phase 0
---

# 2026-10-04 — `phase0-freeze`

Phase 0 of `docs/roadmap.md`: freeze the submission state, build the regression safety net, fix
the five things that are broken, and clean the repository's hygiene before any refactoring.
Mechanical items (0.1, 0.3, 0.4, 0.9, 0.10, 0.11) are done by Claude; the substantive ones
(0.2, 0.5–0.8) by Joon as a learning exercise, following the **Steps** section below, which is the
plan of record for this branch.

## Summary

Tag `iclr2027-submission`; introduce the `docs/` knowledge system and `CLAUDE.md`; untrack
generated files; strip notebook outputs with a git filter; make `icon/` a submodule; fix the
`bias_hyperinit` typo and make the paper's initialisation the default; remove dangling
`HyperFluxFNO` configs and the two scripts that cannot be imported; add a config census script;
build the regression harness from the ICLR checkpoints (CPU goldens); dev tooling, CPU CI,
README. Nothing in `src/` changes behaviour except the hypernetwork-init default (D15).

## Files

Claude's part (this commit series):

- `CLAUDE.md` — new: session protocol, change-document convention, repository facts.
- `docs/index.md`, `docs/roadmap.md`, `docs/architecture.md`, `docs/verification.md`,
  `docs/findings.md`, `docs/glossary.md`, `docs/assessment-2026-09-26.md` — new.
- `docs/decisions/0001…0009-*.md` — new ADRs (tiered coupling, N-D, BCs, WENO5 + references,
  three-part flux, hierarchy + scale, dataclass configs, four-tier testing, tracks).
- `docs/design/fv-conventions.md` — new: the numerics conventions for Phase 2.
- `docs/changes/2026-10-04-phase0-freeze.md` (this file), `docs/changes/2026-10-04-phase1-prune.md` (planned).
- `scripts/check_docs.py` — new: frontmatter, index coverage, Done-ledger order, Files-vs-diff check.
- `.gitignore` — fixed patterns; `pyclaw.log`, `*/pyclaw.log`, `.hypothesis/`, `*.h5`/`*.hdf5` (tests' `.npz` allowed).
- `icon/__pycache__/*.pyc`, `pyclaw.log`, `icon/test_traj_seq.h5` — untracked (0.3).
- `.gitattributes` — nbstripout filter (0.4); all `notebooks/**/*.ipynb` and `icon/TEST.IPYNB` stripped.
- `pyproject.toml` — dev group (`hypothesis`, `chex`, `pre-commit`, `ty`, `nbstripout`); `jax` CUDA moved to
  `[project.optional-dependencies] cuda`; ruff format settings; pytest markers; description; hello-world script removed (0.9, 0.10).
- `uv.lock` — regenerated.
- `.pre-commit-config.yaml` — ruff check/format, nbstripout.
- `.github/workflows/test.yml` — CPU job: ruff + `pytest -m "not regression and not slow"`.
- `README.md` — install, the three commands, test tiers, layout, link to `docs/`.
- `scripts/training/README.md` — `model=` values corrected.
- `src/context_flux_no/__init__.py` — `main()` stub removed.

Joon's part *(planned)*:

- `scripts/training/configs/model/*.yaml` — `bias_hyperinit → bias-hyperinit` (0.6a).
- `src/context_flux_no/nn/hypernetwork.py` — runtime check on the init string; `tests/test_nn_hypernetwork.py` (0.6a).
- `src/context_flux_no/models/multiphysics/hyperfluxfno/hyperfluxfno.py` — `hypernet_init` default → `"bias-hyperinit"` (D15).
- `scripts/training/configs/model/{hyperfluxfno_ViT,hyperfluxno_2d_scalar_global,hyperfluxno_2d_euler_global,hyperfluxno_2d_euler_lskflux}.yaml`, `scripts/training/train_multiphysics_2d.py` — deleted (0.6b, 0.6d).
- `src/context_flux_no/data/__init__.py` — exports (0.6c).
- `scripts/check_configs.py`, `docs/config_census_phase0.txt` (0.6e).
- `docs/bias_hyperinit_audit.csv`; `docs/findings.md` F1 appended with the conclusion (0.7).
- `tests/regression/{conftest.py,cases.yaml,make_goldens.py,test_regression.py,inputs/,goldens/}` (0.8).
- `third_party/icon` submodule; `icon/` removed (0.5).

## Design

- **Documentation system** — adopted from `Jhko725/deep_isochron` with these refinements:
  change documents double as the branch plan (`status: planned → in-progress → landed → merged`);
  a findings ledger separate from the roadmap for anything touching reported results; a
  verification matrix for numerics claims; `check_docs.py` to enforce the bookkeeping rules; a
  glossary; provenance labels kept in design documents. Rejected: keeping the plan in the claude.ai
  project (two sources of truth).
- **Per-phase branches** (`phase0-freeze`, `phase1-prune`, …) rather than one `cleanup` branch,
  so each change document maps to exactly one branch and one PR.
- **CUDA as an optional extra** so CI and the regression tier can use the CPU wheel; GPU machines
  run `uv sync --extra cuda`. Rejected: separate lock files.
- **Tags** — `iclr2027-submission` and `pre-nbstripout` are created locally by Claude but the
  session proxy refuses tag pushes (git and API); Joon pushes them:
  `git push origin iclr2027-submission pre-nbstripout`.

## Bugs fixed

None in this part. The five breakages of assessment §3 are Joon's items 0.6a–0.6d.

## Findings

F1 (bias-hyperinit) is the reason for D15; see `docs/findings.md`.

## Tests

No new tests in Claude's part. Joon's part adds `tests/test_nn_hypernetwork.py` and the regression
tier (`JAX_PLATFORMS=cpu uv run pytest tests/regression --checkpoints-root ./checkpoints`).
`pytest -m "not regression and not slow"` must stay green throughout.

## Open issues

- `icon/` upstream check (is it an unmodified copy of the ICON release?) decides whether the
  submodule points at upstream or at a new `Jhko725/icon-baseline` repository.
- The paper figures exist only as notebook outputs until Phase 6's `scripts/figures/`; the
  `pre-nbstripout` tag preserves them.

## Review notes

| Change | Thoughts | Modifications |
|---|---|---|
| `CLAUDE.md` — session protocol and conventions | | |
| `docs/roadmap.md` — tracks, phase tables, Done ledger | | |
| `docs/architecture.md` — as-found + planned | | |
| `docs/verification.md` — claim → test matrix (all planned) | | |
| `docs/findings.md` — F1–F8 seeded | | |
| `docs/glossary.md` | | |
| `docs/decisions/0001–0009` | | |
| `docs/design/fv-conventions.md` | | |
| `docs/index.md` | | |
| `scripts/check_docs.py` | | |
| `.gitignore`, untracked files | | |
| `.gitattributes` + stripped notebooks | | |
| `pyproject.toml`, `uv.lock` — dev deps, cuda extra, markers | | |
| `.pre-commit-config.yaml` | | |
| `.github/workflows/test.yml` | | |
| `README.md`, `scripts/training/README.md` | | |
| `src/context_flux_no/__init__.py` — stub removed | | |

---

## Steps (plan of record for Joon's part)

Line numbers against `5f67a31`. Run from the repo root; `uv run …` assumed.

### 0.0 Prerequisites (30 min)

- A machine with the ICLR checkpoints under `./checkpoints/` (layout written by `Trainer._make_checkpoint_path`: `checkpoints/<data.name>/<ModelClass>/<LossClass>/seed=<n>/<yy-mm-dd-HH_MM_SS>/`), and the datasets under `./data/datasets/`. GPU optional for Phase 0 — see 0.8 on why CPU is preferable for goldens.
- `uv sync` succeeds; `uv run python -c "import context_flux_no"` succeeds.
- Note the current baseline: `uv run pytest tests -q` → record pass/fail counts in your notes. (Expect all 5 files to pass; `test_models_disco.py:286` is a no-op assert — leave it for now, it's on the list.)

### 0.1 Tag and branch — *done by Claude except the tag pushes*

```bash
git tag -a iclr2027-submission 5f67a31 -m "State at ICLR 2027 submission"
git push origin iclr2027-submission
git switch -c cleanup
```
Why a tag and not a branch: the tag is immutable and is what §1.3 of the plan means by "the ICLR tag is the archive" — everything deleted later is recoverable with `git show iclr2027-submission:path/to/file`.

### 0.2 Confirm the breakages before touching anything (20 min)

You want to *see* each failure once so you know the fix worked.

```bash
uv run python -c "import scripts.training.train_multiphysics_2d"      # or: uv run python scripts/training/train_multiphysics_2d.py --help
#   → ImportError: cannot import name 'HyperFluxFNO'
uv run python -c "from context_flux_no.data import TheWellDataSource"
#   → ImportError
uv run python scripts/training/train_multiphysics.py model=hyperfluxfno_ViT --cfg job | head -3
#   → composes fine (Hydra doesn't instantiate with --cfg) — the failure is only at instantiate time; that is the point of 0.6e
grep -rn "hypernet_init" scripts/training/configs/model | sort | uniq -c
#   → 10 × bias_hyperinit, 2 × bias-hyperinit
```
Nothing to commit.

### 0.3 Untrack generated files and fix `.gitignore` — *done by Claude*

```bash
git rm -r --cached icon/__pycache__ pyclaw.log icon/test_traj_seq.h5
```
`--cached` removes from the index only; the files stay on disk until 0.5 moves `icon/` out. Edit `.gitignore`:

```diff
-*/pyclaw.log
+pyclaw.log
+**/pyclaw.log
+.hypothesis/
+*.h5
+*.hdf5
+!tests/**/*.npz
```
(`*.h5` is safe because datasets live under `/data/` which is already ignored; the `!tests/**/*.npz` line is for the goldens in 0.8.)

Verify: `git ls-files | grep -E "\.pyc$|pyclaw\.log|\.h5$"` prints nothing. Commit: `chore: untrack bytecode, logs and fixtures`.

### 0.4 Strip notebook outputs, with a safety tag first — *done by Claude; push the `pre-nbstripout` tag*

The paper's figures currently exist *only* as notebook outputs (§4f), so tag before stripping:

```bash
git tag -a pre-nbstripout -m "Last commit with notebook outputs"
uv add --dev nbstripout
uv run nbstripout --install --attributes .gitattributes     # adds a clean filter to .gitattributes + .git/config
uv run nbstripout notebooks/**/*.ipynb icon/TEST.IPYNB       # one-time strip of the working tree
git add -A && git commit -m "chore: strip notebook outputs; install nbstripout filter"
```
How the filter works (worth knowing): `.gitattributes` gets `*.ipynb filter=nbstripout`; git runs the filter on `git add`, so outputs never reach the index again, while your local files keep outputs until you re-run the notebook. Anyone cloning must run `nbstripout --install` once — put that in the README (0.10). Check: `du -sh notebooks` should drop from 16 MB to well under 1 MB.

### 0.5 Move `icon/` to a submodule (45 min)

First decide whether it is a *modified* copy of the upstream ICON code. Compare against the original repository (In-Context Operator Networks, Yang et al.): if `diff -r` shows only `h5loader copy.py`, `__pycache__`, `TEST.IPYNB` and the `.h5`, submodule the *upstream* repo directly at a pinned commit and you're done. If there are real modifications, create `xx257xx/icon-baseline`, push the folder's contents there (without the copy file, bytecode or fixture), and submodule that.

```bash
git rm -r icon                                              # after 0.3 the index no longer has the junk files
git commit -m "chore: remove vendored icon/ (moving to submodule)"
git submodule add https://github.com/<owner>/<icon-repo>.git third_party/icon
cd third_party/icon && git checkout <pinned-sha> && cd ../..
git add .gitmodules third_party/icon
git commit -m "chore: add icon baseline as third_party/icon submodule"
```
Submodule semantics: the superproject records *only* the SHA; `git clone --recurse-submodules` (or `git submodule update --init`) is needed to populate it. Add that to the README too. `uv` ignores it because it isn't a workspace member.

### 0.6 Fix the five breakages (1–1.5 h)

**a. `hypernet_init` typo, and make it impossible to recur.**
```bash
sed -i 's/hypernet_init: bias_hyperinit/hypernet_init: bias-hyperinit/' scripts/training/configs/model/*.yaml
grep -rn "hypernet_init" scripts/training/configs/model | grep -vc "bias-hyperinit"   # → 0
```
Then in `src/context_flux_no/nn/hypernetwork.py`, in `HypernetworkHead.__init__` *before* the loop that calls `_make_linear_layer` (around line 60; look for `for target in targets` / the `linears = []` setup):
```python
_INITIALIZATIONS = ("default", "bias-hyperinit")
...
        if initialization not in _INITIALIZATIONS:
            raise ValueError(
                f"Unknown hypernetwork initialization {initialization!r}; "
                f"expected one of {_INITIALIZATIONS}."
            )
```
A `Literal[...]` annotation is not enforced at runtime — that is the whole lesson of this bug. Add a unit test, `tests/test_nn_hypernetwork.py`:
```python
import pytest, jax, equinox as eqx
from context_flux_no.nn.hypernetwork import HypernetworkHead

def test_unknown_init_raises():
    target = eqx.nn.MLP(2, 2, 4, 1, key=jax.random.key(0))
    with pytest.raises(ValueError, match="Unknown hypernetwork initialization"):
        HypernetworkHead(target, in_size=8, num_blocks=2, initialization="bias_hyperinit", key=jax.random.key(1))
```
(Adjust constructor kwargs to the actual signature at line ~35.) **Decided (D15): `bias-hyperinit` becomes the default.** Change the default in the three model classes (`hyperfluxfno.py` lines 208, 343, 561: `hypernet_init: Literal["default", "bias-hyperinit"] = "bias-hyperinit"`) and in `HypernetworkHead.__init__` if it has a default. Any config that omits the key now gets the paper's initialisation; say so in the commit message. Keep the explicit `hypernet_init: bias-hyperinit` lines in the yamls anyway — explicit beats implicit for the census in 0.6e.

**b. Dangling `HyperFluxFNO` targets.**
```bash
git rm scripts/training/configs/model/hyperfluxfno_ViT.yaml \
       scripts/training/configs/model/hyperfluxno_2d_scalar_global.yaml \
       scripts/training/configs/model/hyperfluxno_2d_euler_global.yaml \
       scripts/training/train_multiphysics_2d.py
```
`train_multiphysics_2d.py` is on the Phase 1 deletion list anyway and cannot be imported; removing it here keeps the exit criterion ("every script imports") honest. Check nothing else references `HyperFluxFNO\b`: `grep -rn "HyperFluxFNO\b" src scripts tests` → empty.

**c. `data/__init__.py` exports.**
```python
from .sources import (
    TheWellDataSource as TheWellDataSource,
    WellDatasetSourceBase as WellDatasetSourceBase,
    ZarrWellDatasetSource as ZarrWellDatasetSource,
)
from .transforms import SpatialDownsample as SpatialDownsample   # removed again in Phase 1
```
Then update `scripts/training/train_multiphysics.py` line 7 to import from `context_flux_no.data` (it currently reaches into `.sources`); optional but it makes the package boundary real.

**d. Misnamed LSK config.** `git rm scripts/training/configs/model/hyperfluxno_2d_euler_lskflux.yaml` (it selects `ConvNextFluxNO`, and `hyperfluxno_2d_euler_ndflux.yaml`/the ConvNeXt run are covered elsewhere — check with `grep -l ConvNextFluxNO scripts/training/configs/model/*.yaml` that a ConvNeXt config survives; if not, rename this one to `hyperfluxno_2d_euler_convnext.yaml` instead of deleting).

**e. A config smoke test you will keep.** Create `scripts/check_configs.py`:
```python
"""Instantiate every model config against every data config it can pair with, on tiny shapes.
Usage: uv run python scripts/check_configs.py
"""
import itertools, sys
from pathlib import Path
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CFG_DIR = Path("scripts/training/configs").resolve()
MODELS = sorted(p.stem for p in (CFG_DIR / "model").glob("*.yaml"))
DATAS = sorted(p.stem for p in (CFG_DIR / "data").glob("*.yaml"))

failures = []
with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
    for m, d in itertools.product(MODELS, DATAS):
        try:
            cfg = compose(config_name="config", overrides=[f"model={m}", f"data={d}"])
            # shrink anything shape-bearing so instantiation is cheap
            if "encoder_kwargs" in cfg.model and "grid_size" in cfg.model.encoder_kwargs:
                cfg.model.encoder_kwargs.grid_size = [16] * cfg.data.num_spatial_dims
                cfg.model.encoder_kwargs.patch_size = [4] * cfg.data.num_spatial_dims
            hydra.utils.instantiate(cfg.model)
            print(f"OK    model={m} data={d}")
        except Exception as e:  # noqa: BLE001 — we want the full census
            failures.append((m, d, f"{type(e).__name__}: {e}"))
            print(f"FAIL  model={m} data={d}  {type(e).__name__}: {str(e)[:120]}")
print(f"\n{len(failures)} failing combinations")
sys.exit(1 if failures else 0)
```
Expect many "FAIL"s: every 1D model × 2D data and vice versa, and every model × legacy data yaml (different schema). That census *is* the P6 evidence, and Phase 7 will turn it into a zero. For Phase 0 the requirement is only: no failure whose message is `ImportError`/`AttributeError`/`ValueError: Unknown hypernetwork initialization`. Save the output as `docs/config_census_phase0.txt` — it's a useful before/after artefact.

Commit: `fix: bias-hyperinit spelling + runtime check; remove dangling HyperFluxFNO targets; export data sources; add config census script`.

### 0.7 Bias-HyperInit audit of reported runs (30–60 min)

The runs on wandb carry the *resolved* config, so the string is queryable:
```python
import wandb, pandas as pd
api = wandb.Api()
rows = []
for run in api.runs("jhko725/hyperfluxfno_revised"):      # entity/project from configs/config.yaml; repeat for older projects
    cfg = run.config
    rows.append(dict(id=run.id, name=run.name, state=run.state, created=run.created_at,
                     model=cfg.get("model", {}).get("_target_", "?").rsplit(".", 1)[-1],
                     data=cfg.get("data", {}).get("name"), seed=cfg.get("model", {}).get("key", {}).get("seed"),
                     hypernet_init=cfg.get("model", {}).get("hypernet_init")))
df = pd.DataFrame(rows)
print(df.groupby(["data", "model", "hypernet_init"]).size())
df.to_csv("docs/bias_hyperinit_audit.csv", index=False)
```
Cross-reference the runs behind each paper table (this is also where you'll feel the P9 pain — there is no run-id ↔ table mapping; the timestamps in the notebooks' `CHECKPOINT_DICT`s are the only link). Record the result in the post-submission plan doc (§10.9 of the plan). No repo commit beyond the CSV.

### 0.8 Regression harness (½–1 day — the most important step)

**Design.** Self-contained: each case stores its own small input batch, so tests never need the datasets. Goldens are produced once from the ICLR checkpoints and committed. Tests are skipped, not failed, when checkpoints are absent (so CI without checkpoints stays green).

**Determinism.** Generate *and* run goldens on CPU: `JAX_PLATFORMS=cpu`. XLA GPU reductions are not bitwise reproducible across versions/devices; on CPU, fp32 results are stable to ~1e-6 across runs, so `rtol=1e-5, atol=1e-6` is a real test rather than a flaky one (plan §10.4). A 5-step rollout of one sample on CPU takes seconds even for the 2D Euler model.

**Layout.**
```
tests/regression/
  conftest.py            # --checkpoints-root option, skip logic
  cases.yaml             # one entry per case
  make_goldens.py        # generator (run manually, never in pytest)
  inputs/<case>.npz      # one context window + loss_args, extracted once from the dataset
  goldens/<case>.npz     # u_next, rollout, metadata
  test_regression.py
```

`cases.yaml` — fill the timestamps from your checkpoint tree (`ls checkpoints/*/*/*/seed=0/`):
```yaml
- name: hyperfluxno_local_cubic1d
  checkpoint: cubicflux_1d/HyperFluxFNOLocal/PushforwardOneStepLoss/seed=0/<timestamp>
  input: inputs/cubic1d.npz            # context window [T=20, C=1, 128] + loss_args
  rollout_steps: 5
- name: hno_v2_euler2d
  checkpoint: euler_2d/HyperNeuralOperatorv2/PushforwardOneStepLoss/seed=0/<timestamp>
  input: inputs/euler2d.npz            # [20, 4, 128, 128]
  rollout_steps: 5
- name: hno_euler2d
  checkpoint: euler_2d/HyperNeuralOperator/...
  input: inputs/euler2d.npz
  rollout_steps: 5
- name: disco_cubic1d
  checkpoint: cubicflux_1d/DISCO/...
  input: inputs/cubic1d.npz
  rollout_steps: 5
- name: dpot_cubic1d
  checkpoint: cubicflux_1d/DPOT/...
  input: inputs/cubic1d.npz
  rollout_steps: 5
```
Add a 1D DISCO/DPOT *and* their 2D Euler counterparts if those checkpoints exist — the baselines are what Phase 2's `fv/` migration must *not* change.

**Extracting inputs** (one-off, in a scratch script or notebook): open the dataset the model was evaluated on, take trajectory 0, time steps `0:20` (context length from `configs/training/default.yaml`), and the `loss_args` the script would have passed (`(dt, dx)` for 1D; `(dt, dx, dy)` for 2D — from `ZarrWellDatasetSource.metadata_common` as in `train_multiphysics.py::get_loss_args`). Save with `np.savez(path, u=u, loss_args=np.asarray(loss_args))`. Keep it small: 1D case ≈ 10 kB, 2D case ≈ 5 MB (fine to commit once; `.gitignore` in 0.3 allows `tests/**/*.npz`).

`make_goldens.py`:
```python
"""Generate regression goldens from checkpoints. Run on CPU: JAX_PLATFORMS=cpu uv run python tests/regression/make_goldens.py --checkpoints-root ./checkpoints"""
import argparse, subprocess, sys
from pathlib import Path
import jax, numpy as np, yaml, equinox as eqx
from context_flux_no.training.io import load_model

HERE = Path(__file__).parent

def run_case(case, root):
    model = eqx.nn.inference_mode(load_model(root / case["checkpoint"]), True)
    inp = np.load(HERE / case["input"])
    u, args = jax.numpy.asarray(inp["u"]), tuple(float(a) for a in inp["loss_args"])
    u_next, _ = model(u, args, key=None)
    rollout, _ = model.rollout(u, args, case["rollout_steps"], key=None)
    return dict(u_next=np.asarray(u_next), rollout=np.asarray(rollout))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoints-root", type=Path, required=True); p.add_argument("--only", default=None)
    a = p.parse_args()
    assert jax.default_backend() == "cpu", "generate goldens on CPU for determinism"
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for case in yaml.safe_load((HERE / "cases.yaml").read_text()):
        if a.only and case["name"] != a.only: continue
        out = run_case(case, a.checkpoints_root)
        np.savez(HERE / "goldens" / f"{case['name']}.npz", **out, git_sha=sha, jax_version=jax.__version__, backend=jax.default_backend())
        print("wrote", case["name"], {k: v.shape for k, v in out.items()})
```
`conftest.py`:
```python
import os, pytest
from pathlib import Path
def pytest_addoption(parser):
    parser.addoption("--checkpoints-root", default=os.environ.get("CFNO_CHECKPOINTS"), help="root of checkpoint tree; regression tests skip if unset")
@pytest.fixture(scope="session")
def checkpoints_root(request):
    root = request.config.getoption("--checkpoints-root")
    if not root or not Path(root).exists():
        pytest.skip("regression checkpoints not available")
    return Path(root)
```
`test_regression.py`:
```python
import jax, numpy as np, pytest, yaml
from pathlib import Path
from tests.regression.make_goldens import run_case, HERE
CASES = yaml.safe_load((HERE / "cases.yaml").read_text())

@pytest.mark.regression
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_matches_golden(case, checkpoints_root):
    if jax.default_backend() != "cpu":
        pytest.skip("regression tolerances are calibrated for CPU; run with JAX_PLATFORMS=cpu")
    got = run_case(case, checkpoints_root)
    want = np.load(HERE / "goldens" / f"{case['name']}.npz")
    for k in ("u_next", "rollout"):
        np.testing.assert_allclose(got[k], want[k], rtol=1e-5, atol=1e-6, err_msg=k)
```
Register the marker in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = ["regression: needs ICLR checkpoints (CPU)", "slow: verification suite"]
testpaths = ["tests"]
```
Run: `JAX_PLATFORMS=cpu uv run python tests/regression/make_goldens.py --checkpoints-root ./checkpoints`, then `JAX_PLATFORMS=cpu uv run pytest tests/regression -v --checkpoints-root ./checkpoints`. Run the test twice; if the second run differs, the tolerance needs revisiting *now*, not in Phase 2.

Pitfalls: `load_model` re-instantiates via the stored Hydra config, so it needs the same class names to exist — this is why the shim in Phase 3 matters, and why Phase 1 must not delete any class that a golden's checkpoint uses. `model(u, args, key=None)` — some models split `key` unconditionally; if one raises on `None`, pass `jax.random.key(0)` and note it in `cases.yaml` (a dropout-free inference model must not depend on it; if outputs change with the key under `inference_mode`, that is a bug to log).

Commit: `test: regression harness with goldens from ICLR checkpoints`.

### 0.9 Tooling: formatting, type checking, dev deps, CI — *done by Claude*

```bash
uv add --dev hypothesis chex pre-commit ty
```
`pyproject.toml` additions:
```toml
[tool.ruff]            # keep existing; add
line-length = 88
[tool.ruff.format]
docstring-code-format = true
```
Make CUDA an optional extra so CI can install the CPU wheel (today `jax[cuda13]` is unconditional):
```toml
dependencies = [ "jax>=0.9.0.1", ... ]          # drop the [cuda13] here
[project.optional-dependencies]
cuda = ["jax[cuda13]>=0.9.0.1"]
```
and document `uv sync --extra cuda` for GPU machines (`uv.lock` will change; that is expected).

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: <current>
    hooks: [{id: ruff, args: [--fix]}, {id: ruff-format}]
  - repo: https://github.com/kynan/nbstripout
    rev: <current>
    hooks: [{id: nbstripout}]
```
`uv run pre-commit install`. Run `uv run ruff format .` once and commit that separately (`style: ruff format`) so the diff noise is isolated from real changes.

Minimal CI, `.github/workflows/test.yml`:
```yaml
name: test
on: [push, pull_request]
jobs:
  fast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync                      # CPU jax via the extra split above
      - run: uv run ruff check . && uv run ruff format --check .
      - run: JAX_PLATFORMS=cpu uv run pytest -m "not regression and not slow" -q
```
Regression and verification tiers stay local (checkpoints, time). Commit: `chore: dev tooling, pre-commit, CPU CI`.

### 0.10 README and metadata — *done by Claude*

Fill `README.md` with: one-paragraph description; install (`uv sync`, `--extra cuda`, `nbstripout --install`, `git submodule update --init`); the three commands (generate / train / evaluate — the current script names for now); where checkpoints and data live; how to run each test tier; link to the plan doc. Fix `pyproject` `description`, delete the `[project.scripts]` hello-world entry and `main()` in `src/context_flux_no/__init__.py`. Update `scripts/training/README.md` so the listed `model=` values match the yaml names. Commit: `docs: README and project metadata`.

### Exit checklist

- [ ] `uv run pytest -m "not regression"` green; `JAX_PLATFORMS=cpu uv run pytest tests/regression --checkpoints-root ./checkpoints` green, twice
- [ ] `uv run python scripts/check_configs.py` shows no Import/Attribute/init-string errors (schema mismatches are expected)
- [ ] `uv run python scripts/training/train_multiphysics.py model=hyperfluxno_2d_euler_v2 --cfg job` composes; a 20-step real run with `training.max_steps=20` starts and checkpoints
- [ ] `git ls-files | grep -cE "\.pyc$|\.h5$"` → 0; `du -sh notebooks` < 1 MB; `icon/` gone, `third_party/icon` present
- [ ] CI green on the branch
- [ ] `docs/bias_hyperinit_audit.csv` written and its conclusion recorded in the post-submission doc

---
