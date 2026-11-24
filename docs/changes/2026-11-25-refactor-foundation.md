---
type: change
status: in-progress
updated: 2026-11-25
branch: refactor-foundation
roadmap: Phase 1b (foundation); closes 4.5, 6.1 and the entry-point half of 7.1
sources:
  - "docs/plans/2026-11-25-refactor-foundation.md (the handoff)"
---

# 2026-11-25 — `refactor-foundation`

## Summary

The branch gives the repository the working conventions of
`UniversalForceReconstruction`, removes the code that kept the ICLR checkpoints
loadable, moves the code that awaits its rewrite under `legacy/`, and replaces the
training entry point, the data loader and the evaluation code. It carries the commit of
`workflow-bootstrap` (`d60a19a`), which is not merged separately. The orchestrating
session completes this section in the documents unit.

## Design

The first three points come from the change document of `workflow-bootstrap`, which
this document replaces: one branch has one change document.

- **The rules are copied into the repository.** A cloud session does not load the global `CLAUDE.md`, and until now the rules reached a session only through the Project. Rejected: a pointer to the Project, which a session started from the repository alone cannot follow.
- **The hooks are copied unchanged**, each with a first line naming its source and commit. The amend rule of `block_history_rewrite.py` asks the remote directly, which matters in the shallow clones of a cloud session.
- **`implementer.md` names `src/context_flux_no/legacy/` before that directory exists.** `refactor-foundation` creates it; the boundary is stated here so that the agent file does not change in the middle of the run that uses it.

## Bugs fixed

None yet.

## Tests

The baseline of unit 0 is recorded under *Open issues* until the full tier runs at the
end of the branch.

## Open issues

- The hooks have no automated test in this repository. `UniversalForceReconstruction` has none either.
- **The agent type.** The session started outside the repository and offers no `implementer` type. Each unit runs as a general-purpose subagent on the unit's model, briefed with the *Boundary*, *Stop rule*, *Before and after* and *Report* sections of `.claude/agents/implementer.md` and told not to run `git push`, since the hook of the agent file does not run for that type (handoff section 4, step 3).
- **Six change documents predate the format, where the handoff says eight.** `docs/changes/` at `450f31d` holds six documents dated before 2026-11-25. `PRE_FORMAT_CHANGES` lists those six. [unconfirmed]

- **The new hooks rewrote whitespace in 45 tracked files** (unit 1): trailing spaces and final newlines, among them files that unit 7 moves under `legacy/`. `git diff --ignore-space-at-eol` shows no other change. The alternative was a wider `exclude` in `prek.toml`. [unconfirmed]
- **`tests/conftest.py` arrives in unit 1 with the Hypothesis profiles only.** The handoff creates it in unit 16, but `just test dev` needs the `dev` profile from unit 1 on. Unit 16 adds the import hook. [unconfirmed]
- The item-3 grep of unit 1 filters `^./docs/changes/`, which GNU grep's output never matches; with `^docs/changes/` it prints nothing.

- **ADR conversion (unit 2, W5).** No ADR had *Forces* text, so each `### Forces` holds `*[not recorded]*`, and the template allows that marker. *History* opens with an "accepted" line dated from the ADR's status line. The extra sections of ADR-0001, 0004, 0010 and 0011 follow *Decision* in their original order. Compared as multisets of lines, no sentence of an ADR changed. [unconfirmed]
- **The document types `guide`, `format` and `dataset` and the status `historical` are not in the vocabulary** (unit 2): no document here uses them. [unconfirmed]
- **`check_docs.py` has no architecture rule.** The handoff asked to keep the old check's rule, and the old check had none; the rule of `UniversalForceReconstruction` reports 19 problems on the fenced module map of `docs/architecture.md`. [unconfirmed]
- **The glossary stays a bullet list.** `docs/document-conventions.md` describes the table form, which the glossary takes when it is next rewritten. [unconfirmed]

## Review notes

### Round 1: from `450f31d`

| Change | Thoughts | Modifications |
|---|---|---|
| `.claude/rules/working-rules.md` — copied: the working rules of Joon's global `CLAUDE.md` | | |
| `.claude/rules/python.md` — copied: the design, Python and JAX rules | | |
| `.claude/agents/implementer.md` — adapted: the subagent that implements one commit of a handoff document | | |
| `.claude/hooks/block_history_rewrite.py` — copied: blocks a rebase, a force-push and an amend of a pushed commit | | |
| `.claude/hooks/block_push.py` — copied: blocks `git push` while the implementer runs | | |
| `.claude/settings.json` — registers the history hook | | |
| `CLAUDE.md` — points at `.claude/rules/`, states the history rule and names the agent; check paths follow the move to `scripts/checks/` | | |
| `docs/index.md` — the row of `workflow-bootstrap` replaced by the row of this branch; check paths follow the move | | |
| `docs/changes/2026-11-25-refactor-foundation.md` — this document | | |
| `justfile` — new: recipes `default`, `setup`, `check`, `test`, `test-slow` and `pending`, adapted from `UniversalForceReconstruction` |  |  |
| `prek.toml` — new: the hooks (builtin, ruff, codespell, `check-spelling`, `check-docs`) run through `prek` |  |  |
| `.pre-commit-config.yaml` — deleted: replaced by `prek.toml` |  |  |
| `.github/workflows/test.yml` — the `fast` job runs `just check` and `just test ci`; pushes trigger on `main` only |  |  |
| `scripts/check_docs.py` — moved to `scripts/checks/` |  |  |
| `scripts/checks/check_docs.py` — moved from `scripts/`; the `--files` mode and its *Files* rule removed; root path follows the move |  |  |
| `scripts/checks/check_spelling.py` — moved from `scripts/check_spelling.py`; usage line follows the move |  |  |
| `pyproject.toml` — dev group: `prek` and `rust-just` added, `pre-commit` and `nbstripout` removed; codespell comment path |  |  |
| `uv.lock` — relocked for the dev group change |  |  |
| `README.md` — path of `check_docs.py` updated; the hook install command is `just setup` |  |  |
| `docs/coding-standards.md` — paths of `check_docs.py` and `check_spelling.py` updated, nothing else |  |  |
| `docs/roadmap.md` — paths of `check_docs.py` and `check_spelling.py` updated, nothing else |  |  |
| `scripts/data/` — trailing whitespace and final newlines fixed by the new hooks, no other change |  |  |
| `scripts/training/configs/` — trailing whitespace and final newlines fixed by the new hooks, no other change |  |  |
| `src/context_flux_no/` — trailing whitespace and final newlines fixed by the new hooks, no other change |  |  |
| `tests/conftest.py` — new: the Hypothesis profiles `dev`, `default` and `ci` that `just test` selects |  |  |
| `docs/document-conventions.md` — new: the form of every document type, adapted from `UniversalForceReconstruction`, with rules T1 and T2 |  |  |
| `scripts/checks/check_docs.py` — replaced by `UniversalForceReconstruction`'s: closed vocabularies, required headings with `PRE_FORMAT_CHANGES`, index links, Done ledger, ADR agreement, unconfirmed markers, templates; keeps the `updated` and `branch` rules |  |  |
| `scripts/checks/check_md_math.py` — new, copied from `UniversalForceReconstruction`: both Markdown previews and the frontmatter's form |  |  |
| `scripts/checks/markdown.py` — new, copied from `UniversalForceReconstruction`: the section tree the two checks read |  |  |
| `prek.toml` — registers `check-md-math`; `check-docs` runs on the documents and its own two modules |  |  |
| `pyproject.toml` — dev group: `markdown-it-py` added |  |  |
| `uv.lock` — relocked for `markdown-it-py` |  |  |
| `docs/decisions/` — the eleven ADRs moved under the template's headings (*Motivation*, *Forces* marked *[not recorded]*, *Alternatives rejected*, *History*); no sentence changed; `updated` and `sources` in the frontmatter |  |  |
| `docs/design/` — `kind` added (`fv-conventions.md` derivation, the other four note); `sources` as quoted block lists; `verified_by` of `python-julia-repo.md` quoted so the block parses |  |  |
| `docs/architecture.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/findings.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/glossary.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/roadmap.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/verification.md` — frontmatter only: `sources` as a quoted block list |  |  |
| `docs/index.md` — links `document-conventions.md`; ADRs under *In force* and *Superseded*; ADR-0004 and ADR-0011 shown as amended; the status list replaced by a pointer to the conventions |  |  |
| `scripts/checks/check_changes.py` — new, adapted from `UniversalForceReconstruction`: each round's rows cover exactly the files changed in it (rule T1); base `origin/main`; skips `PRE_FORMAT_CHANGES` |  |  |
| `prek.toml` — registers `check-changes`, which runs on every commit |  |  |
| `CLAUDE.md` — rewritten after `UniversalForceReconstruction`'s: reading list for planning sessions, rules T1 and T2, repository facts; the *Files* rule, notebook filter, submodule and regression tier removed |  |  |
| `docs/plans/2026-11-25-refactor-foundation.md` — new: the handoff document of this branch, verbatim under a frontmatter block |  |  |
| `src/context_flux_no/training/io.py` — deleted with its test `tests/test_training_io.py`: it loaded the ICLR checkpoints (`load_model`, `translate_legacy_config`) |  |  |
| `tests/test_training_io.py` — deleted with `training/io.py` |  |  |
| `tests/regression/` — deleted: the regression tier compared the ICLR checkpoints with goldens and has no replacement |  |  |
| `scripts/diff_checkpoint_leaves.py` — deleted: it compared ICLR checkpoint leaves |  |  |
| `scripts/audit_bias_hyperinit.py` — deleted: it audited the ICLR checkpoints |  |  |
| `scripts/backfill_run_records.py` — deleted: it wrote records for ICLR checkpoint directories |  |  |
| `tests/unit/test_backfill_run_records.py` — deleted with the script it tested |  |  |
| `src/context_flux_no/runs/index.py` — the two `legacy` branches of `load_state` and `load_run` and their docstring sentences removed |  |  |
| `src/context_flux_no/runs/record.py` — `Status` loses `"legacy"`; `RunRecord.from_dict` (so `read`) raises `ValueError` on an unknown status; legacy docstring lines reworded |  |  |
| `tests/unit/test_run_record.py` — the round trip no longer draws `legacy`; new test that `read` raises on an unknown status |  |  |
| `pyproject.toml` — `regression` marker removed; `addopts = "-m 'not slow'"` |  |  |
| `src/context_flux_no/training/loader.py` — deleted: the deprecated xarray segment loader (`SegmentLoaderBackground`, `make_segment_axis`), which nothing imported |  |  |
| `src/context_flux_no/training/__init__.py` — the export of `SegmentLoaderBackground` removed |  |  |
| `src/context_flux_no/data/sources.py` — the comment in `IO_PARAMS` no longer names xarray and h5netcdf; `TheWellDataSource` opens files with `h5py.File` |  |  |
| `pyproject.toml` — `dask[distributed]`, `h5netcdf`, `netcdf4` and `xarray` removed from the dependencies; the notebook and submodule entries removed from ruff's `extend-include` and `extend-exclude` and from codespell's `skip` |  |  |
| `uv.lock` — relocked without the four libraries and their dependencies |  |  |
| `notebooks/` — deleted: five legacy notebook directories |  |  |
| `third_party/icon` — the ICON baseline submodule removed |  |  |
| `.gitmodules` — deleted: it held only the ICON submodule |  |  |
| `prek.toml` — `exclude` loses `notebooks/` and `third_party/` |  |  |
| `scripts/checks/check_spelling.py` — `SKIP_PARTS` loses `third_party`, `icon` and `notebooks` |  |  |
