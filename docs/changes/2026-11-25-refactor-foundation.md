---
type: change
status: in-progress
updated: 2026-11-25
branch: refactor-foundation
roadmap: Phase 1b (foundation); closes 4.5, 6.1 and the entry-point half of 7.1
sources:
  - "Project document claude/refactor-foundation-handoff-2026-11-25.md (the handoff)"
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
| `docs/changes/2026-11-25-workflow-bootstrap.md` — deleted: its content moved into this document | | |
| `docs/changes/2026-11-25-refactor-foundation.md` — this document | | |
| `justfile` — new: recipes `default`, `setup`, `check`, `test`, `test-slow` and `pending`, adapted from `UniversalForceReconstruction` |  |  |
| `prek.toml` — new: the hooks (builtin, ruff, codespell, `check-spelling`, `check-docs`) run through `prek` |  |  |
| `.pre-commit-config.yaml` — deleted: replaced by `prek.toml` |  |  |
| `.github/workflows/test.yml` — the `fast` job runs `just check` and `just test ci`; pushes trigger on `main` only |  |  |
| `scripts/checks/check_docs.py` — moved from `scripts/`; the `--files` mode and its *Files* rule removed; root path follows the move |  |  |
| `scripts/checks/check_spelling.py` — moved from `scripts/`; usage line follows the move |  |  |
| `pyproject.toml` — dev group: `prek` and `rust-just` added, `pre-commit` and `nbstripout` removed; codespell comment path |  |  |
| `uv.lock` — relocked for the dev group change |  |  |
| `README.md` — path of `check_docs.py` updated; the hook install command is `just setup` |  |  |
| `docs/coding-standards.md` — paths of `check_docs.py` and `check_spelling.py` updated, nothing else |  |  |
| `docs/roadmap.md` — paths of `check_docs.py` and `check_spelling.py` updated, nothing else |  |  |
| `scripts/data/` — trailing whitespace and final newlines fixed by the new hooks, no other change |  |  |
| `scripts/training/configs/` — trailing whitespace and final newlines fixed by the new hooks, no other change |  |  |
| `src/context_flux_no/` — trailing whitespace and final newlines fixed by the new hooks, no other change |  |  |
| `tests/conftest.py` — new: the Hypothesis profiles `dev`, `default` and `ci` that `just test` selects |  |  |
