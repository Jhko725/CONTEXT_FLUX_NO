# Working conventions for Claude in this repository

Joon's working rules are in `.claude/rules/`: `working-rules.md` for every session, and
`python.md` when a Python file is read. They are copies of his global `CLAUDE.md`, which
a cloud session does not load. The conventions they rest on are in the
`code-conventions` skill. Where a skill and this repository's files disagree, this
repository's files take precedence: the skill serves several repositories and can
describe a format that this one has replaced. This file carries the session protocol
and the facts Claude cannot infer from the code. Where each convention is enforced here
is [`docs/coding-standards.md`](docs/coding-standards.md) for code and
[`docs/document-conventions.md`](docs/document-conventions.md) for documents; both name
the checks that stand in for prose.

## Start of a session

This reading list is for a session that plans, reviews or orchestrates. A subagent that
implements one commit from a brief reads what its brief names, and nothing else here.

1. Read `docs/index.md`, then `docs/architecture.md` (what the code is now), then the
   current phase in `docs/roadmap.md` (what is being done and why).
2. Find the branch's change document `docs/changes/<YYYY-MM-DD>-<branch>.md`. Its
   *Review notes* table is the current state of the review: rows with *Thoughts* or
   *Modifications* filled in are instructions for this session, and those two columns
   are never edited except when asked. When a handoff document planned the branch, its
   committed copy is under `docs/plans/`.
3. Check `docs/findings.md` for anything that constrains the task, and read the ADRs in
   `docs/decisions/` that cover the area being touched.

## During and at the end of a session

- The change document follows `docs/document-conventions.md` § Change documents and is
  updated in the same commit as the code. Its sections are *Summary*, *Design*, *Bugs
  fixed*, *Tests*, *Open issues* and *Review notes*, in that order, and there is no
  *Files* section. *Review notes* is a log of rounds, each under
  ``### Round <n>: from `<commit>` `` with a table `Change | Thoughts | Modifications`.
  Every file a commit touches gets a row in the open round, which is the last one, in
  that commit. A *Change* cell is one backticked path, a dash, and what changed, on one
  line; a path that ends in `/` covers every changed file under it.
- A round is closed once Joon has filled a *Thoughts* cell of it. Its rows are never
  edited again, and the next change opens a new round that names his latest review
  commit. Round 1 starts from the merge base with `origin/main`. Claude fills
  *Modifications* only when asked. `scripts/checks/check_changes.py` checks the rounds.
- A test's outcome is stated only after that test has run on the code the document
  describes.
- A decision that Claude takes without Joon is marked `[unconfirmed]` where it is
  stated; `just pending` lists every marker. A document whose status is `agreed`,
  `accepted`, `amended` or `merged` carries none.
- `docs/roadmap.md` is the single current plan: finished items are appended to its
  *Done* ledger, never inserted or deleted, in the same commit as the work.
- `docs/architecture.md` follows every module move, rename, addition or removal;
  `docs/index.md` lists every document under `docs/` with its status;
  `docs/findings.md` is appended if anything concerns a reported result, and
  `docs/verification.md` is updated if a numerics claim gained or lost a test.
- `uv run just check` is clean, and the test files of the touched modules pass at the
  `dev` profile (`uv run just test dev <files>`); the full tier runs once at the end.

## Repository facts

- Python 3.13 (`.python-version`, `requires-python`), so `copy.replace` is the
  copy-with-changes. `uv sync` gives a CPU environment with the `dev` group; add
  `--extra cuda` on GPU machines. `clawpack` builds from source, so `uv sync` needs a
  Fortran compiler (`gfortran`). Set `JAX_PLATFORMS=cpu` in a cloud session, which has
  2 cores and 7 GB of memory.
- Every command is a recipe in the `justfile` (`uv run just --list`), and CI runs the
  same recipes. Tiers, tolerances and the checks are in `docs/coding-standards.md`.
- `just check` runs `prek run --all-files`, which sees only tracked files: `git add` a
  new file before running it, or the check passes without having read the file.
- The session that changes a dependency regenerates `uv.lock` with `uv lock` and
  commits it with the change.
- A pushed commit is never amended or rebased, and a branch is never force-pushed: a
  change to a pushed commit is a new commit on top, because Joon may already have
  merged it. A hook in `.claude/settings.json` blocks every `git rebase` and
  force-push, and a `git commit --amend` of a pushed HEAD.
- `.claude/agents/implementer.md` is the subagent that implements one commit of a
  handoff document; a hook in its frontmatter blocks `git push` while it runs.
- Tags cannot be pushed from Claude's sessions (proxy policy); Joon pushes tags.
- The strict checks apply to the clean paths listed under `clean` in
  `[tool.context_flux_no]` of `pyproject.toml`. Everything else in the package is under
  `src/context_flux_no/legacy/`, which stays importable and otherwise untouched until
  the branch that rewrites it. A module joins the clean list by leaving `legacy/` and
  never returns.
- `import juliacall` or `import juliapkg` *installs Julia* (1 GB) on a machine without
  it; probe for the Julia side with `simulations.backends.trixi.available()`, never by
  importing. Set-up is `uv run python scripts/setup_julia.py`.
- `git log -p docs/roadmap.md` is the project diary; `git blame` dates every Done row.
- Numerics conventions (stencil `(a, b)`, faces, padding, BCs, axis handling) are in
  `docs/design/fv-conventions.md` and are not restated in code comments.
- Provenance labels in design documents: **[cited]** = from a named source, **[code]**
  = verified in the tree, **[deduced]** = Claude's inference. Keep them.
