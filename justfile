# Adapted from Jhko725/UniversalForceReconstruction justfile at b80a341.
# The project's commands, one recipe each; `uv run just --list` prints them. CI calls
# these recipes too, so each command list is written once.

set shell := ["bash", "-euo", "pipefail", "-c"]

# The clean scope of rule R7, from `clean` in pyproject.toml; a path that does not exist
# yet is skipped.
clean := `uv run python -c 'import os, tomllib; print(*(p for p in tomllib.load(open("pyproject.toml", "rb"))["tool"]["context_flux_no"]["clean"] if os.path.exists(p)))'`

# Show the recipes
default:
    @just --list

# Sync the environment and install the git hooks
setup:
    uv sync
    uv run prek install

# `check` ends with a grep because ty honors ignore comments, so a clean run does not
# prove there are none. ty and the grep cover every clean path. `burgers.py` is excluded
# from ty until `ViscousBurgers1D.solve`, which returns the cell centers as a tuple, and
# `AbstractPDEFamily.solve`, which declares a list, agree on that type.

# Lint, format, spelling, document, docstring, container and import checks, then ty;
# `prek` sees tracked files only
check ty="uv run ty":
    uv run prek run --all-files
    {{ty}} check {{clean}} --exclude src/context_flux_no/simulations/pde/burgers.py
    ! grep -rn 'ty: ignore' {{clean}}

# Examples: `just test` (dev profile), `just test ci`, `just test dev tests/test_x.py`,
# `just test dev tests/test_x.py -k "a or b"`. Positional arguments, so that a quoted
# `-k` expression reaches pytest as one argument.

# The quick tier
[positional-arguments]
test profile="dev" *paths="tests":
    uv run pytest "${@:2}" -m "not slow" --hypothesis-profile="$1"

# The slow tier
test-slow profile="dev":
    uv run pytest -m slow --hypothesis-profile={{profile}}

# The conventions document is left out of `pending`: it defines the marker rather than
# using it.

# Every [unconfirmed] marker under docs/ with its file and line
pending:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -d docs ] || exit 0
    grep -rnF --include='*.md' --exclude=document-conventions.md --exclude-dir=plans '[unconfirmed]' docs || true
