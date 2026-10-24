"""Import every module of the package; fail loudly on any broken import.

    uv run python scripts/import_all.py

Modules whose only failure is a missing optional native dependency (clawpack/pyclaw) are
reported as skipped, so the script is useful on machines without a Fortran toolchain; CI has one
and treats those as failures too (--strict).
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys

import context_flux_no as pkg


OPTIONAL_NATIVE = ("clawpack", "pyclaw")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strict", action="store_true", help="treat missing clawpack as a failure"
    )
    args = ap.parse_args()

    failed, skipped = [], []
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        try:
            importlib.import_module(m.name)
        except ModuleNotFoundError as e:
            if not args.strict and any(e.name.startswith(n) for n in OPTIONAL_NATIVE):
                skipped.append((m.name, e.name))
            else:
                failed.append((m.name, repr(e)))
        except Exception as e:  # noqa: BLE001 — any import-time error is a failure
            failed.append((m.name, repr(e)))
    for name, dep in skipped:
        print(f"SKIP {name}: optional native dependency {dep!r} not installed")
    for name, err in failed:
        print(f"FAIL {name}: {err}")
    print(f"{len(failed)} failed, {len(skipped)} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
