"""Generate regression goldens from the ICLR checkpoints. CPU only.

    JAX_PLATFORMS=cpu uv run python tests/regression/make_goldens.py --checkpoints-root ./checkpoints

Writes tests/regression/goldens/<case>.npz with u_next, rollout and provenance (git SHA, jax
version, backend). Refuses to run on a non-CPU backend. Re-running overwrites: do that only with a
change-document entry saying which commit changed behaviour and why (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from tests.regression.common import (
    case_is_configured,
    golden_path,
    GOLDENS,
    input_path,
    load_cases,
    run_case,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoints-root", type=Path, required=True)
    ap.add_argument("--only", default=None, help="case name to (re)generate")
    ap.add_argument("--force", action="store_true", help="overwrite an existing golden")
    args = ap.parse_args()

    import jax

    if jax.default_backend() != "cpu":
        raise SystemExit("generate goldens on CPU for determinism: JAX_PLATFORMS=cpu")

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.run(["git", "diff", "--quiet"]).returncode)
    GOLDENS.mkdir(exist_ok=True)

    for case in load_cases():
        if args.only and case["name"] != args.only:
            continue
        if not case_is_configured(case):
            print(f"skip  {case['name']}: checkpoint path still has a placeholder")
            continue
        if not input_path(case).exists():
            print(
                f"skip  {case['name']}: missing {input_path(case).name} (run make_inputs.py)"
            )
            continue
        out = golden_path(case)
        if out.exists() and not args.force:
            print(f"keep  {case['name']}: golden exists (use --force to overwrite)")
            continue
        result = run_case(case, args.checkpoints_root)
        np.savez_compressed(
            out,
            **result,
            git_sha=sha,
            git_dirty=dirty,
            jax_version=jax.__version__,
            backend=jax.default_backend(),
            checkpoint=case["checkpoint"],
        )
        print(
            f"wrote {out.name}  "
            + "  ".join(f"{k}{v.shape}" for k, v in result.items())
        )


if __name__ == "__main__":
    main()
