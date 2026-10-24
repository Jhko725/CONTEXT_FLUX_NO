"""Give the legacy ICLR checkpoints a `metadata.json` so `training.runs.index()` lists them
(ADR-0010, Phase 5). Read-only on the checkpoints themselves.

The ICLR trainer wrote `<root>/<data.name>/<ModelClass>/<LossClass>/seed=<n>/<yy-mm-dd-HH_MM_SS>/`
with orbax step folders inside and the **model config** as orbax `custom_metadata`. This script
writes, into each timestamp directory, a `RunRecord` with `status="legacy"`,
`run_id="legacy-<timestamp>"`, the index columns recovered from the path and the model config,
and — when `--audit docs/bias_hyperinit_audit.csv` (from `scripts/audit_bias_hyperinit.py`) is
given — the wandb runs whose (model, data, seed, encoder, target_network) match, so the
checkpoint can be tied back to its wandb run (`extra.wandb_candidates`; `wandb` is filled when
the match is unique).

    uv run python scripts/backfill_run_records.py --root ./checkpoints --audit docs/bias_hyperinit_audit.csv
    uv run python scripts/backfill_run_records.py --root ./checkpoints --dry-run

Existing `metadata.json` files are left alone unless `--force`.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from context_flux_no.training.run import METADATA_FILE
from context_flux_no.training.run_record import index_from_config, RunRecord
from orbax.checkpoint import v1 as ocp


TIMESTAMP = re.compile(r"^\d{2}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}$")


def legacy_dirs(root: Path) -> list[Path]:
    """Timestamp directories five levels under `root` that contain orbax step folders."""
    out = []
    for ts in sorted(root.glob("*/*/*/seed=*/*")):
        if (
            ts.is_dir()
            and TIMESTAMP.match(ts.name)
            and any(p.is_dir() and p.name.isdigit() for p in ts.iterdir())
        ):
            out.append(ts)
    return out


def read_model_config(ckpt_dir: Path) -> tuple[dict[str, Any], list[int]]:
    with ocp.training.Checkpointer(str(ckpt_dir.resolve())) as c:
        meta = c.root_metadata().custom_metadata or {}
        steps = sorted(int(ci.step) for ci in c.checkpoints)
    return dict(meta), steps


def load_audit(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def match_audit(
    rows: list[dict[str, str]], index: dict[str, Any]
) -> list[dict[str, str]]:
    def same(row: dict[str, str], key: str, col: str) -> bool:
        v = index.get(key)
        return (row.get(col) or "") == ("" if v is None else str(v))

    return [
        r
        for r in rows
        if same(r, "model", "model")
        and same(r, "data", "data")
        and same(r, "seed", "seed")
        and same(r, "encoder", "encoder")
        and same(r, "target_network", "target_network")
    ]


def backfill(
    ckpt_dir: Path, audit: list[dict[str, str]], *, force: bool, dry_run: bool
) -> str:
    target = ckpt_dir / METADATA_FILE
    if target.exists() and not force:
        return "skip (exists)"
    data_name, model_cls, loss_cls, seed_part, ts = ckpt_dir.parts[-5:]
    model_config, steps = read_model_config(ckpt_dir)
    cfg = {
        "model": model_config,
        "data": {"name": data_name},
        "loss_fn": {"_target_": loss_cls},
    }
    index = index_from_config(cfg)
    index["model"] = index["model"] or model_cls
    index["seed"] = (
        index["seed"] if index["seed"] is not None else int(seed_part.split("=")[1])
    )
    candidates = match_audit(audit, index)
    record = RunRecord.create(
        index=index,
        model_config=model_config,
        metric_name="valid_loss",
        run_id=f"legacy-{ts}",
        extra={
            "checkpoint_dir": ".",
            "legacy_path": str(ckpt_dir),
            "steps": steps,
            "wandb_candidates": [
                {
                    k: c.get(k)
                    for k in ("project", "id", "url", "created", "hypernet_init")
                }
                for c in candidates
            ],
            "hypernet_init": model_config.get("hypernet_init"),
        },
    ).replace(
        status="legacy",
        git={},  # the checkpoint's commit is unknown; this script's commit is not it
        package_version="",
        final_step=steps[-1] if steps else None,
        best_step=steps[-1] if len(steps) == 1 else None,
        wandb=(
            {
                "id": candidates[0].get("id"),
                "url": candidates[0].get("url"),
                "project": candidates[0].get("project"),
                "entity": None,
                "mode": "legacy",
            }
            if len(candidates) == 1
            else None
        ),
    )
    if not dry_run:
        record.write(target)
    return (
        f"{'would write' if dry_run else 'wrote'} ({len(candidates)} wandb match(es))"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, default=Path("./checkpoints"))
    ap.add_argument("--audit", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    audit = load_audit(args.audit)
    dirs = legacy_dirs(args.root)
    if not dirs:
        print(f"no legacy checkpoint directories under {args.root}")
    for d in dirs:
        print(f"{d}: {backfill(d, audit, force=args.force, dry_run=args.dry_run)}")


if __name__ == "__main__":
    main()
