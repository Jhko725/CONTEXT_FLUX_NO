"""Shared helpers for the regression tier (used by make_goldens.py and test_regression.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


HERE = Path(__file__).parent
INPUTS = HERE / "inputs"
GOLDENS = HERE / "goldens"
CONTEXT_LENGTH = 20  # configs/training/default.yaml: context_length

RTOL = 1e-5
ATOL = 1e-6


def load_cases() -> list[dict[str, Any]]:
    cases = yaml.safe_load((HERE / "cases.yaml").read_text(encoding="utf-8")) or []
    for case in cases:
        case.setdefault("rollout_steps", 5)
    return cases


def case_is_configured(case: dict[str, Any]) -> bool:
    """A case is configured once its checkpoint path has no placeholder left."""
    return "<" not in case["checkpoint"]


def input_path(case: dict[str, Any]) -> Path:
    return INPUTS / f"{case['input']}.npz"


def golden_path(case: dict[str, Any]) -> Path:
    return GOLDENS / f"{case['name']}.npz"


def load_input(case: dict[str, Any]) -> tuple[np.ndarray, tuple[float, ...]]:
    data = np.load(input_path(case))
    return data["u"], tuple(float(a) for a in data["loss_args"])


def run_case(case: dict[str, Any], checkpoints_root: Path) -> dict[str, np.ndarray]:
    """Load the checkpoint, run one forward pass and a rollout on the stored input.

    Imports are local so that collecting the tests never imports JAX-heavy modules.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from context_flux_no.training.io import load_model

    model = eqx.nn.inference_mode(
        load_model(checkpoints_root / case["checkpoint"]), True
    )
    u_np, loss_args = load_input(case)
    u = jnp.asarray(u_np)
    # Inference-mode models must not depend on the key; a fixed key keeps models that split
    # unconditionally from raising on None.
    key = jax.random.key(0)
    u_next, _ = model(u, loss_args, key=key)
    rollout, _ = model.rollout(u, loss_args, case["rollout_steps"], key=key)
    return {"u_next": np.asarray(u_next), "rollout": np.asarray(rollout)}
