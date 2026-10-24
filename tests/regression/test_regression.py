"""Regression tier: ICLR checkpoints must reproduce their goldens (docs/verification.md V22)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.regression.common import (
    ATOL,
    case_is_configured,
    golden_path,
    input_path,
    load_cases,
    RTOL,
    run_case,
)


CASES = load_cases()


@pytest.mark.regression
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_matches_golden(case, checkpoints_root):
    import jax

    if jax.default_backend() != "cpu":
        pytest.skip(
            "regression tolerances are calibrated for CPU; run with JAX_PLATFORMS=cpu"
        )
    if not case_is_configured(case):
        pytest.skip("checkpoint path not configured in cases.yaml")
    if not input_path(case).exists() or not golden_path(case).exists():
        pytest.skip("input or golden missing (run make_inputs.py / make_goldens.py)")

    got = run_case(case, checkpoints_root)
    want = np.load(golden_path(case))
    for key in ("u_next", "rollout"):
        np.testing.assert_allclose(
            got[key], want[key], rtol=RTOL, atol=ATOL, err_msg=key
        )


def test_cases_are_well_formed():
    """Runs everywhere: the cases file itself must be sane even without checkpoints."""
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names)), "duplicate case names"
    for case in CASES:
        assert set(case) >= {
            "name",
            "checkpoint",
            "input",
            "dataset",
            "rollout_steps",
        }, case["name"]
        assert case["rollout_steps"] >= 1
