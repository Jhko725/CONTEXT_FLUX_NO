import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--checkpoints-root",
        default=os.environ.get("CFNO_CHECKPOINTS"),
        help="root of the ICLR checkpoint tree; regression tests skip when unset or missing",
    )


@pytest.fixture(scope="session")
def checkpoints_root(request) -> Path:
    root = request.config.getoption("--checkpoints-root")
    if not root or not Path(root).exists():
        pytest.skip(
            "regression checkpoints not available (set --checkpoints-root or CFNO_CHECKPOINTS)"
        )
    return Path(root)
