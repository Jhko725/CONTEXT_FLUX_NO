# Regression tier — goldens from the ICLR checkpoints

Pins the forward pass and a short autoregressive rollout of every model class that has an ICLR
checkpoint, so that the Phase 2–3 refactors can be checked against the submitted behaviour
(ADR-0008; `docs/verification.md` row V22). Everything runs **on CPU** (`JAX_PLATFORMS=cpu`):
GPU XLA reductions are not bitwise reproducible across versions, CPU fp32 is stable to ~1e-6, so
`rtol=1e-5, atol=1e-6` is a real test rather than a flaky one.

Three one-off steps on the machine that has `./checkpoints/` and `./data/datasets/`:

```bash
# 1. fill in the checkpoint directories in cases.yaml (ls checkpoints/*/*/*/seed=0/)
# 2. extract one context window per dataset (self-contained inputs, committed once)
JAX_PLATFORMS=cpu uv run python tests/regression/make_inputs.py --data-root ./data/datasets
# 3. generate the goldens
JAX_PLATFORMS=cpu uv run python tests/regression/make_goldens.py --checkpoints-root ./checkpoints
```

Then, on every machine with the checkpoints:

```bash
JAX_PLATFORMS=cpu uv run pytest tests/regression -m regression --checkpoints-root ./checkpoints
```

Run it **twice** after generating; if the second run differs, the tolerance needs revisiting *now*.
Without `--checkpoints-root` (or `CFNO_CHECKPOINTS`), or without a case's golden file, the tests
*skip* — CI without checkpoints stays green.

Files: `cases.yaml` (one entry per case), `inputs/<name>.npz` (context window + `loss_args`),
`goldens/<name>.npz` (`u_next`, `rollout`, provenance), `make_inputs.py`, `make_goldens.py`,
`conftest.py`, `test_regression.py`. Goldens are **never regenerated** without a change-document
entry saying which commit changed behaviour and why (CLAUDE.md).
