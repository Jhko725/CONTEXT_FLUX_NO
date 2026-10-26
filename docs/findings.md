---
type: findings
status: current
updated: 2026-10-24
sources: [code, claude, joon]
---

# Findings ledger

Append-only. Anything discovered while working that bears on a *reported result* (paper tables,
datasets, claims) or constrains a later phase. Each row names its evidence and the action taken.
Items that are only cleanup go in the roadmap, not here.

| # | Date | Finding | Evidence | Affects | Action / status |
|---|---|---|---|---|---|
| F1 | 2026-09-26 | 10 of 12 model configs spell `hypernet_init: bias_hyperinit`; `HypernetworkHead` matches only `"bias-hyperinit"` and otherwise falls through to default init. Runs launched from those configs most likely did **not** use Bias-HyperInit although the config says so. | `scripts/training/configs/model/*.yaml`; `src/context_flux_no/nn/hypernetwork.py:95` | every 1D result; paper appendix on initialisation | Phase 0.6a fixes spelling, adds runtime check, makes `bias-hyperinit` the default (D15). Phase 0.7 audits wandb runs → conclusion to be appended here and to the post-submission plan. **2026-10-26:** the 1D ICLR checkpoints store the misspelled value in their orbax config, so the 0.6a guard refused to rebuild them; `training/io.py::translate_legacy_config` reads it as `default` (what it meant at training time) when loading — the weights overwrite the initialization, so no forward pass changes. |
| F2 | 2026-09-26 | The FV update `u − dt·ΔF/dx` exists in 8 copies with 4 face-alignment conventions (`jnp.diff` on N+1 faces; pad-diff-slice in `lskflux`; `f − roll(f, 1)` in `convnext`; concatenate + `dynamic_slice` in `fluxfno.py`). | assessment §4d | any cross-target comparison in the paper | Phase 2g migrates each onto `fv/`; any numerical mismatch is appended here as its own row. |
| F3 | 2026-09-26 | `nn/operators/fourier.py` does `rfftn(v, axes=(-1,))` then `irfftn(out, s=n_grids)` over all spatial axes — a 1-D forward with an N-D inverse; the file's own TODO at :97 flags it. | `fourier.py:97–109` | any 2D use of `NaiveFNO` / `FNOTargetNetwork` | Deferred: fixed with a test when the FNO baselines are touched (Phase 2g/4.8). Until then 2D FNO-based results are suspect. |
| F4 | 2026-09-26 | `stack_grid=False` is ignored by `HyperFluxFNOLocal` and `ContextAppendedFluxNO` (grid always appended); `FluxModel` hard-codes `in_channels + num_spatial_dims`. | `hyperfluxfno.py:637, :841, :601, :811` | the `stack_grid` ablation, if reported | Resolved structurally in Phase 3 (`DataSpec` + ADR-0003 makes coordinate channels an explicit ablation). |
| F5 | 2026-09-26 | DISCO `split(key, 3)` then `keys[3]` (clamped → key reused); `HyperNeuralOperator` uses `keys[3]` for both `project_operator` and `hypernetwork_head`. | `disco.py:121/167`; `hyperfluxfno.py:395/424` | initialisation RNG of DISCO and HNO trainings; not their loaded forward pass | **Fixed 2026-10-24** (`phase1-prune`, commit 5); guarded by `tests/unit/test_key_hygiene.py`. Bit-parity of fresh trainings with ICLR runs was already lost at D15. |
| F6 | 2026-09-26 | UNet blocks in `disco/operatornet.py` and `target_networks/unet.py` are byte-identical and share bugs: `keys_bot[0]` reused for both bottleneck convs; `add_boundary_mask` allocates a channel never filled. | assessment §4d | DISCO and UNet-target results | Key reuse **fixed 2026-10-24** (`phase1-prune`, commit 5; found by the hygiene test). Unused mask channel and the duplication itself: Phase 2g/3. |
| F7 | 2026-09-26 | Evaluation metrics are defined per notebook (`compute_metrics` ×7, `metrics_long_rollout` ×4) and may differ; checkpoint selection is by hand-maintained timestamp dictionaries with no run-id link. | assessment §4f, P9 | reproducibility of every table; suspected copy-paste in Table 16 | Phase 5 (run identity) and Phase 6 (one `evaluation/`); Table 16 to be re-derived from `runs.index()`. |
| F8 | 2026-09-26 | `ViscousBurgers1D.solve` never writes the last frame; uses left-edge `x` while everything else is cell-centred; `CubicFlux2D` data is first-order Rusanov. | assessment §4e | Burgers and 2D-cubic datasets | Phase 4 regenerates these families with the reference backends. |
| F9 | 2026-10-24 | `jax.flatten_util` was never imported explicitly; on JAX 0.9 `HypernetworkHead` and DISCO's head raise `AttributeError` when a model is constructed outside a training script (where optax/equinox had imported it first). | `nn/hypernetwork.py:97`, `disco/disco.py:53` | standalone model construction, e.g. `load_model` in the regression tier | Fixed (`phase1-prune`, commit 0). Latent breakage, not result-affecting. |
