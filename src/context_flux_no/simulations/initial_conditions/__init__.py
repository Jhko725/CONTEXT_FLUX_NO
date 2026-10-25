"""Initial-condition samplers for data generation (moved from `waveforms/`, 2026-10-25):
Gaussian random fields, periodic random step functions, truncated Fourier series. Each has
`sample(centers, key) -> [n_eqn, *N]`. The `InitialCondition` contract of roadmap 4.2
(`sample(key, grid, layout, bcs)`, valid for the family for any seed) replaces the per-sampler
signatures when tier 0 lands."""
