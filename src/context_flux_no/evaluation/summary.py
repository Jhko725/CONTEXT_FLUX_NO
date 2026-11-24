"""Host-side reductions of the per-step arrays that `rollout` returns.

`summarize_rollouts` and `summarize_conservation_drift` reduce the diagnostics of one
model to a few numbers. `match_trajectories` pairs the trajectories of two test sets, and
`mean_by_bin` averages segment errors over groups of trajectories. Every function takes
and returns NumPy arrays.
"""

from typing import Any

import numpy as np
from jaxtyping import Bool, Float, Int

from .metrics import first_true_step, RolloutReferences, valid_horizon


def _fraction_by_step(
    flags: Bool[np.ndarray, " *batch steps"], steps: tuple[int, ...]
) -> dict[int, float]:
    """Return the fraction of rows that are true at any step up to each of `steps`."""
    first = first_true_step(flags)
    return {step: float(np.mean(first <= step)) for step in steps}


def _median_over_finite(
    values: Float[np.ndarray, " *batch steps"],
    nonfinite: Bool[np.ndarray, " *batch steps"],
    steps: tuple[int, ...],
) -> dict[int, float]:
    """Return the median of `values` at each of `steps` over the finite rollouts."""
    medians = {}
    for step in steps:
        column = values[..., step - 1][~nonfinite[..., step - 1]]
        column = column[np.isfinite(column)]
        medians[step] = float(np.median(column)) if column.size else float("nan")
    return medians


def summarize_rollouts(
    l2: Float[np.ndarray, " *batch trajectories steps"],
    nonfinite: Bool[np.ndarray, " *batch trajectories steps"],
    area_fraction: Float[np.ndarray, " *batch trajectories steps thresholds"],
    total_variation: Float[np.ndarray, " *batch trajectories steps"],
    references: RolloutReferences,
    *,
    exclude_constant_references: bool,
    minima: Float[np.ndarray, " *batch trajectories steps quantities"] | None = None,
    horizon: int = 80,
    report_steps: tuple[int, ...] = (20, 40, 80),
) -> dict[str, Any]:
    """Reduce the per-step diagnostics of one model to its failure and error summary.

    The leading `*batch` axes, such as the seeds of one model, are pooled with the
    trajectories. Only the first `horizon` steps are used.

    A trajectory has a constant reference when the total variation of its reference is
    zero at every step of the horizon. This happens for a scalar law whose initial
    condition is a constant state, such as a periodic step function with one jump.
    Both baselines then have zero error and the reference has zero range, so every
    model error counts as a crossing of the identity baseline at step 1 and as
    diverged, the area fraction is 1 wherever the prediction differs, and the
    total-variation ratio is infinite.

    **Arguments:**

    - `l2`, `nonfinite`, `area_fraction`, `total_variation`: The fields of
        `RolloutDiagnostics`.
    - `references`: The reference quantities of the same trajectories.
    - `exclude_constant_references`: Whether the trajectories with a constant
        reference are left out of every number of the summary.
    - `minima`: The minima of the quantities that must stay positive, or `None`.
    - `horizon`: The number of rollout steps that enter the summary.
    - `report_steps`: The steps at which the failure fractions, the area fraction and
        the total-variation ratio are reported.

    **Returns:**

    A dictionary with the following entries.

    - `num_rollouts`: The number of rollouts that enter the summary.
    - `num_constant_references`: The number of trajectories with a constant reference,
        whether or not they are left out.
    - `valid_horizon_quartiles`: The 25th, 50th and 75th percentile of the valid
        horizon, where a rollout that stays below the identity baseline counts as
        `horizon`.
    - `fraction_never_crossed`: The fraction of rollouts that stay below the identity
        baseline.
    - `fraction_crossed_at_step_1`: The fraction that is no better than the identity
        baseline at the first step.
    - `fraction_nonfinite`, `fraction_diverged`, `fraction_nonpositive`: The fraction
        of rollouts affected at any step up to each of `report_steps`. A rollout has
        diverged at a step where its error exceeds that of the spatial-mean baseline or
        is non-finite. `fraction_nonpositive` is `None` without `minima`.
    - `median_area_fraction`: The median error area fraction per threshold, over the
        rollouts that are finite at the step.
    - `median_total_variation_ratio`: The median of the total variation of the
        prediction over that of the reference, over the same rollouts.

    **Raises:**

    `ValueError` if the rollouts are shorter than `horizon`, or a step of
    `report_steps` exceeds it.
    """
    num_steps = l2.shape[-1]
    if num_steps < horizon:
        raise ValueError(f"The rollouts have {num_steps} steps, fewer than {horizon}.")
    if any(step > horizon for step in report_steps):
        raise ValueError(f"report_steps={report_steps} exceed the horizon {horizon}.")

    is_constant = np.all(references.total_variation[..., :horizon] == 0.0, axis=-1)
    keep = ~is_constant if exclude_constant_references else np.ones_like(is_constant)

    l2 = l2[..., keep, :horizon]
    nonfinite = nonfinite[..., keep, :horizon]
    area_fraction = area_fraction[..., keep, :, :]
    l2_identity = references.l2_identity[keep, :horizon]
    l2_spatial_mean = references.l2_spatial_mean[keep, :horizon]

    horizons = valid_horizon(l2, l2_identity, nonfinite)
    with np.errstate(invalid="ignore", divide="ignore"):
        diverged = (l2 > l2_spatial_mean) | nonfinite | ~np.isfinite(l2)
        ratio = (
            total_variation[..., keep, :horizon]
            / references.total_variation[keep, :horizon]
        )
    if minima is not None:
        minima = minima[..., keep, :, :]

    summary = {
        "num_rollouts": int(horizons.size),
        "num_constant_references": int(is_constant.sum()),
        "valid_horizon_quartiles": [
            float(q) for q in np.percentile(np.minimum(horizons, horizon), [25, 50, 75])
        ],
        "fraction_never_crossed": float(np.mean(horizons > horizon)),
        "fraction_crossed_at_step_1": float(np.mean(horizons == 1)),
        "fraction_nonfinite": _fraction_by_step(nonfinite, report_steps),
        "fraction_diverged": _fraction_by_step(diverged, report_steps),
        "fraction_nonpositive": None,
        "median_area_fraction": [
            _median_over_finite(
                area_fraction[..., :horizon, i], nonfinite, report_steps
            )
            for i in range(area_fraction.shape[-1])
        ],
        "median_total_variation_ratio": _median_over_finite(
            ratio, nonfinite, report_steps
        ),
    }
    if minima is not None:
        with np.errstate(invalid="ignore"):
            nonpositive = np.any(minima[..., :horizon, :] <= 0.0, axis=-1)
        summary["fraction_nonpositive"] = _fraction_by_step(nonpositive, report_steps)
    return summary


def summarize_conservation_drift(
    conservation_drift: Float[np.ndarray, " *batch trajectories steps channels"],
    *,
    horizon: int = 80,
) -> dict[str, Any]:
    """Reduce the conservation drift of one model to three numbers per channel.

    Each rollout gives three numbers: the mean of its drift over the steps, its drift
    at the last step, and its largest drift. The summary of each is the median over the
    rollouts with the 25th and 75th percentiles. The leading `*batch` axes, such as the
    seeds of a model, are pooled with the trajectories. The drift of a diverging
    rollout grows without bound, so a mean over rollouts would be set by a few of them.

    A rollout whose drift is non-finite at any step of the horizon is left out, and the
    fraction that remains is returned.

    **Arguments:**

    - `conservation_drift`: The field of `RolloutDiagnostics` or `RolloutReferences`
        of that name.
    - `horizon`: The number of rollout steps that enter the summary.

    **Returns:**

    A dictionary with the following entries. The first three have the shape
    `(3, channels)` and hold the 25th percentile, the median and the 75th percentile
    over the rollouts.

    - `mean`: The mean of the drift of a rollout over its steps.
    - `final`: The drift of a rollout at the last step.
    - `worst`: The largest drift of a rollout.
    - `fraction_finite`: The fraction of rollouts that enter the summary, per channel.

    **Raises:**

    `ValueError` if the rollouts are shorter than `horizon`.
    """
    num_steps, num_channels = conservation_drift.shape[-2:]
    if num_steps < horizon:
        raise ValueError(f"The rollouts have {num_steps} steps, fewer than {horizon}.")
    drift = conservation_drift[..., :horizon, :].reshape(-1, horizon, num_channels)
    is_finite = np.all(np.isfinite(drift), axis=1)

    def quartiles(
        per_rollout: Float[np.ndarray, " rollouts channels"],
    ) -> Float[np.ndarray, " 3 channels"]:
        result = np.full((3, num_channels), np.nan)
        for channel in range(num_channels):
            values = per_rollout[is_finite[:, channel], channel]
            if values.size:
                result[:, channel] = np.percentile(values, [25, 50, 75])
        return result

    with np.errstate(invalid="ignore"):
        return {
            "mean": quartiles(drift.mean(axis=1)),
            "final": quartiles(drift[:, -1]),
            "worst": quartiles(drift.max(axis=1)),
            "fraction_finite": is_finite.mean(axis=0),
        }


def match_trajectories(
    fields: Float[np.ndarray, " n *shape"],
    coefficients: Float[np.ndarray, " n params"],
    partner_fields: Float[np.ndarray, " m *shape"],
    partner_coefficients: Float[np.ndarray, " m params"],
    *,
    atol: float,
) -> Int[np.ndarray, " n"]:
    """Find, for each trajectory of one test set, its partner in a paired test set.

    Two trajectories are partners when their coefficients are equal and their `fields`
    agree within `atol` in every entry. `fields` is a quantity that pairing leaves
    unchanged, such as the initial momentum of two shallow-water sets that differ in
    the initial depth only.

    **Arguments:**

    - `fields`: The paired quantity of each trajectory of the first set.
    - `coefficients`: The coefficients of each trajectory of the first set.
    - `partner_fields`: The paired quantity of each trajectory of the second set.
    - `partner_coefficients`: The coefficients of each trajectory of the second set.
    - `atol`: The largest difference in any entry of `fields` between partners.

    **Returns:**

    The index of the partner of each trajectory, or -1 for a trajectory without one.

    **Raises:**

    `ValueError` if two trajectories share a partner, because the sets are then not
    paired one to one.
    """
    by_coefficients: dict[bytes, list[int]] = {}
    for index, row in enumerate(np.ascontiguousarray(partner_coefficients)):
        by_coefficients.setdefault(row.tobytes(), []).append(index)

    partners = np.full(len(fields), -1, dtype=np.int64)
    for index, row in enumerate(np.ascontiguousarray(coefficients)):
        candidates = np.asarray(by_coefficients.get(row.tobytes(), []), dtype=np.int64)
        if candidates.size == 0:
            continue
        distance = np.abs(partner_fields[candidates] - fields[index]).reshape(
            len(candidates), -1
        )
        distance = distance.max(axis=1)
        if distance.min() <= atol:
            partners[index] = candidates[distance.argmin()]

    matched = partners[partners >= 0]
    if len(np.unique(matched)) != len(matched):
        raise ValueError("Two trajectories share a partner; the sets are not paired.")
    return partners


def mean_by_bin(
    values: Float[np.ndarray, " *batch n"],
    bin_indices: Int[np.ndarray, " n"],
    num_bins: int,
) -> tuple[Float[np.ndarray, " *batch {num_bins}"], Int[np.ndarray, " {num_bins}"]]:
    """Average `values` over the entries of each bin.

    **Arguments:**

    - `values`: The values, with the binned axis last.
    - `bin_indices`: The bin of each entry of that axis. An entry with an index outside
        0 to `num_bins - 1` belongs to no bin.
    - `num_bins`: The number of bins.

    **Returns:**

    The mean per bin, which is NaN for an empty bin, and the number of entries per bin.
    A non-finite value makes the mean of its bin non-finite.
    """
    in_range = (bin_indices >= 0) & (bin_indices < num_bins)
    counts = np.bincount(bin_indices[in_range], minlength=num_bins)
    one_hot = bin_indices[:, None] == np.arange(num_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (
            np.stack(
                [values[..., one_hot[:, b]].sum(axis=-1) for b in range(num_bins)],
                axis=-1,
            )
            / counts
        )
    return means, counts
