"""Checks of the host-side reductions in `context_flux_no.evaluation.summary`.

Every expected value comes from a case that is listed by hand.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from context_flux_no.evaluation.metrics import RolloutReferences
from context_flux_no.evaluation.summary import (
    match_trajectories,
    mean_by_bin,
    summarize_conservation_drift,
    summarize_rollouts,
)
from jaxtyping import Float


def test_match_trajectories_finds_partners_in_a_shuffled_subset():
    """Partners are found in a shuffled subset whose fields differ below the tolerance.

    Four trajectories lie in two coefficient groups, and the partner set holds three of
    them in another order. Trajectory 1 has no partner, and trajectories 0 and 2 have
    equal fields and are told apart by their coefficients.
    """
    fields = np.asarray([[0.0, 1.0], [2.0, 3.0], [0.0, 1.0], [5.0, 6.0]])
    coefficients = np.asarray([[1.0], [1.0], [2.0], [2.0]])
    order = [2, 0, 3]
    partner_fields = fields[order] + 1e-9
    partners = match_trajectories(
        fields, coefficients, partner_fields, coefficients[order], atol=1e-6
    )
    assert partners.tolist() == [1, -1, 0, 2]


def test_match_trajectories_rejects_a_shared_partner():
    """Two trajectories with the same partner raise."""
    fields = np.zeros((2, 3))
    coefficients = np.ones((2, 1))
    with pytest.raises(ValueError, match="share a partner"):
        match_trajectories(fields, coefficients, fields[:1], coefficients[:1], atol=0.0)


def test_mean_by_bin():
    """Each bin averages its entries; an entry outside the bins is dropped, and an empty
    bin has a NaN mean."""
    values = np.asarray([[1.0, 3.0, 10.0, 7.0, 100.0], [2.0, 4.0, 20.0, 8.0, 200.0]])
    means, counts = mean_by_bin(values, np.asarray([0, 0, 1, 3, -1]), num_bins=4)
    assert counts.tolist() == [2, 1, 0, 1]
    assert np.allclose(means[:, [0, 1, 3]], [[2.0, 10.0, 7.0], [3.0, 20.0, 8.0]])
    assert np.isnan(means[:, 2]).all()


def _three_references() -> RolloutReferences:
    """The references of three rollouts of four steps, the first of them constant.

    Each rollout's references are built alone, as `rollout_references` builds them, and
    stacked along a leading rollout axis, as a vmap over the rollouts stacks them.
    """
    identity = (0.1, 0.2, 0.3, 0.4)
    rollouts = [
        RolloutReferences(
            l2_identity=jnp.asarray(l2_identity),
            l2_spatial_mean=jnp.full(4, l2_spatial_mean),
            total_variation=jnp.full(4, total_variation),
            conservation_drift=jnp.zeros((4, 1)),
        )
        for l2_identity, l2_spatial_mean, total_variation in zip(
            ((0.0,) * 4, identity, identity), (0.0, 1.0, 1.0), (0.0, 2.0, 4.0)
        )
    ]
    return jax.tree.map(lambda *leaves: np.stack(leaves), *rollouts)


def _three_l2_errors() -> Float[np.ndarray, " 3 4"]:
    """The model errors of the three rollouts of `_three_references`.

    Rollout 0 has a constant reference, so its error of 0.1 is a crossing at step 1.
    Rollout 1 reaches the identity-baseline error at step 3, and rollout 2 stays below
    it.
    """
    return np.array(
        [[0.1, 0.1, 0.1, 0.1], [0.05, 0.1, 0.35, 0.5], [0.01, 0.02, 0.03, 0.04]]
    )


def _summary_of_three_rollouts(
    exclude_constant_references: bool,
    minima: Float[np.ndarray, " 3 4 quantities"] | None = None,
) -> dict[str, Any]:
    """Summarize the three rollouts of `_three_l2_errors` with area fractions 1, 0.25
    and 0.5 and total-variation ratios infinite, 0.5 and 1."""
    area_fraction = np.stack(
        (np.ones((4, 1)), np.full((4, 1), 0.25), np.full((4, 1), 0.5))
    )
    total_variation = np.stack((np.full(4, 0.2), np.full(4, 1.0), np.full(4, 4.0)))
    return summarize_rollouts(
        _three_l2_errors(),
        np.zeros((3, 4), dtype=bool),
        area_fraction,
        total_variation,
        _three_references(),
        exclude_constant_references=exclude_constant_references,
        minima=minima,
        horizon=4,
        report_steps=(4,),
    )


def test_summary_counts_a_constant_reference_as_a_failure_at_step_1():
    """Kept in, the constant reference crosses and diverges at step 1, and its infinite
    total-variation ratio is left out of the median."""
    summary = _summary_of_three_rollouts(exclude_constant_references=False)

    assert summary["num_rollouts"] == 3
    assert summary["num_constant_references"] == 1
    # Valid horizons 1, 3 and 4.
    np.testing.assert_allclose(summary["valid_horizon_quartiles"], [2.0, 3.0, 3.5])
    assert summary["fraction_crossed_at_step_1"] == pytest.approx(1 / 3)
    assert summary["fraction_never_crossed"] == pytest.approx(1 / 3)
    assert summary["fraction_diverged"][4] == pytest.approx(1 / 3)
    assert summary["fraction_nonpositive"] is None
    # Area fractions 1, 0.25 and 0.5.
    assert summary["median_area_fraction"][0][4] == pytest.approx(0.5)
    # The median of the finite ratios 0.5 and 1.
    assert summary["median_total_variation_ratio"][4] == pytest.approx(0.75)


def test_summary_without_constant_references():
    """Left out, the constant reference enters no number but its count."""
    summary = _summary_of_three_rollouts(exclude_constant_references=True)

    assert summary["num_rollouts"] == 2
    assert summary["num_constant_references"] == 1
    # Valid horizons 3 and 4.
    np.testing.assert_allclose(summary["valid_horizon_quartiles"], [3.25, 3.5, 3.75])
    assert summary["fraction_crossed_at_step_1"] == 0.0
    assert summary["fraction_never_crossed"] == pytest.approx(0.5)
    assert summary["fraction_diverged"][4] == 0.0
    assert summary["median_area_fraction"][0][4] == pytest.approx(0.375)
    assert summary["median_total_variation_ratio"][4] == pytest.approx(0.75)


def test_summary_counts_a_nonpositive_minimum():
    """A rollout whose minimum of one quantity reaches zero counts as nonpositive.

    Of the two rollouts that remain, rollout 2 reaches a zero minimum of its second
    quantity at step 3.
    """
    minima = np.ones((3, 4, 2))
    minima[2, 2, 1] = 0.0
    summary = _summary_of_three_rollouts(
        exclude_constant_references=True, minima=minima
    )
    assert summary["fraction_nonpositive"] == {4: pytest.approx(0.5)}


def test_summary_pools_seeds_after_the_exclusion():
    """With a leading seed axis, every seed loses the same trajectory."""
    l2 = _three_l2_errors()
    two_seeds = summarize_rollouts(
        np.stack((l2, l2)),
        np.zeros((2, 3, 4), dtype=bool),
        np.ones((2, 3, 4, 1)),
        np.ones((2, 3, 4)),
        _three_references(),
        exclude_constant_references=True,
        horizon=4,
        report_steps=(4,),
    )

    assert two_seeds["num_rollouts"] == 4
    assert two_seeds["num_constant_references"] == 1
    # Valid horizons 3, 4, 3 and 4.
    np.testing.assert_allclose(two_seeds["valid_horizon_quartiles"], [3.0, 3.5, 4.0])
    assert two_seeds["fraction_never_crossed"] == pytest.approx(0.5)


def test_conservation_drift_summary_of_pooled_rollouts():
    """Summarize the drift of two seeds with three rollouts of three steps each.

    The last rollout turns non-finite at its last step and is left out. The other five
    are pooled, and one of them drifts a thousand times more than the rest.
    """
    drift = np.array(
        [
            [[0.0, 0.2, 0.1], [0.3, 0.1, 0.2], [0.1, 0.1, 0.4]],
            [[0.2, 0.4, 0.3], [100.0, 300.0, 500.0], [0.1, 0.1, np.inf]],
        ]
    )[..., None]
    summary = summarize_conservation_drift(drift, horizon=3)

    # Per rollout, in order: time means 0.1, 0.2, 0.2, 0.3, 300; last steps 0.1, 0.2,
    # 0.4, 0.3, 500; largest values 0.2, 0.3, 0.4, 0.4, 500. The quartiles of five
    # sorted values are the second, third and fourth.
    np.testing.assert_allclose(summary["mean"][:, 0], [0.2, 0.2, 0.3])
    np.testing.assert_allclose(summary["final"][:, 0], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(summary["worst"][:, 0], [0.3, 0.4, 0.4])
    np.testing.assert_allclose(summary["fraction_finite"], [5 / 6])
    # A shorter horizon ends before the non-finite step, so all six enter. Their last
    # steps are 0.1, 0.1, 0.1, 0.2, 0.4 and 300, with the median between 0.1 and 0.2.
    shorter = summarize_conservation_drift(drift, horizon=2)
    np.testing.assert_allclose(shorter["final"][1], [0.15])
    np.testing.assert_allclose(shorter["fraction_finite"], [1.0])


def test_conservation_drift_summary_without_a_finite_rollout():
    """Without a finite rollout, every quartile is NaN and the finite fraction zero."""
    drift = np.full((2, 3, 1), np.inf)
    summary = summarize_conservation_drift(drift, horizon=3)
    assert np.all(np.isnan(summary["mean"]))
    np.testing.assert_allclose(summary["fraction_finite"], [0.0])
