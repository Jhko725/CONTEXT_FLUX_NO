"""Evaluation of trained models: the error metrics and rollout diagnostics of
`metrics`, the rollouts of `rollout` on the accelerator, their host-side reductions in
`summary`, the compute cost of `cost`, and the `EvaluationRecord` of `record` that
`scripts/evaluate.py` writes beside a result."""

from .cost import (
    count_flops as count_flops,
    measure as measure,
    Measurement as Measurement,
)
from .metrics import (
    conservation_drift as conservation_drift,
    count_jumps as count_jumps,
    error_area_fraction as error_area_fraction,
    error_power_spectrum as error_power_spectrum,
    first_true_step as first_true_step,
    l2_norm as l2_norm,
    l_infty_norm as l_infty_norm,
    largest_jump as largest_jump,
    num_wavenumber_bins as num_wavenumber_bins,
    radial_power_spectrum as radial_power_spectrum,
    relative_l2_error as relative_l2_error,
    relative_l2_error_per_channel as relative_l2_error_per_channel,
    relative_l_infty_error as relative_l_infty_error,
    rollout_diagnostics as rollout_diagnostics,
    rollout_references as rollout_references,
    RolloutDiagnostics as RolloutDiagnostics,
    RolloutReferences as RolloutReferences,
    segment_errors as segment_errors,
    SegmentErrors as SegmentErrors,
    spatial_mean_field as spatial_mean_field,
    total_variation as total_variation,
    valid_horizon as valid_horizon,
)
from .record import EvaluationRecord as EvaluationRecord
from .rollout import (
    evaluate_references as evaluate_references,
    evaluate_rollouts as evaluate_rollouts,
    evaluate_segment_baselines as evaluate_segment_baselines,
    evaluate_segments as evaluate_segments,
    Operator as Operator,
    PositiveQuantities as PositiveQuantities,
    RolloutEvaluation as RolloutEvaluation,
    SegmentBaselines as SegmentBaselines,
)
from .summary import (
    match_trajectories as match_trajectories,
    mean_by_bin as mean_by_bin,
    summarize_conservation_drift as summarize_conservation_drift,
    summarize_rollouts as summarize_rollouts,
)
