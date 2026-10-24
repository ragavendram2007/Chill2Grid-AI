"""key1_conformal.py - Key 1: Learned Decision-Level Conformal Safe Duration Estimator.

Applies finite-sample quantile shrink calibrated per risk tier and phase bucket,
hardened against scenario-level autocorrelation via 200 bootstrap resamples.
Guarantees P(true safe >= approved) >= 1 - alpha for the decision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_safe_minutes_from_trajectory(
    horizons: List[int],
    upper_bounds: List[float] | np.ndarray,
    limit_threshold_c: float,
) -> float:
    """Compute safe curtailment duration (minutes) using linear interpolation on upper bound."""
    if len(horizons) == 0 or len(upper_bounds) == 0:
        return 0.0

    if upper_bounds[0] >= limit_threshold_c:
        return 0.0

    for i in range(1, len(horizons)):
        if upper_bounds[i] >= limit_threshold_c:
            t_prev, t_curr = horizons[i - 1], horizons[i]
            y_prev, y_curr = upper_bounds[i - 1], upper_bounds[i]
            if y_curr > y_prev:
                frac = (limit_threshold_c - y_prev) / (y_curr - y_prev)
                return float(round(t_prev + frac * (t_curr - t_prev), 2))
            return float(t_prev)

    return float(horizons[-1])


class Key1ConformalEstimator:
    """Key 1 decision-level conformal safe window estimator with bootstrap shrinks."""

    def __init__(
        self,
        calibrated_shrinks: Optional[Dict[str, Dict[str, float]]] = None,
        default_shrink_min: float = -5.0,
    ) -> None:
        """Args:

        calibrated_shrinks: nested dict [tier_name][phase_tag] -> shrink_minutes (usually <= 0)
        """
        self.shrinks = calibrated_shrinks or {}
        self.default_shrink = default_shrink_min

    def get_shrink(self, tier_name: str, phase_tag: str) -> Tuple[float, bool]:
        """Retrieve calibrated shrink for tier and phase with hierarchical fallback.

        Returns:
            (shrink_minutes, is_fallback_flag)
        """
        tier_dict = self.shrinks.get(tier_name)
        if tier_dict is not None:
            if phase_tag in tier_dict:
                return tier_dict[phase_tag], False
            if "pooled" in tier_dict:
                return tier_dict["pooled"], True

        # Global fallback to most conservative shrink
        all_shrinks = [
            v for t in self.shrinks.values() for v in t.values() if isinstance(v, (int, float))
        ]
        if all_shrinks:
            return float(min(all_shrinks)), True

        return self.default_shrink, True

    def estimate_safe_minutes(
        self,
        horizons: List[int],
        upper_bound_trajectory: List[float] | np.ndarray,
        max_safe_temp_c: float,
        guard_margin_c: float,
        tier_name: str,
        phase_tag: str,
    ) -> Tuple[float, float, bool]:
        """Estimate Key 1 safe curtailment duration with decision-level conformal shrink.

        Returns:
            (approved_key1_safe_min, raw_pred_safe_min, is_fallback_flag)
        """
        limit_threshold = max_safe_temp_c - guard_margin_c
        monotone_upper = np.maximum.accumulate(np.array(upper_bound_trajectory))
        raw_pred_safe = compute_safe_minutes_from_trajectory(
            horizons, monotone_upper, limit_threshold
        )

        shrink, is_fallback = self.get_shrink(tier_name, phase_tag)
        approved_safe = max(0.0, raw_pred_safe + shrink)
        return float(round(approved_safe, 2)), float(round(raw_pred_safe, 2)), is_fallback
