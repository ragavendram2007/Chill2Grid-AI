"""ood.py - Out-Of-Distribution (OOD) Guard for Safety Guardian.

Monitors input feature bounds against metadata training ranges and checks
for excessive model disagreement (|hybrid mean - physics| at 30 and 60 min).
Never extrapolates silently: forces fallback to Key 2 with an extra 20% margin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np


class OODGuard:
    """Out-of-Distribution detector based on training feature bounds and model disagreement."""

    def __init__(
        self,
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        disagreement_threshold_30m: float = 1.85,
        disagreement_threshold_60m: float = 2.95,
        metadata_path: Optional[Path] = None,
    ) -> None:
        self.feature_bounds = feature_bounds or {}
        self.disagreement_threshold_30m = disagreement_threshold_30m
        self.disagreement_threshold_60m = disagreement_threshold_60m

        if metadata_path is not None and metadata_path.is_file():
            self._load_from_metadata(metadata_path)

    def _load_from_metadata(self, metadata_path: Path) -> None:
        """Load feature ranges and thresholds from model metadata."""
        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
            if "feature_bounds" in data:
                self.feature_bounds = {k: tuple(v) for k, v in data["feature_bounds"].items()}
            t_dict = data.get("ood_disagreement_thresholds") or data.get("ood_thresholds")
            if t_dict:
                self.disagreement_threshold_30m = float(t_dict.get("disagreement_30m", self.disagreement_threshold_30m))
                self.disagreement_threshold_60m = float(t_dict.get("disagreement_60m", self.disagreement_threshold_60m))
        except Exception:
            pass

    def check_features(self, features: Dict[str, float]) -> Tuple[bool, str]:
        """Check if any input observable feature is outside training domain bounds."""
        for feat_name, val in features.items():
            if feat_name in self.feature_bounds:
                low, high = self.feature_bounds[feat_name]
                if val < low - 1e-4 or val > high + 1e-4:
                    return True, f"Feature '{feat_name}'={val:.2f} out of training range [{low:.2f}, {high:.2f}]"
        return False, "Features within training bounds"

    def check_model_disagreement(
        self,
        hybrid_at_30: float,
        physics_at_30: float,
        hybrid_at_60: float,
        physics_at_60: float,
    ) -> Tuple[bool, str]:
        """Check if hybrid vs physics disagreement exceeds 99th percentile calibration threshold."""
        diff_30 = abs(hybrid_at_30 - physics_at_30)
        diff_60 = abs(hybrid_at_60 - physics_at_60)

        if diff_30 > self.disagreement_threshold_30m:
            return True, f"High model disagreement at 30m ({diff_30:.2f}°C > 99th pct {self.disagreement_threshold_30m:.2f}°C)"
        if diff_60 > self.disagreement_threshold_60m:
            return True, f"High model disagreement at 60m ({diff_60:.2f}°C > 99th pct {self.disagreement_threshold_60m:.2f}°C)"

        return False, "Model agreement acceptable"

    def is_ood(
        self,
        features: Dict[str, float],
        hybrid_at_30: Optional[float] = None,
        physics_at_30: Optional[float] = None,
        hybrid_at_60: Optional[float] = None,
        physics_at_60: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Combined OOD test on features and forecast divergence."""
        feat_ood, feat_msg = self.check_features(features)
        if feat_ood:
            return True, f"OOD: {feat_msg}"

        if (
            hybrid_at_30 is not None
            and physics_at_30 is not None
            and hybrid_at_60 is not None
            and physics_at_60 is not None
        ):
            dis_ood, dis_msg = self.check_model_disagreement(
                hybrid_at_30, physics_at_30, hybrid_at_60, physics_at_60
            )
            if dis_ood:
                return True, f"OOD: {dis_msg}"

        return False, "IN_DISTRIBUTION"
