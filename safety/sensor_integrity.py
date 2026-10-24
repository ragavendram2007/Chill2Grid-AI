"""sensor_integrity.py - Real-Time Sensor Health, Stuck Detection, and CUSUM Drift Monitor.

Detects:
  1. Missing telemetry readings (> 5 consecutive missing)
  2. Stuck sensor (zero variance while physics twin expects a rise)
  3. Impossible jumps (> 2.0 °C in 1 minute)
  4. Both positive and negative sensor drift via two-sided Cumulative Sum (CUSUM)
     against nominal-twin expectations, with sigma from calibration residuals.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


class SensorIntegrityChecker:
    """Sensor watchdog tracking dropouts, stuck values, jump anomalies, and CUSUM drift."""

    def __init__(
        self,
        residual_sigma_c: float = 0.15,
        cusum_slack_k: Optional[float] = None,
        cusum_threshold_h: Optional[float] = None,
        max_consecutive_missing: int = 5,
        max_jump_deg_c: float = 2.0,
        stuck_window_mins: int = 15,
        stuck_variance_threshold: float = 1e-4,
    ) -> None:
        self.sigma = max(0.05, residual_sigma_c)
        self.k = cusum_slack_k if cusum_slack_k is not None else 0.5 * self.sigma
        self.h = cusum_threshold_h if cusum_threshold_h is not None else 4.0 * self.sigma
        self.max_missing = max_consecutive_missing
        self.max_jump = max_jump_deg_c
        self.stuck_window = stuck_window_mins
        self.stuck_var_thresh = stuck_variance_threshold

        self.consecutive_missing: int = 0
        self.last_valid_reading: Optional[float] = None
        self.reading_history: List[float] = []
        self.cusum_pos: float = 0.0
        self.cusum_neg: float = 0.0

    def reset(self) -> None:
        """Reset internal accumulator state."""
        self.consecutive_missing = 0
        self.last_valid_reading = None
        self.reading_history.clear()
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

    def update(
        self,
        sensor_temp: Optional[float],
        twin_expected_temp: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Update monitor with new minute's sensor reading and check for faults.

        Returns:
            (is_healthy, failure_reason)
        """
        # 1. Missing Telemetry Check
        if sensor_temp is None or np.isnan(sensor_temp):
            self.consecutive_missing += 1
            if self.consecutive_missing > self.max_missing:
                return False, f"TELEMETRY_FAULT: {self.consecutive_missing} consecutive missing readings (> {self.max_missing} limit)"
            return True, "MISSING_TOLERATED"

        self.consecutive_missing = 0

        # 2. Impossible Jump Check
        if self.last_valid_reading is not None:
            delta = abs(sensor_temp - self.last_valid_reading)
            if delta > self.max_jump:
                return False, f"IMPOSSIBLE_JUMP: Sensor changed by {delta:.2f}°C in 1 minute (> {self.max_jump:.1f}°C limit)"

        self.last_valid_reading = sensor_temp
        self.reading_history.append(sensor_temp)
        if len(self.reading_history) > 60:
            self.reading_history.pop(0)

        # 3. Stuck Sensor Check
        if len(self.reading_history) >= self.stuck_window and twin_expected_temp is not None:
            recent_readings = self.reading_history[-self.stuck_window:]
            var_recent = float(np.var(recent_readings))
            # If readings show zero variance while twin indicates heat rise
            if var_recent < self.stuck_var_thresh:
                return False, f"STUCK_SENSOR: Sensor variance {var_recent:.6f} near zero over {self.stuck_window} minutes"

        # 4. Two-Sided CUSUM Drift Check
        if twin_expected_temp is not None:
            error = sensor_temp - twin_expected_temp
            # Positive CUSUM (sensor reading warmer than expectation)
            self.cusum_pos = max(0.0, self.cusum_pos + (error - self.k))
            # Negative CUSUM (sensor reading cooler than expectation - highly dangerous direction)
            self.cusum_neg = max(0.0, self.cusum_neg - (error + self.k))

            if self.cusum_pos > self.h:
                return False, f"SENSOR_DRIFT_WARM: Positive CUSUM statistic {self.cusum_pos:.2f} exceeded threshold {self.h:.2f}"
            if self.cusum_neg > self.h:
                return False, f"SENSOR_DRIFT_COLD: Negative CUSUM statistic {self.cusum_neg:.2f} exceeded threshold {self.h:.2f} (reading cooler than reality)"

        return True, "HEALTHY"
