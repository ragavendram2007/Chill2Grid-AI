"""debt_ledger.py - Thermal Debt Ledger, Inter-Event Cooldown, and 24h Budget Enforcer.

Tracks:
  1. Minimum cooldown gap between curtailment events per risk tier
  2. Rolling 24-hour curtailment duration budget per facility
  3. Exponentially decaying thermal debt using physical twin time constant:
     tau_twin = R_wall * C_cargo * 60 (minutes)
     Debt(t) = sum_i [ Q_i * exp(-(t - t_end) / tau_twin) ]
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

from core.contracts import Facility
from safety.rules import RiskTierConfig, get_facility_tier


@dataclass
class CurtailmentRecord:
    """Historical record of an executed demand response curtailment."""

    start_min: float
    end_min: float
    duration_min: float
    heat_injected_kwh: float


class ThermalDebtLedger:
    """Per-facility rolling budget and thermal debt accounting ledger."""

    def __init__(self) -> None:
        self.history: Dict[str, List[CurtailmentRecord]] = {}

    def reset(self) -> None:
        """Clear all recorded curtailment events."""
        self.history.clear()

    def record_event(
        self,
        facility_id: str,
        start_min: float,
        duration_min: float,
        heat_injected_kwh: float = 0.0,
    ) -> None:
        """Record an executed curtailment in the ledger."""
        if facility_id not in self.history:
            self.history[facility_id] = []
        rec = CurtailmentRecord(
            start_min=start_min,
            end_min=start_min + duration_min,
            duration_min=duration_min,
            heat_injected_kwh=heat_injected_kwh,
        )
        self.history[facility_id].append(rec)

    def get_time_since_last_event(self, facility_id: str, current_time_min: float) -> float:
        """Return elapsed minutes since last curtailment event ended, or infinity if none."""
        events = self.history.get(facility_id, [])
        if not events:
            return float("inf")
        last_end = max(e.end_min for e in events)
        return max(0.0, current_time_min - last_end)

    def get_rolling_24h_curtailed_min(self, facility_id: str, current_time_min: float) -> float:
        """Compute total minutes curtailed within the trailing 24 hours (1440 minutes)."""
        events = self.history.get(facility_id, [])
        cutoff = current_time_min - 1440.0
        tot_min = 0.0
        for e in events:
            if e.end_min > cutoff:
                # Overlap with [cutoff, current_time_min]
                ev_start_clipped = max(cutoff, e.start_min)
                ev_end_clipped = min(current_time_min, e.end_min)
                if ev_end_clipped > ev_start_clipped:
                    tot_min += (ev_end_clipped - ev_start_clipped)
        return float(tot_min)

    def compute_decaying_thermal_debt(
        self,
        facility: Facility,
        current_time_min: float,
    ) -> float:
        """Compute un-dissipated thermal debt using exponential decay:

        Formula:
          tau = R_wall * C_cargo * 60 (minutes)
          Debt(t) = sum_i [ Q_i * exp(-(t - t_end) / tau) ] in kWh thermal.
        """
        events = self.history.get(facility.id, [])
        if not events:
            return 0.0

        tau_min = facility.insulation_r_deg_c_per_kw * facility.thermal_mass_kwh_per_c * 60.0
        tau_min = max(60.0, tau_min)

        tot_debt = 0.0
        for e in events:
            if current_time_min >= e.end_min:
                dt = current_time_min - e.end_min
                q_rem = e.heat_injected_kwh * math.exp(-dt / tau_min)
                tot_debt += q_rem
        return float(tot_debt)

    def evaluate_gate5(
        self,
        facility: Facility,
        tier: RiskTierConfig,
        current_time_min: float,
        proposed_duration_min: float,
    ) -> Tuple[bool, float, float, str]:
        """Evaluate Gate 5 thermal debt ledger rules.

        Returns:
            (allowed, approved_duration, wait_min, reason)
        """
        # 1. Minimum Inter-Event Gap Check
        time_since = self.get_time_since_last_event(facility.id, current_time_min)
        if time_since < tier.min_gap_between_events_min:
            wait_min = tier.min_gap_between_events_min - time_since
            return (
                False,
                0.0,
                math.ceil(wait_min),
                f"THERMAL_DEBT_COOLDOWN: Minimum inter-event gap active ({time_since:.0f}m < {tier.min_gap_between_events_min:.0f}m tier requirement, wait {math.ceil(wait_min)}m)",
            )

        # 2. Rolling 24h Budget Check
        used_24h = self.get_rolling_24h_curtailed_min(facility.id, current_time_min)
        budget_rem = max(0.0, tier.rolling_24h_budget_min - used_24h)
        if budget_rem < 10.0:
            return (
                False,
                0.0,
                0.0,
                f"BUDGET_EXHAUSTED: 24h curtailment budget exhausted ({used_24h:.0f}m used of {tier.rolling_24h_budget_min:.0f}m limit)",
            )

        allowed_dur = min(proposed_duration_min, budget_rem)
        return True, allowed_dur, 0.0, f"PASS: Cooldown elapsed ({time_since:.0f}m), budget remaining {budget_rem:.0f}m"
