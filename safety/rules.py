"""rules.py - Safety Guardian Risk Tiers, Margins, and Operational Limits.

Rules Version: 1.0.0
All limits and margins are labeled simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

RULES_VERSION: str = "1.0.0"


@dataclass(frozen=True)
class RiskTierConfig:
    """Configuration parameters for a Safety Guardian risk tier."""

    tier_name: str
    target_facilities: List[str]
    confidence_level: float
    guard_margin_c: float
    abort_margin_c: float
    max_single_curtailment_min: float
    restart_reserve_min: float
    rolling_24h_budget_min: float
    min_gap_between_events_min: float
    min_actionable_window_min: float = 10.0
    duty_cycle_max_threshold: float = 0.90
    max_recovery_min: float = 180.0
    tripwire_lockout_ambient_c: Optional[float] = None
    calibrated_lambda: float = 1.0
    restart_lag_min: float = 2.0


TIER_CRITICAL = RiskTierConfig(
    tier_name="CRITICAL",
    target_facilities=["FAC-VAX-03"],
    confidence_level=0.99,
    guard_margin_c=0.70,
    abort_margin_c=0.50,
    max_single_curtailment_min=20.0,
    restart_reserve_min=10.0,
    rolling_24h_budget_min=60.0,
    min_gap_between_events_min=120.0,
    min_actionable_window_min=8.0,
    duty_cycle_max_threshold=0.90,
    max_recovery_min=180.0,
    tripwire_lockout_ambient_c=35.0,  # Prohibit curtailment when outdoor temp >= 35 C
    calibrated_lambda=0.80,
    restart_lag_min=3.0,
)

TIER_HIGH = RiskTierConfig(
    tier_name="HIGH",
    target_facilities=["FAC-ICE-02", "FAC-SEA-05", "FAC-MLK-06"],
    confidence_level=0.95,
    guard_margin_c=0.30,
    abort_margin_c=0.20,
    max_single_curtailment_min=45.0,
    restart_reserve_min=5.0,
    rolling_24h_budget_min=120.0,
    min_gap_between_events_min=60.0,
    min_actionable_window_min=10.0,
    duty_cycle_max_threshold=0.90,
    max_recovery_min=180.0,
    tripwire_lockout_ambient_c=42.0,
    calibrated_lambda=0.40,
    restart_lag_min=2.0,
)

TIER_STANDARD = RiskTierConfig(
    tier_name="STANDARD",
    target_facilities=["FAC-FZN-01", "FAC-PRD-04"],
    confidence_level=0.90,
    guard_margin_c=0.20,
    abort_margin_c=0.15,
    max_single_curtailment_min=60.0,
    restart_reserve_min=5.0,
    rolling_24h_budget_min=180.0,
    min_gap_between_events_min=45.0,
    min_actionable_window_min=10.0,
    duty_cycle_max_threshold=0.90,
    max_recovery_min=180.0,
    tripwire_lockout_ambient_c=None,
    calibrated_lambda=0.35,
    restart_lag_min=2.0,
)

RISK_TIERS: Dict[str, RiskTierConfig] = {
    "CRITICAL": TIER_CRITICAL,
    "HIGH": TIER_HIGH,
    "STANDARD": TIER_STANDARD,
}

# Mapping by Facility ID because core.contracts.Facility does not have a risk_tier field
FACILITY_TIER_MAP: Dict[str, str] = {
    "FAC-VAX-03": "CRITICAL",
    "FAC-ICE-02": "HIGH",
    "FAC-SEA-05": "HIGH",
    "FAC-MLK-06": "HIGH",
    "FAC-FZN-01": "STANDARD",
    "FAC-PRD-04": "STANDARD",
}


def get_facility_tier(facility_id: str) -> RiskTierConfig:
    """Retrieve risk tier configuration for a given facility ID."""
    tier_name = FACILITY_TIER_MAP.get(facility_id, "STANDARD")
    return RISK_TIERS[tier_name]
