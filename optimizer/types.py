"""types.py - Public Dataclasses and Configuration for Part 5 Grid Event Optimizer.

Contains:
  - GridEvent: Specification of target demand reduction event.
  - Candidate: A proposed (facility, start_offset, duration, restart_delay) combination.
  - PlannedAction: Scheduled curtailment action approved by Guardian.
  - Plan: Full fleet dispatch plan with baseline, event, and rebound power curves.
  - OptimizerConfig: Documented objective weights and constraint allowances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GridEvent:
    """Specification of a grid curtailment request."""
    start_min: int
    duration_min: int
    target_kw: float
    price_series: Optional[List[float]] = None
    emission_series: Optional[List[float]] = None


@dataclass
class Candidate:
    """A feasible candidate curtailment option for one facility evaluated across K futures."""
    facility_id: str
    start_offset_min: int
    duration_min: int
    restart_delay_min: int
    approval_fraction: float
    approved_min_p10: float
    expected_kw_cut_p10: float
    expected_kw_cut_mean: float
    verdict_id: str
    profile_delta_kw: List[float] = field(default_factory=list)
    restart_slack_min: float = 0.0
    fatigue_debt_kwh: float = 0.0


@dataclass
class PlannedAction:
    """Action scheduled for a specific facility in the final plan."""
    facility_id: str
    start_min: int
    duration_min: int
    restart_delay_min: int
    expected_kw_cut_p10: float
    expected_kw_cut_mean: float
    verdict_id: str
    approval_fraction: float
    reason: str = ""


@dataclass
class Plan:
    """Complete, verified grid event plan."""
    actions: List[PlannedAction]
    fleet_baseline_power: List[float]
    event_power_curve: List[float]
    rebound_curve: List[float]
    rebound_peak_staggered: float
    rebound_peak_unstaggered: float
    shortfall_kw: float
    binding_constraints: Dict[str, str]
    reasons: Dict[str, str]
    valid_until_min: float
    solver_status: str
    library_vs_exact_error: Dict[str, float]
    summary: str = ""


@dataclass
class OptimizerConfig:
    """Documented objective weights and optimization parameters."""
    w_short: float = 100.0       # Weight on capacity shortfall below target kW
    w_peak: float = 10.0         # Weight on rebound peak power
    w_debt: float = 1.0          # Weight on accumulated thermal fatigue debt
    w_slack: float = 0.1         # Bonus weight for utilizing guardian restart slack
    peak_allowance: float = 0.0  # Fraction allowed above baseline peak (0.0 = strict 'no new peak')
    k_futures: int = 8           # Number of thermostat phase futures sampled
    min_approvals_required: int = 7  # Must approve in at least 7 of 8 futures (87.5%)
    replan_threshold_pct: float = 0.10  # Trigger single replan if exact rebound > library by >10%
