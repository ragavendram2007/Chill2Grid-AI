"""plan_event.py - Grid Curtailment Event Optimizer.

Module for Phase B:
  1. Candidate generation under strict Safety Guardian evaluation (trying offsets 0, 10, 20, 30m).
  2. Knapsack/greedy allocation targeting requested reduction kW while minimizing rebound peak.
  3. Staggered restart allocation bounded by SafetyVerdict.restart_slack_min.
  4. Pre-output re-validation via SafetyGuardian.evaluate_fleet (fail-closed replanning).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.contracts import CurtailmentAction, Facility, FacilityState
from safety.safety_guardian import SafetyGuardian, get_default_guardian, lookup_max_curtail_min
from safety.types import DecisionEnum, FleetSnapshot, SafetyVerdict
from simulation.thermal_model import cop, simulate_temperature


@dataclass
class GridEventRequest:
    """Request from grid operator or automated DR signal."""
    start_min: int
    duration_min: int
    target_reduction_kw: float
    outside_temp_c: float = 30.0


@dataclass
class FacilityPlan:
    """Scheduled curtailment decision for a specific facility."""
    facility_id: str
    action: CurtailmentAction
    approved_duration_min: float
    start_offset_min: int
    restart_delay_min: int
    expected_kw_cut: float
    verdict: SafetyVerdict


@dataclass
class Plan:
    """Complete, re-validated grid event dispatch plan."""
    facility_plans: List[FacilityPlan]
    expected_kw_cut_curve: List[float]
    rebound_curve_unstaggered: List[float]
    rebound_curve_staggered: List[float]
    rebound_peak_unstaggered: float
    rebound_peak_staggered: float
    total_approved_minutes: float
    total_cut_kw: float
    unmet_target_kw: float
    excluded_reasons: Dict[str, str]
    plan_valid_until: float


def plan_grid_event(
    event: GridEventRequest,
    snapshots: Dict[str, FleetSnapshot],
    facilities: List[Facility],
    guardian: Optional[SafetyGuardian] = None,
) -> Plan:
    """Plan grid curtailment event under strict Fail-Closed Safety Guardian gating."""
    if guardian is None:
        guardian = get_default_guardian()

    fac_map = {f.id: f for f in facilities}
    candidate_plans: List[FacilityPlan] = []
    excluded_reasons: Dict[str, str] = {}

    # Save debt ledger state so candidate exploration does not mutate active history
    saved_history = {k: list(v) for k, v in guardian.debt_ledger.history.items()}

    # 1. Candidate Generation across facilities and start offsets
    for fac_id, snap in snapshots.items():
        fac = fac_map.get(fac_id)
        if not fac:
            continue

        best_approved_dur = 0.0
        best_verdict: Optional[SafetyVerdict] = None
        best_offset = 0
        best_action: Optional[CurtailmentAction] = None

        # Fast pre-filter
        max_lookup = lookup_max_curtail_min(fac.id, snap.temp_c, snap.outside_temp_c)

        # Offsets: 0, 10, 20, 30 min (to clear G2 equipment lockouts if deferred)
        offsets_to_try = [0, 10, 20, 30]

        for offset in offsets_to_try:
            # Restore clean history before evaluating hypothetical offset
            guardian.debt_ledger.history = {k: list(v) for k, v in saved_history.items()}

            req_start = int(event.start_min + offset)
            req_dur = int(max(10, event.duration_min - offset))
            if req_dur < 10:
                continue

            action = CurtailmentAction(
                facility_id=fac_id,
                start_min=req_start,
                duration_min=req_dur,
            )

            # Shift snapshot timestamp for offset evaluation
            shifted_snap = FleetSnapshot(
                facility_id=snap.facility_id,
                temp_c=snap.temp_c,
                outside_temp_c=snap.outside_temp_c,
                humidity_pct=snap.humidity_pct,
                product_load_fraction=snap.product_load_fraction,
                door_openings_per_hour=snap.door_openings_per_hour,
                compressor_on=snap.compressor_on,
                sensor_ok=snap.sensor_ok,
                timestamp_min=snap.timestamp_min + float(offset),
                minutes_in_current_state=snap.minutes_in_current_state + float(offset),
            )

            verdict = guardian.evaluate_request(action, shifted_snap)

            if verdict.is_approved and verdict.approved_duration_min > best_approved_dur:
                best_approved_dur = verdict.approved_duration_min
                best_verdict = verdict
                best_offset = offset
                best_action = CurtailmentAction(
                    facility_id=fac_id,
                    start_min=req_start,
                    duration_min=int(verdict.approved_duration_min),
                )

        if best_verdict and best_action and best_approved_dur >= 10.0:
            candidate_plans.append(
                FacilityPlan(
                    facility_id=fac_id,
                    action=best_action,
                    approved_duration_min=best_approved_dur,
                    start_offset_min=best_offset,
                    restart_delay_min=0,
                    expected_kw_cut=fac.rated_power_kw,
                    verdict=best_verdict,
                )
            )
        else:
            # Document refusal reason
            refusal_verdict = guardian.evaluate_request(
                CurtailmentAction(fac_id, int(event.start_min), int(event.duration_min)), snap
            )
            excluded_reasons[fac_id] = f"{refusal_verdict.binding_constraint}: {refusal_verdict.plain_English}"

    # 2. Objective & Knapsack Selection
    # Greedy by efficiency: rated_kw / (thermal debt / headroom)
    def priority_score(cand: FacilityPlan) -> float:
        fac = fac_map[cand.facility_id]
        snap = snapshots[cand.facility_id]
        headroom = max(0.5, fac.max_safe_temp_c - snap.temp_c)
        return float(cand.expected_kw_cut * cand.approved_duration_min / headroom)

    candidate_plans.sort(key=priority_score, reverse=True)

    selected_plans: List[FacilityPlan] = []
    current_cut_kw = 0.0

    for cand in candidate_plans:
        # Check if target is already satisfied
        if current_cut_kw >= event.target_reduction_kw and len(selected_plans) > 0:
            excluded_reasons[cand.facility_id] = "TARGET_SATISFIED: Fleet target capacity already achieved"
            continue

        selected_plans.append(cand)
        current_cut_kw += cand.expected_kw_cut

    # 3. Staggered Recovery Optimization
    # Assign restart_delay_min sequentially to smooth recovery peak
    stagger_step = 5  # 5-minute stagger steps
    for idx, plan_item in enumerate(selected_plans):
        max_slack = int(max(0.0, plan_item.verdict.restart_slack_min))
        # Stagger later facilities if slack permits
        desired_stagger = min(idx * stagger_step, max_slack)
        plan_item.restart_delay_min = desired_stagger

    # 5. Re-validation Gate via evaluate_fleet
    # Restore pre-evaluation ledger history before fleet evaluation
    guardian.debt_ledger.history = {k: list(v) for k, v in saved_history.items()}
    final_actions = [p.action for p in selected_plans]
    final_snaps = [snapshots[p.facility_id] for p in selected_plans]
    fleet_verdicts = guardian.evaluate_fleet(final_actions, final_snaps)

    valid_plans: List[FacilityPlan] = []
    for plan_item, f_verdict in zip(selected_plans, fleet_verdicts):
        if not f_verdict.is_approved:
            excluded_reasons[plan_item.facility_id] = f"REVALIDATION_FAILED: {f_verdict.plain_English}"
        else:
            valid_plans.append(plan_item)

    # 4. Construct Power Curves & Rebound Analytics
    horizon_min = int(event.start_min + event.duration_min + 120)
    kw_cut_curve = [0.0] * horizon_min
    rebound_unstaggered = [0.0] * horizon_min
    rebound_staggered = [0.0] * horizon_min

    for plan_item in valid_plans:
        act = plan_item.action
        fac = fac_map[plan_item.facility_id]
        p_cut = plan_item.expected_kw_cut

        # Curtailment cut window
        for m in range(act.start_min, min(horizon_min, act.start_min + act.duration_min)):
            kw_cut_curve[m] += p_cut

        # Recovery pulse (unstaggered: immediate restart)
        rec_start_un = act.start_min + act.duration_min
        rec_dur = max(15, int(act.duration_min * 0.75))
        for m in range(rec_start_un, min(horizon_min, rec_start_un + rec_dur)):
            rebound_unstaggered[m] += p_cut * 1.15  # pull-down compressor run

        # Recovery pulse (staggered: shifted by restart_delay_min)
        rec_start_st = rec_start_un + plan_item.restart_delay_min
        for m in range(rec_start_st, min(horizon_min, rec_start_st + rec_dur)):
            rebound_staggered[m] += p_cut * 1.15

    peak_unstaggered = max(rebound_unstaggered) if rebound_unstaggered else 0.0
    peak_staggered = max(rebound_staggered) if rebound_staggered else 0.0

    # Ensure staggering property: staggering never increases rebound peak
    if peak_staggered > peak_unstaggered:
        peak_staggered = peak_unstaggered
        rebound_staggered = list(rebound_unstaggered)
        for p in valid_plans:
            p.restart_delay_min = 0

    total_cut_kw = sum(p.expected_kw_cut for p in valid_plans)
    total_app_min = sum(p.approved_duration_min for p in valid_plans)
    unmet_kw = max(0.0, event.target_reduction_kw - total_cut_kw)
    plan_lease = min((p.verdict.valid_until_min for p in valid_plans), default=event.start_min + 30.0)

    return Plan(
        facility_plans=valid_plans,
        expected_kw_cut_curve=kw_cut_curve,
        rebound_curve_unstaggered=rebound_unstaggered,
        rebound_curve_staggered=rebound_staggered,
        rebound_peak_unstaggered=peak_unstaggered,
        rebound_peak_staggered=peak_staggered,
        total_approved_minutes=total_app_min,
        total_cut_kw=total_cut_kw,
        unmet_target_kw=unmet_kw,
        excluded_reasons=excluded_reasons,
        plan_valid_until=plan_lease,
    )
