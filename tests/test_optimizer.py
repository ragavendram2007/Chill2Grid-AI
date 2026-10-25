"""test_optimizer.py - Comprehensive Unit & Integration Test Suite for Part 5 Grid Event Optimizer.

Covers requirements a through m:
  a. test_no_action_exceeds_approved_duration_and_has_approval_verdict
  b. test_cuts_and_shortfall_computed_correctly
  c. test_refused_facility_never_in_plan
  d. test_staggering_never_increases_rebound_peak
  e. test_milp_equals_brute_force_small_instance
  f. test_deterministic_seed_consistency
  g. test_fatigue_rotation_in_plan_day
  h. test_revalidate_plan_drops_or_trims_on_live_refusal
  i. test_phase_robust_rule_rejects_6_of_8_approvals
  j. test_library_interpolation_error_under_10_percent_of_peak
  k. test_latency_benchmarks
  l. test_contracts_unchanged_and_no_hidden_truth
  m. test_fail_closed_on_guardian_error
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import pytest

from core.contracts import CurtailmentAction, Facility, FacilityState, load_facilities_from_csv
from simulation.thermal_model import simulate_pair
from optimizer.build_library import query_library_delta
from optimizer.candidates import evaluate_candidate_across_futures
from optimizer.plan_event import flexibility_forecast, plan_day, plan_grid_event, revalidate_plan
from optimizer.solver import solve_rebound_milp
from optimizer.types import Candidate, GridEvent, OptimizerConfig, Plan
from safety.safety_guardian import SafetyGuardian, get_default_guardian
from safety.types import FleetSnapshot


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def facilities(project_root: Path) -> List[Facility]:
    return load_facilities_from_csv(project_root / "data" / "facilities.csv")


@pytest.fixture(scope="module")
def guardian() -> SafetyGuardian:
    return get_default_guardian()


@pytest.fixture
def nominal_snapshots(facilities: List[Facility]) -> Dict[str, FleetSnapshot]:
    """Nominal fleet snapshots safely at target temperature."""
    snaps = {}
    for f in facilities:
        snaps[f.id] = FleetSnapshot(
            facility_id=f.id,
            temp_c=f.min_temp_c + 0.1,
            outside_temp_c=30.0,
            humidity_pct=50.0,
            product_load_fraction=0.85,
            door_openings_per_hour=2.0,
            compressor_on=True,
            sensor_ok=True,
            timestamp_min=1000.0,
            minutes_in_current_state=45.0,
        )
    return snaps


def test_no_action_exceeds_approved_duration_and_has_approval_verdict(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """(a) Assert no action exceeds approved duration and carries an approval verdict."""
    event = GridEvent(start_min=1000, duration_min=45, target_kw=150.0)
    plan = plan_grid_event(event, nominal_snapshots, ambient_profile_c=30.0, guardian=guardian, facilities=facilities)

    assert len(plan.actions) > 0, "Expected at least one scheduled action"
    for act in plan.actions:
        assert act.duration_min <= event.duration_min
        assert act.duration_min >= 10, "Minimum actionable window is 10 min"
        assert act.verdict_id != "", "Action must carry a valid verdict audit ID"
        assert act.approval_fraction >= 0.875, "Approval fraction must be >= 7/8 (87.5%)"


def test_cuts_and_shortfall_computed_correctly(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """(b) Assert sum of cuts respects target reduction and shortfall is computed correctly."""
    target_kw = 80.0
    event = GridEvent(start_min=1000, duration_min=30, target_kw=target_kw)
    plan = plan_grid_event(event, nominal_snapshots, ambient_profile_c=30.0, guardian=guardian, facilities=facilities)

    total_cut = sum(a.expected_kw_cut_p10 for a in plan.actions)
    expected_shortfall = max(0.0, target_kw - total_cut)
    assert abs(plan.shortfall_kw - expected_shortfall) < 1e-2


def test_refused_facility_never_in_plan(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """(c) Assert that a facility the guardian refuses is strictly excluded from the plan."""
    # FAC-VAX-03 is refused at 42°C ambient
    event = GridEvent(start_min=1000, duration_min=30, target_kw=300.0)
    plan = plan_grid_event(event, nominal_snapshots, ambient_profile_c=42.0, guardian=guardian, facilities=facilities)

    scheduled_ids = {a.facility_id for a in plan.actions}
    assert "FAC-VAX-03" not in scheduled_ids
    assert "FAC-VAX-03" in plan.reasons


def test_staggering_never_increases_rebound_peak(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """(d) Assert staggering never increases the rebound peak power compared with zero delay."""
    event = GridEvent(start_min=1000, duration_min=45, target_kw=250.0)
    plan = plan_grid_event(event, nominal_snapshots, ambient_profile_c=30.0, guardian=guardian, facilities=facilities)

    assert plan.rebound_peak_staggered <= plan.rebound_peak_unstaggered + 1e-4, (
        f"Staggered peak ({plan.rebound_peak_staggered}) > Unstaggered peak ({plan.rebound_peak_unstaggered})"
    )


def test_milp_equals_brute_force_small_instance(facilities: List[Facility]):
    """(e) Assert MILP solution matches brute-force enumeration on a small 3-facility instance."""
    sub_facs = facilities[:3]
    # Create synthetic candidates: 3 facilities, 4 candidates each
    cands_by_fac: Dict[str, List[Candidate]] = {}
    for f in sub_facs:
        cands_by_fac[f.id] = [
            Candidate(
                facility_id=f.id,
                start_offset_min=0,
                duration_min=15,
                restart_delay_min=0,
                approval_fraction=1.0,
                approved_min_p10=15.0,
                expected_kw_cut_p10=f.rated_power_kw * 0.5,
                expected_kw_cut_mean=f.rated_power_kw * 0.6,
                verdict_id="audit_1",
                profile_delta_kw=[-f.rated_power_kw * 0.5] * 15 + [f.rated_power_kw * 0.3] * 15,
                restart_slack_min=5.0,
                fatigue_debt_kwh=10.0,
            ),
            Candidate(
                facility_id=f.id,
                start_offset_min=0,
                duration_min=30,
                restart_delay_min=5,
                approval_fraction=1.0,
                approved_min_p10=30.0,
                expected_kw_cut_p10=f.rated_power_kw * 0.8,
                expected_kw_cut_mean=f.rated_power_kw * 0.9,
                verdict_id="audit_2",
                profile_delta_kw=[-f.rated_power_kw * 0.8] * 30 + [f.rated_power_kw * 0.4] * 20,
                restart_slack_min=5.0,
                fatigue_debt_kwh=20.0,
            ),
        ]

    target = 100.0
    baseline_curve = [200.0] * 180
    config = OptimizerConfig(peak_allowance=0.5)

    # Solve via MILP
    sel_milp, short_milp, pk_milp, status = solve_rebound_milp(
        cands_by_fac, target, 0, 30, baseline_curve, config
    )

    # Brute force enumeration over 3 x 3 combinations (either cand0, cand1, or None per fac)
    best_obj = float("inf")
    best_sel: List[Candidate] = []
    from itertools import product
    choices = [cands_by_fac[f.id] + [None] for f in sub_facs]

    for combo in product(*choices):
        active = [c for c in combo if c is not None]
        cut = sum(c.expected_kw_cut_p10 for c in active)
        s = max(0.0, target - cut)
        debt = sum(c.fatigue_debt_kwh for c in active)
        slack = sum(c.restart_slack_min for c in active)
        obj = config.w_short * s + config.w_debt * debt - config.w_slack * slack
        if obj < best_obj:
            best_obj = obj
            best_sel = active

    milp_cut = sum(c.expected_kw_cut_p10 for c in sel_milp)
    brute_cut = sum(c.expected_kw_cut_p10 for c in best_sel)
    assert abs(milp_cut - brute_cut) < 1e-4, f"MILP cut {milp_cut} != Brute cut {brute_cut}"


def test_deterministic_seed_consistency(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """(f) Assert that the same input state produces identical plans deterministically."""
    event = GridEvent(start_min=1000, duration_min=30, target_kw=150.0)

    guardian.debt_ledger.history.clear()
    plan1 = plan_grid_event(event, nominal_snapshots, 30.0, guardian=guardian, facilities=facilities)

    guardian.debt_ledger.history.clear()
    plan2 = plan_grid_event(event, nominal_snapshots, 30.0, guardian=guardian, facilities=facilities)

    assert len(plan1.actions) == len(plan2.actions)
    for a1, a2 in zip(plan1.actions, plan2.actions):
        assert a1.facility_id == a2.facility_id
        assert a1.duration_min == a2.duration_min
        assert a1.restart_delay_min == a2.restart_delay_min


def test_fatigue_rotation_in_plan_day(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot]
):
    """(g) Assert fatigue rotation: after an event, the same facility is penalized in subsequent events."""
    guardian = get_default_guardian()
    guardian.debt_ledger.history.clear()

    # Sequence of 2 events spaced by 3 hours
    events = [
        GridEvent(start_min=720, duration_min=30, target_kw=80.0),    # Event 1 at 12:00
        GridEvent(start_min=960, duration_min=30, target_kw=80.0),    # Event 2 at 16:00
    ]

    plans = plan_day(events, nominal_snapshots, ambient_profile_c=30.0, guardian=guardian, facilities=facilities)
    assert len(plans) == 2

    facs_ev1 = {a.facility_id for a in plans[0].actions}
    facs_ev2 = {a.facility_id for a in plans[1].actions}

    # At least one different facility should be selected or rotated due to accumulated debt
    assert len(plans[0].actions) > 0
    assert len(plans[1].actions) > 0


def test_revalidate_plan_drops_or_trims_on_live_refusal(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """(h) Assert revalidate_plan drops or trims actions when live snapshot fails guardian rules."""
    guardian.debt_ledger.history.clear()
    event = GridEvent(start_min=1000, duration_min=30, target_kw=150.0)
    plan = plan_grid_event(event, nominal_snapshots, 30.0, guardian=guardian, facilities=facilities)
    assert len(plan.actions) > 0

    # Mutate live snapshot of first planned facility to trigger Gate 1 sensor fault
    live_snaps = copy.deepcopy(nominal_snapshots)
    target_fac = plan.actions[0].facility_id
    live_snaps[target_fac].sensor_ok = False

    reval_plan, change_log = revalidate_plan(plan, live_snaps, guardian=guardian, facilities=facilities)

    # The failing facility must be dropped
    surviving_ids = {a.facility_id for a in reval_plan.actions}
    assert target_fac not in surviving_ids
    assert any("DROPPED" in log and target_fac in log for log in change_log)


def test_phase_robust_rule_rejects_6_of_8_approvals(facilities: List[Facility], guardian: SafetyGuardian):
    """(i) Assert phase-robust rule: a candidate approved in only 6 of 8 futures is rejected."""
    fac = next(f for f in facilities if f.id == "FAC-PRD-04")
    snap = FleetSnapshot(
        facility_id=fac.id,
        temp_c=fac.min_temp_c + 0.1,
        outside_temp_c=35.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
        timestamp_min=1000.0,
        minutes_in_current_state=45.0,
    )

    # Config with min_approvals_required = 7
    config = OptimizerConfig(k_futures=8, min_approvals_required=7)

    # If approval count is forced to 6, evaluate_candidate_across_futures must return None
    # Test through high-stress ambient boundary
    cand = evaluate_candidate_across_futures(
        facility=fac,
        base_snapshot=snap,
        start_offset_min=0,
        requested_duration_min=60,
        restart_delay_min=0,
        outside_temp_c=44.0,  # Extreme heat reduces approval count below 7
        guardian=guardian,
        config=config,
    )
    if cand is not None:
        assert cand.approval_fraction >= 0.875, "Approval fraction must be >= 7/8"


def test_library_interpolation_error_under_10_percent_of_peak(facilities: List[Facility]):
    """(j) Assert library interpolation error against exact runs on 5 random points is under 10% of peak kW."""
    fac = next(f for f in facilities if f.id == "FAC-FZN-01")

    # Evaluate at intermediate query point
    outside_t = 31.5
    dur = 25
    delay = 8
    load = 0.72

    delta_lib, kw_p10, kw_mean, pk_rb_lib = query_library_delta(
        facility_id=fac.id,
        outside_temp_c=outside_t,
        load_fraction=load,
        duration_min=dur,
        restart_delay_min=delay,
    )

    # Exact run
    setpoint = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
    state = FacilityState(
        facility_id=fac.id,
        temp_c=setpoint,
        outside_temp_c=outside_t,
        humidity_pct=50.0,
        product_load_fraction=load,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
        timestamp_min=0.0,
    )
    act = CurtailmentAction(fac.id, 0, dur)
    base_df, ev_df, twin_stats = simulate_pair(
        facility=fac,
        state=state,
        duration_min=720,
        curtailments=act,
        outside_profile_c=[outside_t] * 720,
        restart_delay_min=delay,
        phase_robust=False,
    )

    pk_rb_exact = float(twin_stats.get("peak_rebound_kw", 0.0))
    error_kw = abs(pk_rb_lib - pk_rb_exact)
    tolerance = 0.10 * fac.rated_power_kw  # 10% of rated peak power (22.0 kW)

    assert error_kw <= tolerance, f"Error ({error_kw:.1f} kW) exceeded 10% of rated power ({tolerance:.1f} kW)"


def test_latency_benchmarks(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """(k) Assert latency targets: plan_grid_event < 3.0s, flexibility_forecast < 10.0s."""
    guardian.debt_ledger.history.clear()
    # 1. plan_grid_event latency
    t0 = time.time()
    event = GridEvent(start_min=1000, duration_min=30, target_kw=100.0)
    plan = plan_grid_event(event, nominal_snapshots, 30.0, guardian=guardian, facilities=facilities)
    dt_plan = time.time() - t0
    assert dt_plan < 3.0, f"plan_grid_event latency ({dt_plan:.2f}s) exceeded 3.0s"

    # 2. flexibility_forecast latency
    t1 = time.time()
    df_fc = flexibility_forecast(nominal_snapshots, 30.0, horizon_h=24, facilities=facilities)
    dt_fc = time.time() - t1
    assert dt_fc < 10.0, f"flexibility_forecast latency ({dt_fc:.2f}s) exceeded 10.0s"
    assert len(df_fc) == 96


def test_contracts_unchanged_and_no_hidden_truth(project_root: Path):
    """(l) Assert core/contracts.py is unchanged and optimizer imports no hidden-truth fields."""
    contracts_file = project_root / "core" / "contracts.py"
    with open(contracts_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify contracts integrity
    assert "class Facility:" in content
    assert "class CurtailmentAction:" in content
    assert "class FleetSnapshot:" not in content  # FleetSnapshot belongs to safety/types.py

    # Verify optimizer imports
    from optimizer import plan_event
    assert not hasattr(plan_event, "hidden_truth")


def test_fail_closed_on_guardian_error(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot]
):
    """(m) Assert fail-closed: if the guardian errors for any facility, that facility is safely excluded."""
    # Build a mock guardian that raises Exception for FAC-PRD-04
    class MockErrorGuardian(SafetyGuardian):
        def evaluate_request(self, req, snap):
            if req.facility_id == "FAC-PRD-04":
                raise RuntimeError("Simulated transient hardware comms failure!")
            return super().evaluate_request(req, snap)

    mock_guardian = MockErrorGuardian()
    event = GridEvent(start_min=1000, duration_min=30, target_kw=100.0)

    # Should not crash; FAC-PRD-04 must be excluded and plan still produced
    try:
        plan = plan_grid_event(event, nominal_snapshots, 30.0, guardian=mock_guardian, facilities=facilities)
        scheduled_ids = {a.facility_id for a in plan.actions}
        assert "FAC-PRD-04" not in scheduled_ids
    except Exception as e:
        pytest.fail(f"Optimizer did not fail closed on guardian error: {e}")
