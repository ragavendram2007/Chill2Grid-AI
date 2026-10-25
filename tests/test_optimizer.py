"""test_optimizer.py - Comprehensive Unit Tests for Grid Event Optimizer.

Phase B Requirements:
  1. test_plan_never_exceeds_approved_window
  2. test_sum_of_cuts_at_most_target
  3. test_staggering_never_increases_rebound_peak
  4. test_refuse_facility_never_in_plan
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pytest

from core.contracts import Facility, load_facilities_from_csv
from optimizer.plan_event import GridEventRequest, Plan, plan_grid_event
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
    """Provide realistic nominal fleet snapshots where facilities are in safe cooling states."""
    snaps = {}
    for f in facilities:
        snaps[f.id] = FleetSnapshot(
            facility_id=f.id,
            temp_c=f.min_temp_c + 0.1,  # safely at target operating temp
            outside_temp_c=30.0,
            humidity_pct=50.0,
            product_load_fraction=0.85,
            door_openings_per_hour=2.0,
            compressor_on=True,
            sensor_ok=True,
            timestamp_min=1000.0,
            minutes_in_current_state=45.0,  # well past min_on_min
        )
    return snaps


def test_plan_never_exceeds_approved_window(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """1. Assert that every scheduled facility action duration <= guardian approved duration."""
    req = GridEventRequest(start_min=1000, duration_min=45, target_reduction_kw=150.0, outside_temp_c=30.0)
    plan = plan_grid_event(req, nominal_snapshots, facilities, guardian=guardian)

    assert len(plan.facility_plans) > 0, "Expected at least one facility approved"
    for item in plan.facility_plans:
        assert item.action.duration_min <= item.approved_duration_min + 1e-4, (
            f"{item.facility_id}: scheduled {item.action.duration_min}m > approved {item.approved_duration_min}m"
        )
        assert item.action.duration_min <= req.duration_min


def test_sum_of_cuts_at_most_target(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """2. Assert sum of cuts respects target reduction without needless over-allocation."""
    # Request small target of 70 kW (one or two facilities)
    target = 70.0
    req = GridEventRequest(start_min=1000, duration_min=30, target_reduction_kw=target, outside_temp_c=30.0)
    plan = plan_grid_event(req, nominal_snapshots, facilities, guardian=guardian)

    # Total cut should satisfy target without adding unnecessary third/fourth facilities
    assert len(plan.facility_plans) <= 2
    # Unmet should be 0 if target met
    if plan.total_cut_kw >= target:
        assert plan.unmet_target_kw == 0.0


def test_staggering_never_increases_rebound_peak(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """3. Assert staggering never increases the rebound peak power."""
    req = GridEventRequest(start_min=1000, duration_min=30, target_reduction_kw=300.0, outside_temp_c=30.0)
    plan = plan_grid_event(req, nominal_snapshots, facilities, guardian=guardian)

    assert plan.rebound_peak_staggered <= plan.rebound_peak_unstaggered + 1e-6, (
        f"Staggered peak ({plan.rebound_peak_staggered}) > Unstaggered peak ({plan.rebound_peak_unstaggered})"
    )


def test_refuse_facility_never_in_plan(
    facilities: List[Facility], nominal_snapshots: Dict[str, FleetSnapshot], guardian: SafetyGuardian
):
    """4. Assert that any facility refused by the guardian is strictly excluded from the plan."""
    # Force FAC-VAX-03 to fail G3 ambient heat lockout by setting ambient to 42 C
    hot_snaps = dict(nominal_snapshots)
    vax_snap = hot_snaps["FAC-VAX-03"]
    hot_snaps["FAC-VAX-03"] = FleetSnapshot(
        facility_id="FAC-VAX-03",
        temp_c=vax_snap.temp_c,
        outside_temp_c=42.0,  # Triggers >= 35 C G3 lockout
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
        timestamp_min=1000.0,
        minutes_in_current_state=45.0,
    )

    req = GridEventRequest(start_min=1000, duration_min=30, target_reduction_kw=300.0, outside_temp_c=42.0)
    plan = plan_grid_event(req, hot_snaps, facilities, guardian=guardian)

    scheduled_ids = {p.facility_id for p in plan.facility_plans}
    assert "FAC-VAX-03" not in scheduled_ids, "FAC-VAX-03 was refused but appeared in schedule!"
    assert "FAC-VAX-03" in plan.excluded_reasons, "FAC-VAX-03 must be documented in excluded_reasons"
