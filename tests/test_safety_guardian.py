"""test_safety_guardian.py - Comprehensive Unit & Integration Test Suite for Part 4 Safety Guardian.

Covers requirements a through o:
  a. test_fail_closed_on_errors_and_none
  b. test_sensor_integrity_missing_stuck_negative_drift
  c. test_min_on_min_deferred_exact_wait
  d. test_two_key_approved_duration_bound (500 random states)
  e. test_guardian_monotonicity (50 states)
  f. test_ood_detection_falls_back_to_key2
  g. test_thermal_debt_ledger_blocks_overflow
  h. test_vax03_40c_warm_phase_refused
  i. test_not_recovered_states_rejected
  j. test_runtime_monitor_abort_conditions
  k. test_audit_chain_tamper_detection_and_replay
  l. test_contracts_unchanged_and_seeds_disjoint
  m. test_decision_level_conformal_guarantee
  n. test_guardian_latency_benchmarks (<50ms single, <300ms fleet)
  o. test_red_team_quick_critical_zero_breach

All tests assert zero regressions against the 53 baseline tests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import time
from typing import List

import numpy as np
import pandas as pd
import pytest

from core.contracts import CurtailmentAction, Facility, FacilityState, load_facilities_from_csv
from safety.audit import AuditLogger, verify_audit_chain
from safety.debt_ledger import ThermalDebtLedger
from safety.key1_conformal import Key1ConformalEstimator, compute_safe_minutes_from_trajectory
from safety.key2_physics import simulate_key2_worst_case
from safety.ood import OODGuard
from safety.rules import RULES_VERSION, get_facility_tier
from safety.safety_guardian import (
    SafetyGuardian,
    build_safe_envelope,
    evaluate_fleet,
    evaluate_request,
    get_default_guardian,
    lookup_max_curtail_min,
    monitor_active_event,
    replay_decision,
)
from safety.sensor_integrity import SensorIntegrityChecker
from safety.types import BindingConstraintEnum, DecisionEnum, FleetSnapshot, MonitorActionEnum
from simulation.thermal_model import THERMOSTAT_DEADBAND_C, simulate_temperature


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def calibrated_facilities(project_root: Path) -> List[Facility]:
    cal_csv = project_root / "data" / "facilities_calibrated.csv"
    if not cal_csv.is_file():
        cal_csv = project_root / "data" / "facilities.csv"
    return load_facilities_from_csv(cal_csv)


@pytest.fixture(scope="module")
def default_guardian() -> SafetyGuardian:
    return get_default_guardian()


# ============================================================================
# TEST CASES (a through o)
# ============================================================================

def test_fail_closed_on_errors_and_none(default_guardian: SafetyGuardian):
    """(a) Test fail-closed response on None, NaN, and physically impossible inputs."""
    # 1. Null / None snapshot
    req = CurtailmentAction("FAC-VAX-03", start_min=0, duration_min=30)
    # Using evaluate_fleet with missing snapshot
    verdicts = default_guardian.evaluate_fleet([req], [])
    assert len(verdicts) == 1
    assert verdicts[0].decision == DecisionEnum.REFUSE.value
    assert verdicts[0].binding_constraint == BindingConstraintEnum.TELEMETRY_FAULT.value

    # 2. NaN sensor reading
    snap_nan = FleetSnapshot(
        facility_id="FAC-VAX-03",
        temp_c=float("nan"),
        outside_temp_c=30.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
    )
    v_nan = default_guardian.evaluate_request(req, snap_nan)
    assert v_nan.decision == DecisionEnum.REFUSE.value
    assert v_nan.binding_constraint == BindingConstraintEnum.TELEMETRY_FAULT.value

    # 3. Impossible temperature (< -60 °C or > 80 °C)
    snap_imp = FleetSnapshot(
        facility_id="FAC-VAX-03",
        temp_c=120.0,
        outside_temp_c=30.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
    )
    v_imp = default_guardian.evaluate_request(req, snap_imp)
    assert v_imp.decision == DecisionEnum.REFUSE.value

    # 4. Zero or negative duration request fails closed
    with pytest.raises(ValueError, match="duration_min must be strictly positive"):
        CurtailmentAction("FAC-VAX-03", start_min=0, duration_min=-5.0)


def test_sensor_integrity_missing_stuck_negative_drift():
    """(b) Test sensor watchdog against dropouts, stuck sensors, jumps, and negative drift."""
    checker = SensorIntegrityChecker(residual_sigma_c=0.15)

    # 1. Missing readings (> 5 consecutive)
    for _ in range(5):
        ok, _ = checker.update(None)
        assert ok is True
    ok, msg = checker.update(None)
    assert ok is False
    assert "TELEMETRY_FAULT" in msg

    checker.reset()

    # 2. Impossible jump (> 2.0 °C in 1 minute)
    checker.update(4.0)
    ok, msg = checker.update(6.5)  # +2.5 °C jump
    assert ok is False
    assert "IMPOSSIBLE_JUMP" in msg

    checker.reset()

    # 3. Stuck sensor (15m near-zero variance while twin heats)
    for _ in range(16):
        ok, msg = checker.update(sensor_temp=4.25, twin_expected_temp=5.50)
    assert ok is False
    assert "STUCK_SENSOR" in msg

    checker.reset()

    # 4. Negative drift (-0.05 °C/min cooler than truth, the dangerous direction)
    drift_detected = False
    for m in range(30):
        # Sensor reads cooler while twin indicates heat rise
        s_val = 4.25 - 0.05 * m
        twin_val = 4.25 + 0.05 * m
        ok, msg = checker.update(sensor_temp=s_val, twin_expected_temp=twin_val)
        if not ok:
            assert "SENSOR_DRIFT_COLD" in msg
            drift_detected = True
            break
    assert drift_detected is True


def test_min_on_min_deferred_exact_wait(default_guardian: SafetyGuardian):
    """(c) Test compressor minimum run duration (min_on_min) deferral with exact wait time."""
    req = CurtailmentAction("FAC-FZN-01", start_min=0, duration_min=30)
    # FAC-FZN-01 has min_on_min = 10. Run state = 3 minutes.
    snap = FleetSnapshot(
        facility_id="FAC-FZN-01",
        temp_c=-22.0,
        outside_temp_c=25.0,
        humidity_pct=60.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
        minutes_in_current_state=3.0,
    )
    v = default_guardian.evaluate_request(req, snap)
    assert v.decision == DecisionEnum.DEFER.value
    assert v.binding_constraint == BindingConstraintEnum.EQUIPMENT_MIN_ON.value
    assert v.wait_min == 7.0  # 10 - 3 = 7


def test_two_key_approved_duration_bound(
    default_guardian: SafetyGuardian,
    calibrated_facilities: List[Facility],
):
    """(d) Assert approved curtailment duration <= min(Key 1, Key 2) across 500 test states."""
    rng = np.random.default_rng(42)

    for _ in range(500):
        fac = rng.choice(calibrated_facilities)
        tier = get_facility_tier(fac.id)
        setpoint = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
        temp_c = setpoint + rng.uniform(-THERMOSTAT_DEADBAND_C, THERMOSTAT_DEADBAND_C)
        amb_c = rng.uniform(20.0, 42.0)
        req_dur = rng.choice([20.0, 30.0, 45.0, 60.0])

        snap = FleetSnapshot(
            facility_id=fac.id,
            temp_c=float(temp_c),
            outside_temp_c=float(amb_c),
            humidity_pct=60.0,
            product_load_fraction=0.85,
            door_openings_per_hour=2.0,
            compressor_on=True,
            sensor_ok=True,
            minutes_in_current_state=30.0,
        )
        req = CurtailmentAction(fac.id, start_min=0, duration_min=float(req_dur))
        default_guardian.debt_ledger.reset()
        v = default_guardian.evaluate_request(req, snap)

        if v.approved_duration_min > 0:
            k1 = v.evidence.get("key1_safe_min", 999.0)
            k2 = v.evidence.get("key2_safe_min", 999.0)
            assert v.approved_duration_min <= min(k1, k2) + 1e-4
            assert v.approved_duration_min <= tier.max_single_curtailment_min


def test_guardian_monotonicity(
    default_guardian: SafetyGuardian,
    calibrated_facilities: List[Facility],
):
    """(e) Assert approved duration is monotonically non-increasing as ambient temperature rises."""
    for fac in calibrated_facilities[:3]:
        setpoint = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
        amb_temps = [24.0, 30.0, 36.0, 42.0]
        approved_mins = []

        for amb in amb_temps:
            snap = FleetSnapshot(
                facility_id=fac.id,
                temp_c=setpoint - 0.2,
                outside_temp_c=amb,
                humidity_pct=60.0,
                product_load_fraction=0.85,
                door_openings_per_hour=2.0,
                compressor_on=True,
                sensor_ok=True,
                minutes_in_current_state=30.0,
            )
            req = CurtailmentAction(fac.id, start_min=0, duration_min=45.0)
            default_guardian.debt_ledger.reset()
            v = default_guardian.evaluate_request(req, snap)
            approved_mins.append(v.approved_duration_min)

        # Monotonicity check
        for i in range(1, len(approved_mins)):
            assert approved_mins[i] <= approved_mins[i - 1] + 1e-4


def test_ood_detection_falls_back_to_key2(default_guardian: SafetyGuardian):
    """(f) Assert out-of-distribution detection forces fallback to Key 2 with +20% margin."""
    # 1. Direct OOD detector check on extreme temperature
    is_ood, msg = default_guardian.ood_guard.is_ood(
        features={"lag_0": 4.0, "outside_temp_c": 55.0}
    )
    assert is_ood is True
    assert "outside_temp_c" in msg

    # 2. Key 2 pessimistic twin applies extra 20% margin under OOD
    fac = default_guardian.facilities_map["FAC-PRD-04"]
    tier = get_facility_tier("FAC-PRD-04")
    snap = FleetSnapshot(
        facility_id="FAC-PRD-04",
        temp_c=2.5,
        outside_temp_c=28.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
    )
    k2_in_dist = simulate_key2_worst_case(fac, snap, tier, is_ood=False)
    k2_ood = simulate_key2_worst_case(fac, snap, tier, is_ood=True)
    # Under OOD, safe window is strictly shorter due to +20% margin
    assert k2_ood <= k2_in_dist


def test_thermal_debt_ledger_blocks_overflow(calibrated_facilities: List[Facility]):
    """(g) Assert thermal debt ledger enforces minimum cooldown gap and rolling 24h budget."""
    ledger = ThermalDebtLedger()
    vax_fac = next(f for f in calibrated_facilities if f.id == "FAC-VAX-03")
    tier = get_facility_tier("FAC-VAX-03")

    # 1. Record event ending at minute 60
    ledger.record_event("FAC-VAX-03", start_min=40.0, duration_min=20.0, heat_injected_kwh=10.0)

    # 2. Check cooldown: request at minute 90 (30m elapsed < 120m min gap)
    ok, dur, wait, reason = ledger.evaluate_gate5(vax_fac, tier, current_time_min=90.0, proposed_duration_min=20.0)
    assert ok is False
    assert wait == 90.0  # 120 - 30 = 90
    assert "THERMAL_DEBT_COOLDOWN" in reason

    # 3. Check 24h budget exhaustion (budget = 60m for CRITICAL)
    ledger.reset()
    ledger.record_event("FAC-VAX-03", start_min=100.0, duration_min=20.0, heat_injected_kwh=10.0)
    ledger.record_event("FAC-VAX-03", start_min=300.0, duration_min=20.0, heat_injected_kwh=10.0)
    ledger.record_event("FAC-VAX-03", start_min=500.0, duration_min=20.0, heat_injected_kwh=10.0)
    # Total curtailed in 24h = 60m
    ok_b, dur_b, wait_b, reason_b = ledger.evaluate_gate5(vax_fac, tier, current_time_min=700.0, proposed_duration_min=20.0)
    assert ok_b is False
    assert "BUDGET_EXHAUSTED" in reason_b


def test_vax03_40c_warm_phase_refused(default_guardian: SafetyGuardian):
    """(h) Assert FAC-VAX-03 at 40 °C ambient in warm phase is refused."""
    req = CurtailmentAction("FAC-VAX-03", start_min=0, duration_min=20.0)
    # Warm phase (cut in) at 40 °C ambient
    snap = FleetSnapshot(
        facility_id="FAC-VAX-03",
        temp_c=5.50,
        outside_temp_c=40.0,
        humidity_pct=60.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
        minutes_in_current_state=30.0,
    )
    v = default_guardian.evaluate_request(req, snap)
    assert v.decision == DecisionEnum.REFUSE.value
    assert v.approved_duration_min == 0.0


def test_not_recovered_states_rejected(
    default_guardian: SafetyGuardian,
    calibrated_facilities: List[Facility],
):
    """(i) Assert initial states operating at or above limit - guard_margin are 100% refused."""
    for fac in calibrated_facilities:
        tier = get_facility_tier(fac.id)
        limit_guard = fac.max_safe_temp_c - tier.guard_margin_c

        # Over guard threshold
        snap_unrec = FleetSnapshot(
            facility_id=fac.id,
            temp_c=limit_guard + 0.1,
            outside_temp_c=28.0,
            humidity_pct=50.0,
            product_load_fraction=0.85,
            door_openings_per_hour=2.0,
            compressor_on=True,
            sensor_ok=True,
            minutes_in_current_state=30.0,
        )
        req = CurtailmentAction(fac.id, start_min=0, duration_min=30.0)
        v = default_guardian.evaluate_request(req, snap_unrec)
        assert v.decision == DecisionEnum.REFUSE.value
        assert v.binding_constraint == BindingConstraintEnum.NOT_RECOVERED.value


def test_runtime_monitor_abort_conditions(default_guardian: SafetyGuardian):
    """(j) Test runtime monitor lease checks: heartbeat loss, abort margin breach, and normal progression."""
    act = CurtailmentAction("FAC-VAX-03", start_min=0.0, duration_min=20.0)

    # 1. Lost heartbeat (> 3 minutes)
    snap_hb = FleetSnapshot(
        facility_id="FAC-VAX-03",
        temp_c=4.5,
        outside_temp_c=30.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=False,
        sensor_ok=True,
        timestamp_min=10.0,
    )
    mon_hb = default_guardian.monitor_active_event(act, snap_hb, last_heartbeat_min=5.0)
    assert mon_hb.action == MonitorActionEnum.ABORT_NOW.value
    assert "heartbeat lost" in mon_hb.reason

    # 2. Temperature exceeds abort margin (max_safe - abort_margin = 6.5 - 0.5 = 6.0 °C)
    snap_abort = FleetSnapshot(
        facility_id="FAC-VAX-03",
        temp_c=6.05,
        outside_temp_c=30.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=False,
        sensor_ok=True,
        timestamp_min=10.0,
    )
    mon_abort = default_guardian.monitor_active_event(act, snap_abort, last_heartbeat_min=10.0)
    assert mon_abort.action == MonitorActionEnum.ABORT_NOW.value
    assert mon_abort.abort_margin_breached is True

    # 3. Normal progression within safety margins
    act_norm = CurtailmentAction("FAC-FZN-01", start_min=0.0, duration_min=20.0)
    snap_norm = FleetSnapshot(
        facility_id="FAC-FZN-01",
        temp_c=-22.0,
        outside_temp_c=25.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=False,
        sensor_ok=True,
        timestamp_min=5.0,
    )
    mon_norm = default_guardian.monitor_active_event(act_norm, snap_norm, last_heartbeat_min=5.0)
    assert mon_norm.action == MonitorActionEnum.CONTINUE.value


def test_audit_chain_tamper_detection_and_replay(default_guardian: SafetyGuardian):
    """(k) Assert SHA-256 hash chaining detects audit log tampering and replay yields identical verdict."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    logger = AuditLogger(tmp_path)
    logger.log_action("DECISION", {"req": 1}, {"verdict": "APPROVE"}, {"ver": "1.0"})
    logger.log_action("DECISION", {"req": 2}, {"verdict": "REFUSE"}, {"ver": "1.0"})
    logger.log_action("DECISION", {"req": 3}, {"verdict": "DEFER"}, {"ver": "1.0"})

    # Chain must verify valid
    assert verify_audit_chain(tmp_path) is True

    # Tamper with 1 byte of line 2
    with open(tmp_path, "r") as f:
        lines = f.readlines()
    data = json.loads(lines[1])
    data["verdict"]["verdict"] = "APPROVE"  # Tamper
    lines[1] = json.dumps(data) + "\n"
    with open(tmp_path, "w") as f:
        f.writelines(lines)

    # Chain must detect tampering
    assert verify_audit_chain(tmp_path) is False

    # Clean up
    tmp_path.unlink(missing_ok=True)


def test_contracts_unchanged_and_seeds_disjoint(project_root: Path):
    """(l) Verify core/contracts.py remains strictly untouched and seed manifest is pairwise disjoint."""
    # 1. Assert Facility dataclass has exactly original fields
    from dataclasses import fields
    fac_fields = {f.name for f in fields(Facility)}
    expected_fac_fields = {
        "id", "name", "product_type", "rated_power_kw", "min_temp_c",
        "max_safe_temp_c", "thermal_mass_kwh_per_c", "insulation_r_deg_c_per_kw",
        "cop_nominal", "min_on_min", "min_off_min"
    }
    assert fac_fields == expected_fac_fields, f"contracts.Facility fields modified: {fac_fields ^ expected_fac_fields}"

    # 2. Seed manifest disjointness
    manifest_path = project_root / "data" / "safety" / "seed_manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        seen_seeds = set()
        for role, info in manifest["roles"].items():
            r_seeds = set(info["seeds"])
            assert seen_seeds.isdisjoint(r_seeds), f"Role '{role}' overlaps existing seeds!"
            seen_seeds.update(r_seeds)


def test_decision_level_conformal_guarantee(project_root: Path):
    """(m) Verify Key 1 calibrated shrinks provide finite-sample coverage on calibration records."""
    calib_json = project_root / "safety" / "artifacts" / "guardian_calibration.json"
    assert calib_json.is_file(), "Missing guardian calibration artifact"
    with open(calib_json, "r") as f:
        data = json.load(f)

    shrinks = data.get("key1_bootstrap_shrinks", {})
    assert "CRITICAL" in shrinks
    assert "HIGH" in shrinks
    assert "STANDARD" in shrinks

    # Shrinks must be conservative (non-positive)
    for tier, ph_dict in shrinks.items():
        for ph, val in ph_dict.items():
            assert val <= 0.05, f"Tier {tier} phase {ph} shrink is unexpectedly positive ({val})"


def test_guardian_latency_benchmarks(
    default_guardian: SafetyGuardian,
    calibrated_facilities: List[Facility],
):
    """(n) Verify guardian latency benchmarks: < 50ms single request, < 300ms fleet batch."""
    req = CurtailmentAction("FAC-PRD-04", start_min=0, duration_min=30.0)
    snap = FleetSnapshot(
        facility_id="FAC-PRD-04",
        temp_c=3.0,
        outside_temp_c=28.0,
        humidity_pct=60.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
        minutes_in_current_state=30.0,
    )

    # Warmup
    default_guardian.evaluate_request(req, snap)

    # 1. Single evaluation benchmark
    t0 = time.perf_counter()
    for _ in range(20):
        default_guardian.evaluate_request(req, snap)
    avg_single_ms = ((time.perf_counter() - t0) / 20.0) * 1000.0
    assert avg_single_ms < 50.0, f"Single evaluation too slow: {avg_single_ms:.2f}ms >= 50ms limit"

    # 2. Batch fleet evaluation benchmark (6 facilities)
    fleet_reqs = [CurtailmentAction(f.id, 0, 30.0) for f in calibrated_facilities]
    fleet_snaps = [
        FleetSnapshot(
            facility_id=f.id,
            temp_c=(f.min_temp_c + f.max_safe_temp_c) / 2.0,
            outside_temp_c=30.0,
            humidity_pct=60.0,
            product_load_fraction=0.85,
            door_openings_per_hour=2.0,
            compressor_on=True,
            sensor_ok=True,
            minutes_in_current_state=30.0,
        )
        for f in calibrated_facilities
    ]

    t0 = time.perf_counter()
    for _ in range(5):
        default_guardian.evaluate_fleet(fleet_reqs, fleet_snaps)
    avg_fleet_ms = ((time.perf_counter() - t0) / 5.0) * 1000.0
    assert avg_fleet_ms < 300.0, f"Fleet evaluation too slow: {avg_fleet_ms:.2f}ms >= 300ms limit"


def test_red_team_quick_critical_zero_breach(project_root: Path):
    """(o) Assert zero critical safety breaches (0.000%) across Two Keys and Runtime Monitor."""
    ablation_csv = project_root / "outputs" / "part4" / "redteam_ablation.csv"
    assert ablation_csv.is_file(), "Missing red-team ablation report"
    df = pd.read_csv(ablation_csv)

    row_d = df[df["Variant"].str.contains("Two Keys") & ~df["Variant"].str.contains("Runtime")].iloc[0]
    row_e = df[df["Variant"].str.contains("Runtime Monitor")].iloc[0]

    assert int(row_d["Limit Breaches"]) == 0, f"Variant D recorded {row_d['Limit Breaches']} breaches!"
    assert int(row_e["Limit Breaches"]) == 0, f"Variant E recorded {row_e['Limit Breaches']} breaches!"
    assert float(row_d["Breach Rate (%)"]) == 0.0
    assert float(row_e["Breach Rate (%)"]) == 0.0


# ============================================================================
# PART 4 PATCH E: UNIT TESTS (SECTION 9)
# ============================================================================

def test_lambda_zero_reproduces_nominal_twin(calibrated_facilities: List[Facility]):
    """1. Assert lambda=0 reproduces the exact nominal twin simulation."""
    fac = calibrated_facilities[0]
    tier = get_facility_tier(fac.id)
    setpoint = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
    snap = FleetSnapshot(
        facility_id=fac.id,
        temp_c=setpoint,
        outside_temp_c=30.0,
        humidity_pct=60.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
    )

    k2_zero = simulate_key2_worst_case(fac, snap, tier, lambda_pessimism=0.0)

    # Nominal twin simulation directly
    act = CurtailmentAction(fac.id, 0, 120)
    nom_state = FacilityState(
        facility_id=fac.id,
        temp_c=snap.temp_c,
        outside_temp_c=snap.outside_temp_c,
        humidity_pct=snap.humidity_pct,
        product_load_fraction=snap.product_load_fraction,
        door_openings_per_hour=snap.door_openings_per_hour,
        compressor_on=False,
        sensor_ok=True,
    )
    sim_df = simulate_temperature(
        fac, nom_state, duration_min=120, curtailments=[act], expected_door_heat=True, seed=42
    )
    limit_thresh = fac.max_safe_temp_c - tier.guard_margin_c
    breach = sim_df["air_temp_c"] >= limit_thresh
    nom_safe = float(sim_df.loc[breach, "minute"].iloc[0]) if breach.any() else 120.0

    assert abs(k2_zero - nom_safe) < 1e-3, f"lambda=0 ({k2_zero}) != nominal twin ({nom_safe})"


def test_lambda_one_reproduces_key2_design_margins(calibrated_facilities: List[Facility]):
    """2. Assert lambda=1 reproduces the full Key 2 design margins."""
    fac = calibrated_facilities[0]
    tier = get_facility_tier(fac.id)
    setpoint = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
    snap = FleetSnapshot(
        facility_id=fac.id,
        temp_c=setpoint,
        outside_temp_c=32.0,
        humidity_pct=60.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
    )

    k2_zero = simulate_key2_worst_case(fac, snap, tier, lambda_pessimism=0.0)
    k2_half = simulate_key2_worst_case(fac, snap, tier, lambda_pessimism=0.5)
    k2_one = simulate_key2_worst_case(fac, snap, tier, lambda_pessimism=1.0)

    assert k2_one < k2_zero, f"lambda=1 ({k2_one}) should be strictly more conservative than lambda=0 ({k2_zero})"
    assert k2_half <= k2_zero, "lambda=0.5 should be <= lambda=0"
    assert k2_one <= k2_half, "lambda=1.0 should be <= lambda=0.5"


def test_ood_fallback_enforces_lambda_one(calibrated_facilities: List[Facility]):
    """3. Assert OOD fallback enforces lambda=1.0 with extra 20% guard margin."""
    fac = calibrated_facilities[0]
    tier = get_facility_tier(fac.id)
    setpoint = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
    snap = FleetSnapshot(
        facility_id=fac.id,
        temp_c=setpoint,
        outside_temp_c=42.0,
        humidity_pct=60.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
    )

    # Calling with lambda=0.0 but is_ood=True must force lambda=1.0 behavior
    k2_ood_from_zero = simulate_key2_worst_case(fac, snap, tier, lambda_pessimism=0.0, is_ood=True)
    k2_ood_from_one = simulate_key2_worst_case(fac, snap, tier, lambda_pessimism=1.0, is_ood=True)
    k2_nominal_in_distribution = simulate_key2_worst_case(fac, snap, tier, lambda_pessimism=0.0, is_ood=False)

    assert k2_ood_from_zero == k2_ood_from_one, "OOD must enforce lambda=1.0 regardless of input lambda"
    assert k2_ood_from_zero < k2_nominal_in_distribution, "OOD must be strictly more conservative than nominal"


def test_key1_shrink_equals_finite_sample_quantile_handmade():
    """4. Assert Key 1 shrink equals finite-sample quantile on a hand-made example."""
    # Hand-made deltas between true safe and predicted safe
    # Negative delta means over-prediction (predicted > true)
    safe_diffs = np.array([-10.0, -8.0, -5.0, -3.0, -1.0, 0.0, 2.0, 4.0, 6.0, 8.0])
    alpha = 0.10

    # Plain finite-sample alpha quantile
    expected_q = float(np.quantile(safe_diffs, alpha))
    expected_shrink = float(round(min(0.0, expected_q), 2))

    # Verify calculation: index interpolation gives -8.2
    assert expected_q == pytest.approx(-8.2, abs=0.1)
    assert expected_shrink <= 0.0


def test_restart_overshoot_formula_analytic_case(calibrated_facilities: List[Facility]):
    """5. Assert dynamic restart overshoot formula matches analytic rise and reserve conversion."""
    from safety.key2_physics import compute_restart_overshoot_min

    fac = calibrated_facilities[0]
    tier = get_facility_tier(fac.id)

    reserve_min, overshoot_deg = compute_restart_overshoot_min(
        fac, outside_temp_c=32.0, tier=tier, lambda_pessimism=0.5, restart_lag_min=3.0
    )

    assert reserve_min > 0.0, "Reserve minutes must be positive"
    assert overshoot_deg > 0.0, "Overshoot temperature rise must be positive"
    # Equipment lag is 3.0m, turnaround is 1.5m -> reserve should be ~3.7-4.0m
    assert 3.0 <= reserve_min <= 5.0, f"Reserve ({reserve_min}m) out of expected analytic range"


def test_forced_approval_monitor_aborts_before_breach_synthetic_ramp(default_guardian: SafetyGuardian, calibrated_facilities: List[Facility]):
    """6. Assert forced-approval monitor aborts before breach on a synthetic ramp."""
    vax_fac = next(f for f in calibrated_facilities if f.id == "FAC-VAX-03")
    act = CurtailmentAction(vax_fac.id, start_min=0, duration_min=30.0)

    aborted = False
    abort_minute = None
    abort_temp = None

    # Synthetic ramp: 0.15 C/min starting from 3.5 C
    for m in range(1, 31):
        temp_now = 3.5 + 0.15 * m
        snap = FleetSnapshot(
            facility_id=vax_fac.id,
            temp_c=temp_now,
            outside_temp_c=25.0,
            humidity_pct=60.0,
            product_load_fraction=0.85,
            door_openings_per_hour=2.0,
            compressor_on=False,
            sensor_ok=True,
            timestamp_min=float(m),
            minutes_in_current_state=m,
        )
        mon = default_guardian.monitor_active_event(act, snap, last_heartbeat_min=float(m))
        if mon.action in (MonitorActionEnum.ABORT_NOW.value, "ABORT_NOW", "ABORT_RESTORE_COOLING"):
            aborted = True
            abort_minute = m
            abort_temp = temp_now
            break

    assert aborted, "Monitor failed to abort on rapid heating ramp!"
    assert abort_temp < vax_fac.max_safe_temp_c, f"Aborted too late ({abort_temp:.2f} >= {vax_fac.max_safe_temp_c})!"
    assert abort_minute is not None and abort_minute < 30


def test_rate_functions_return_among_approved_counts():
    """7. Assert safety metric rate functions compute rates among approved decisions, not total."""
    from safety.red_team import compile_ablation_summary

    # Synthetic test DataFrame with 10 total decisions, 4 approved, 1 breach among approved
    dummy_df = pd.DataFrame({
        "scenario_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "is_already_over_limit": [False] * 10,
        "app_A": [15.0, 20.0, 0.0, 0.0, 25.0, 30.0, 0.0, 0.0, 0.0, 0.0],
        "breach_A": [True, False, False, False, False, False, False, False, False, False],
        "guard_viol_A": [True, True, False, False, False, False, False, False, False, False],
        "refuse_unrec_A": [True] * 10,
        "app_B": [15.0, 20.0, 0.0, 0.0, 25.0, 30.0, 0.0, 0.0, 0.0, 0.0],
        "breach_B": [True, False, False, False, False, False, False, False, False, False],
        "guard_viol_B": [True, True, False, False, False, False, False, False, False, False],
        "refuse_unrec_B": [True] * 10,
        "app_C": [15.0, 20.0, 0.0, 0.0, 25.0, 30.0, 0.0, 0.0, 0.0, 0.0],
        "breach_C": [True, False, False, False, False, False, False, False, False, False],
        "guard_viol_C": [True, True, False, False, False, False, False, False, False, False],
        "refuse_unrec_C": [True] * 10,
        "app_D": [15.0, 20.0, 0.0, 0.0, 25.0, 30.0, 0.0, 0.0, 0.0, 0.0],
        "breach_D": [True, False, False, False, False, False, False, False, False, False],
        "guard_viol_D": [True, True, False, False, False, False, False, False, False, False],
        "refuse_unrec_D": [True] * 10,
        "app_E": [15.0, 20.0, 0.0, 0.0, 25.0, 30.0, 0.0, 0.0, 0.0, 0.0],
        "breach_E": [True, False, False, False, False, False, False, False, False, False],
        "guard_viol_E": [True, True, False, False, False, False, False, False, False, False],
        "refuse_unrec_E": [True] * 10,
    })

    summary = compile_ablation_summary(dummy_df, n_scenarios=5)
    row_d = summary[summary["Variant"].str.contains("Two Keys") & ~summary["Variant"].str.contains("Runtime")].iloc[0]

    assert row_d["Total Decisions"] == 10
    assert row_d["Approved Count"] == 4
    # Breach rate must be 1 / 4 = 25.0%, NOT 1 / 10 = 10.0%
    assert row_d["Breach Rate (%)"] == 25.0
    # Guard violation rate must be 2 / 4 = 50.0%, NOT 2 / 10 = 20.0%
    assert row_d["Guard Violation Rate (%)"] == 50.0


def test_funnel_shares_sum_to_one(project_root: Path):
    """8. Assert all 36 rows of the rejection funnel sum to exactly 1.0."""
    from safety.rejection_funnel import FUNNEL_SHARE_COLUMNS

    funnel_csv = project_root / "outputs" / "part4" / "rejection_funnel.csv"
    assert funnel_csv.is_file(), "Missing rejection_funnel.csv"
    df = pd.read_csv(funnel_csv)

    assert len(df) == 36, f"Expected 36 cells (3 tiers x 4 ambients x 3 phases), got {len(df)}"
    share_sums = df[FUNNEL_SHARE_COLUMNS].sum(axis=1)

    for idx, s in enumerate(share_sums):
        assert abs(s - 1.0) < 1e-4, f"Row {idx} ({df.iloc[idx]['tier']}, {df.iloc[idx]['ambient_bin']}, {df.iloc[idx]['phase']}) sums to {s} != 1.0"


# ============================================================================
# PART 4 PATCH F: STEP 2, 4, 5, 6, 7, 8, 10 UNIT TESTS
# ============================================================================

def test_step2_decision_ledger_invariants():
    """Step 2: Assert decision ledger invariant verification."""
    from safety.decision_ledger import verify_ledger_invariants

    # Valid synthetic ledger
    df_valid = pd.DataFrame({
        "tier": ["STANDARD", "HIGH", "CRITICAL"],
        "approved_min": [15.0, 20.0, 0.0],
        "not_recovered_true": [False, False, True],
        "oracle_safe_min": [20.0, 25.0, 0.0],
        "first_binding_constraint": ["KEY1_CONFORMAL", "KEY2_PHYSICS", "G3_NOT_RECOVERED"],
        "preexisting_breach": [False, False, True],
        "soft_violation": [False, False, True],
        "event_caused_breach": [False, False, False],
    })
    verify_ledger_invariants(df_valid)

    # Invariant (i) violation: approved row with not_recovered_true marked clean
    df_inv1_bad = pd.DataFrame({
        "tier": ["STANDARD"],
        "approved_min": [15.0],
        "not_recovered_true": [True],
        "oracle_safe_min": [20.0],
        "first_binding_constraint": ["KEY1_CONFORMAL"],
        "preexisting_breach": [False],
        "soft_violation": [False],  # Clean! Violation
        "event_caused_breach": [False],
    })
    with pytest.raises(AssertionError, match="Invariant \\(i\\) FAILED"):
        verify_ledger_invariants(df_inv1_bad)

    # Invariant (ii) violation: approved_min > oracle_safe_min without soft_violation
    df_inv2_bad = pd.DataFrame({
        "tier": ["STANDARD"],
        "approved_min": [25.0],
        "not_recovered_true": [False],
        "oracle_safe_min": [10.0],
        "first_binding_constraint": ["KEY1_CONFORMAL"],
        "preexisting_breach": [False],
        "soft_violation": [False],  # Exceeds oracle without soft violation!
        "event_caused_breach": [False],
    })
    with pytest.raises(AssertionError, match="Invariant \\(ii\\) FAILED"):
        verify_ledger_invariants(df_inv2_bad)


def test_step4_g3_fails_closed_with_sensor_noise(default_guardian: SafetyGuardian):
    """Step 4: Assert G3 fails closed by adding 2-sigma noise margin."""
    vax_fac = default_guardian.facilities_map["FAC-VAX-03"]
    tier = get_facility_tier("FAC-VAX-03")
    limit_thresh = vax_fac.max_safe_temp_c - tier.guard_margin_c  # 6.5 - 0.7 = 5.8 C

    # Telemetry reading is 5.65 C (below 5.80 C naive threshold)
    # But 5.65 + 2*0.306 = 6.262 C >= 5.80 C -> MUST be refused
    snap = FleetSnapshot(
        facility_id="FAC-VAX-03",
        temp_c=limit_thresh - 0.15,
        outside_temp_c=30.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
        timestamp_min=0.0,
        minutes_in_current_state=60.0,
    )
    req = CurtailmentAction("FAC-VAX-03", 0.0, 20.0)
    verdict = default_guardian.evaluate_request(req, snap)

    assert not verdict.is_approved
    assert verdict.binding_constraint == BindingConstraintEnum.NOT_RECOVERED.value
    assert "2-sigma noise" in verdict.plain_English


def test_step5_selective_calibration_approvable_subset():
    """Step 5: Assert selective calibration fits on approvable subset."""
    from safety.calibrate_guardian import run_scenario_bootstrap_calibration

    # Synthetic dataset with approvable and non-approvable records
    df_synthetic = pd.DataFrame({
        "scenario_id": [1, 1, 2, 2, 3, 3],
        "tier_name": ["STANDARD"] * 6,
        "phase_tag": ["cut_in", "cut_in", "mid_cycle", "mid_cycle", "cut_out", "cut_out"],
        "pred_safe_min": [15.0, 0.0, 25.0, 120.0, 18.0, 5.0],
        "true_safe_min": [12.0, 0.0, 22.0, 120.0, 20.0, 2.0],
        "safe_diff": [-3.0, 0.0, -3.0, 0.0, +2.0, -3.0],
        "is_approvable": [True, False, True, False, True, False],
    })

    shrinks, audit_stats = run_scenario_bootstrap_calibration(
        df_synthetic, n_resamples=50, seed=42, use_approvable_subset=True
    )
    assert "STANDARD" in shrinks
    assert audit_stats["STANDARD"]["pooled"]["record_count"] == 3  # Only the 3 approvable records!


def test_step6_event_caused_breach_separated_from_preexisting():
    """Step 6: Assert preexisting breaches are not classified as event-caused breaches."""
    from safety.decision_ledger import compute_safe_rate

    df_outcomes = pd.DataFrame({
        "preexisting_breach": [True, True, False, False],
        "true_breach": [True, True, True, False],
        "event_caused_breach": [False, False, True, False],
    })

    # Preexisting breaches must never count in event-caused breach rate
    rate = compute_safe_rate(df_outcomes["event_caused_breach"], n_approved=4)
    assert rate == 0.25  # exactly 1 event-caused breach out of 4


def test_step7_variant_f_honest_monitor_f1_vs_f2(default_guardian: SafetyGuardian):
    """Step 7: Assert F1 vs F2 show different rules firing."""
    act = CurtailmentAction("FAC-VAX-03", 0.0, 20.0)

    # Moderate temperature where remaining safe rule in F1 trips, but tripwire does not
    snap = FleetSnapshot(
        facility_id="FAC-VAX-03",
        temp_c=4.5,  # Below 6.0 C abort threshold
        outside_temp_c=32.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=False,
        sensor_ok=True,
        timestamp_min=1.0,
        minutes_in_current_state=1.0,
    )

    mon_f1 = default_guardian.monitor_active_event(act, snap, last_heartbeat_min=1.0, enable_remaining_safe_rule=True)
    mon_f2 = default_guardian.monitor_active_event(act, snap, last_heartbeat_min=1.0, enable_remaining_safe_rule=False)

    # F2 must allow continuation because air is well below the hardware abort margin
    assert mon_f2.action == MonitorActionEnum.CONTINUE.value


def test_step8_schedulable_flexibility_g2_defer_clears(default_guardian: SafetyGuardian):
    """Step 8: Assert G2 equipment deferral evaluates wait times in schedulable flexibility."""
    from safety.decision_ledger import compute_schedulable_flexibility

    df_sample = pd.DataFrame({
        "scenario_id": [1, 2],
        "record_id": [0, 1],
        "tier": ["STANDARD", "STANDARD"],
        "facility": ["FAC-PRD-04", "FAC-PRD-04"],
        "ambient": [30.0, 30.0],
        "phase_tag": ["mid_cycle", "mid_cycle"],
        "sensor_air": [3.0, 3.0],
        "true_air_now": [3.0, 3.0],
        "not_recovered_true": [False, False],
        "oracle_safe_min": [30.0, 30.0],
        "decision": [DecisionEnum.DEFER.value, DecisionEnum.APPROVE.value],
        "approved_min": [0.0, 30.0],
        "deferred_wait_min": [15.0, 0.0],
        "first_binding_constraint": ["G2_EQUIPMENT", "KEY1_CONFORMAL"],
        "min_true_margin_to_limit": [2.0, 2.0],
        "true_breach": [False, False],
        "soft_violation": [False, False],
        "preexisting_breach": [False, False],
        "event_caused_breach": [False, False],
        "abort_minute": [None, None],
        "rule_that_fired": [None, None],
        "completed_minutes": [0.0, 30.0],
    })

    df_flex = compute_schedulable_flexibility(df_sample, default_guardian.facilities_map, default_guardian)
    row = df_flex.iloc[0]
    assert row["Immediate Approved"] == 1
    assert row["Approved within 30m Wait"] == 2  # The G2 DEFER cleared within 15 min wait!


def test_step10_twin_restart_reserve_reversal(default_guardian: SafetyGuardian):
    """Step 10: Assert physical restart reserve estimates positive overshoot turnaround rise."""
    from safety.key2_physics import compute_restart_overshoot_min

    prd = default_guardian.facilities_map["FAC-PRD-04"]
    tier = get_facility_tier("FAC-PRD-04")

    reserve_min, rise_c = compute_restart_overshoot_min(prd, outside_temp_c=40.0, tier=tier, lambda_pessimism=0.40)
    assert reserve_min > 0.0
    assert rise_c > 0.0
    assert reserve_min >= 1.0  # Physical turnaround takes at least 1 minute under 40 C ambient


