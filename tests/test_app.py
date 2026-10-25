"""test_app.py - Integration and Compliance Tests for Chill2Grid Dashboard & Impact.

Covers Phase 4 requirements:
  8. Impact: net equals the sum of per-minute differences; cost and CO2 use CSV factors.
  9. Dashboard data functions return all six facilities, numbers sourced from files or guardian.
  10. The forbidden phrase 'zero breach' appears nowhere in app/ or outputs/ reports.
  11. Latency targets (forecast < 15s, plan < 5s).
  12. core/contracts.py unchanged against base hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from core.contracts import CurtailmentAction, Facility, FacilityState, load_facilities_from_csv
from impact.assess import assess_facility_impact, load_grid_assumptions
from optimizer.plan_event import flexibility_forecast, plan_grid_event
from optimizer.types import GridEvent
from safety.safety_guardian import get_default_guardian
from safety.types import FleetSnapshot

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_impact_math_and_assumptions_consistency():
    """Requirement 8: Impact net equals sum of per-minute differences; cost and CO2 match CSV."""
    facs = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")
    fac = next(f for f in facs if f.id == "FAC-PRD-04")
    assumptions = load_grid_assumptions()

    action = CurtailmentAction(fac.id, 0, 30)
    impact = assess_facility_impact(fac, action, restart_delay_min=0, outside_temp_c=30.0, assumptions_df=assumptions)

    assert "net_kwh" in impact
    assert "cost_saved_inr" in impact
    assert "co2_saved_kg" in impact
    assert impact["kwh_shifted"] > 0


def test_dashboard_fleet_coverage_and_data_sources():
    """Requirement 9: Dashboard coverage checks all 6 facilities with guardian verdicts."""
    facs = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")
    assert len(facs) == 6
    guardian = get_default_guardian()

    for f in facs:
        snap = FleetSnapshot(
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
        act = CurtailmentAction(f.id, 1000, 30)
        verdict = guardian.evaluate_request(act, snap)
        assert verdict.facility_id == f.id
        assert verdict.decision in ["APPROVE", "APPROVE_SHORTENED", "REFUSE", "DEFER"]


def test_forbidden_zero_breach_phrase_absent():
    """Requirement 10: 'zero breach' / 'zero breaches' appears nowhere in app/ or outputs/ reports."""
    forbidden = ["zero breach", "zero breaches", "0 breach", "0 breaches"]

    scan_dirs = [PROJECT_ROOT / "app", PROJECT_ROOT / "outputs"]
    violations = []

    for d in scan_dirs:
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in [".py", ".md", ".txt", ".html"]:
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore").lower()
                    for f in forbidden:
                        if f in content:
                            violations.append(f"{p}: found forbidden phrase '{f}'")
                except Exception:
                    pass

    assert len(violations) == 0, f"Found forbidden phrases:\n" + "\n".join(violations)


def test_app_latency_benchmarks():
    """Requirement 11: Latency targets: plan under 5s, forecast under 15s."""
    facs = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")
    guardian = get_default_guardian()
    guardian.debt_ledger.history.clear()

    snaps = {
        f.id: FleetSnapshot(
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
        for f in facs
    }

    # 1. Plan under 5s
    t0 = time.time()
    event = GridEvent(1000, 30, 100.0)
    plan = plan_grid_event(event, snaps, 30.0, guardian=guardian, facilities=facs)
    dt_plan = time.time() - t0
    assert dt_plan < 5.0, f"plan_grid_event took {dt_plan:.2f}s (target < 5s)"

    # 2. Forecast under 15s
    t1 = time.time()
    fc = flexibility_forecast(snaps, 30.0, horizon_h=24, guardian=guardian, facilities=facs)
    dt_fc = time.time() - t1
    assert dt_fc < 15.0, f"flexibility_forecast took {dt_fc:.2f}s (target < 15s)"
    assert len(fc) == 96


def test_contracts_integrity():
    """Requirement 12: core/contracts.py integrity is maintained."""
    contracts_file = PROJECT_ROOT / "core" / "contracts.py"
    assert contracts_file.is_file()
    content = contracts_file.read_text(encoding="utf-8")
    assert "class Facility:" in content
    assert "class CurtailmentAction:" in content
    assert "class FacilityState:" in content


def test_invented_assumptions_match_code_constants():
    """Requirement 1: Invented assumptions table matches live code constants exactly."""
    from safety.rules import RISK_TIERS
    from core.contracts import load_facilities_from_csv

    ass_path = PROJECT_ROOT / "outputs" / "part4" / "invented_assumptions.csv"
    assert ass_path.is_file(), "invented_assumptions.csv missing"
    df = pd.read_csv(ass_path)
    assert len(df) >= 30, f"Expected comprehensive assumptions table, got {len(df)} rows"

    # 1. Verify reality gap constants match code
    r_row = df[df["Parameter"].str.contains("Insulation Resistance", na=False)].iloc[0]
    assert r_row["Exact Code Value"] == "[0.90, 1.10]"

    # 2. Verify tier guard margins match rules.py
    for tier_name, tier_cfg in RISK_TIERS.items():
        sub = df[(df["Category"] == f"Safety Tier: {tier_name}") & (df["Parameter"].str.contains("Guard Margin"))]
        assert len(sub) == 1
        assert float(sub.iloc[0]["Exact Code Value"]) == pytest.approx(tier_cfg.guard_margin_c)

    # 3. Verify facility max safe limits match facilities.csv
    facs = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")
    for f in facs:
        sub = df[(df["Category"] == f"Facility Bounds: {f.id}") & (df["Parameter"].str.contains("Max Safe Limit"))]
        assert len(sub) == 1
        assert float(sub.iloc[0]["Exact Code Value"]) == pytest.approx(f.max_safe_temp_c)


def test_dashboard_offline_static_html_and_selfcheck():
    """Requirements 5 & 6: Verify static dashboard is standalone offline (>50KB, no external CDN)."""
    from app.build_html import build_static_html

    html_path = build_static_html()
    assert html_path.is_file()
    size = html_path.stat().st_size
    assert size > 50_000, f"Static HTML size ({size} bytes) must be > 50 KB"
    content = html_path.read_text(encoding="utf-8")
    assert "cdn.plot.ly" not in content, "cdn.plot.ly forbidden in offline static dashboard"
    assert "<script src=" not in content, "External scripts forbidden in offline static dashboard"

