"""selfcheck.py - End-to-End Prototype Integration Self-Check.

Executes:
  1. Load fleet configuration
  2. Compute 24-hour flexibility forecast
  3. Evaluate Safety Guardian verdicts across fleet
  4. Plan 60-minute grid event at 18:00 on 35°C day (target 40% fleet power)
  5. Assess physical & economic impact with grid assumptions
  6. Rebuild static offline dashboard (outputs/dashboard_static.html)

Prints a concise PASS/FAIL list with timings.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Dict, List, Tuple

import pandas as pd

from app.build_html import build_static_html
from core.contracts import CurtailmentAction, Facility, load_facilities_from_csv
from impact.assess import assess_plan_impact, load_grid_assumptions
from optimizer.plan_event import flexibility_forecast, plan_grid_event
from optimizer.types import GridEvent
from safety.safety_guardian import get_default_guardian
from safety.types import FleetSnapshot

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_selfcheck() -> bool:
    print("=" * 65)
    print("      CHILL2GRID AI - SUBMISSION INTEGRATION SELF-CHECK       ")
    print("=" * 65)

    checklist: List[Tuple[str, str, float]] = []
    overall_success = True

    # 1. Load Fleet
    t0 = time.time()
    try:
        facilities = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")
        assert len(facilities) == 6
        checklist.append(("Load fleet facilities (6 rooms)", "PASS", time.time() - t0))
    except Exception as e:
        checklist.append((f"Load fleet facilities: {e}", "FAIL", time.time() - t0))
        overall_success = False

    # 2. Flexibility Forecast
    t0 = time.time()
    try:
        snaps = {
            f.id: FleetSnapshot(
                facility_id=f.id,
                temp_c=f.min_temp_c + 0.1,
                outside_temp_c=35.0,
                humidity_pct=50.0,
                product_load_fraction=0.85,
                door_openings_per_hour=2.0,
                compressor_on=True,
                sensor_ok=True,
                timestamp_min=1080.0,
                minutes_in_current_state=45.0,
            )
            for f in facilities
        }
        guardian = get_default_guardian()
        guardian.debt_ledger.history.clear()
        df_fc = flexibility_forecast(snaps, 35.0, horizon_h=24, guardian=guardian, facilities=facilities)
        assert len(df_fc) == 96
        checklist.append(("24-hour flexibility forecast (96 steps)", "PASS", time.time() - t0))
    except Exception as e:
        checklist.append((f"Flexibility forecast: {e}", "FAIL", time.time() - t0))
        overall_success = False

    # 3. Guardian Evaluation
    t0 = time.time()
    try:
        verdicts = {}
        for f in facilities:
            act = CurtailmentAction(f.id, 1080, 60)
            v = guardian.evaluate_request(act, snaps[f.id])
            verdicts[f.id] = v.decision
        checklist.append(("Safety Guardian fleet gating", "PASS", time.time() - t0))
    except Exception as e:
        checklist.append((f"Guardian fleet gating: {e}", "FAIL", time.time() - t0))
        overall_success = False

    # 4. Plan Grid Event (60m at 18:00, 35°C, target 40% fleet power)
    t0 = time.time()
    fleet_kw = sum(f.rated_power_kw for f in facilities)
    target_kw = fleet_kw * 0.40
    try:
        event = GridEvent(start_min=1080, duration_min=60, target_kw=target_kw)
        guardian.debt_ledger.history.clear()
        plan = plan_grid_event(event, snaps, 35.0, guardian=guardian, facilities=facilities)
        assert len(plan.actions) > 0
        checklist.append((f"Plan 60m event (target {target_kw:.0f} kW)", "PASS", time.time() - t0))
    except Exception as e:
        checklist.append((f"Plan grid event: {e}", "FAIL", time.time() - t0))
        overall_success = False
        plan = None

    # 5. Impact Assessment
    t0 = time.time()
    try:
        assumptions = load_grid_assumptions()
        if plan is not None:
            impact = assess_plan_impact(plan, facilities, outside_temp_c=35.0, assumptions_df=assumptions)
            checklist.append((f"Impact assessment ({impact['peak_shaved_kw']:.1f} kW shaved)", "PASS", time.time() - t0))
        else:
            checklist.append(("Impact assessment (skipped)", "FAIL", time.time() - t0))
            overall_success = False
    except Exception as e:
        checklist.append((f"Impact assessment: {e}", "FAIL", time.time() - t0))
        overall_success = False

    # 6. Build Static HTML Dashboard & Verify Standalone Offline (> 50 KB)
    t0 = time.time()
    try:
        html_path = build_static_html()
        assert html_path.is_file(), "Static HTML file missing"
        size = html_path.stat().st_size
        assert size > 50_000, f"Static HTML size ({size} bytes) must be > 50 KB"
        content = html_path.read_text(encoding="utf-8")
        assert "<script src=" not in content, "Static HTML must have zero external script references"
        checklist.append((f"Static dashboard (>50KB, offline: {size/1024:.1f} KB)", "PASS", time.time() - t0))
    except Exception as e:
        checklist.append((f"Static dashboard: {e}", "FAIL", time.time() - t0))
        overall_success = False

    # Print Results
    for item, status, duration in checklist:
        print(f"[{status}] {item:<48} ({duration:.2f}s)")
    print("=" * 65)

    if overall_success:
        print("OVERALL STATUS: ALL CHECKS PASSED (Ready for submission)")
    else:
        print("OVERALL STATUS: SOME CHECKS FAILED")
    return overall_success


if __name__ == "__main__":
    import sys
    ok = run_selfcheck()
    sys.exit(0 if ok else 1)
