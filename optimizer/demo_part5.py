"""demo_part5.py - Demonstration Script for Part 5 Grid Event Optimizer.

Runs:
  - 60-minute grid curtailment event at 18:00 (minute 1080) on a 35.0°C summer day.
  - Target: 40% of fleet power (~384 kW).
  - Displays: Table of actions, plan power curve metrics, kW cut P10 and mean,
    rebound peak with and without staggering, refusals with documented reasons.
"""

from __future__ import annotations

import time
import pandas as pd

from core.contracts import load_facilities_from_csv
from optimizer.plan_event import plan_grid_event
from optimizer.types import GridEvent
from safety.safety_guardian import get_default_guardian
from safety.types import FleetSnapshot
from optimizer.build_library import PROJECT_ROOT


def run_demo_part5() -> None:
    t0 = time.time()
    print("\n" + "=" * 80)
    print("  PART 5 GRID-EVENT OPTIMIZER: LIVE DEMONSTRATION")
    print("=" * 80)
    print("Scenario: Summer Heat Wave (35.0°C) | Event: 18:00 to 19:00 (min 1080-1140) | Target: 40% fleet power")
    print("-" * 80)

    facilities = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")
    guardian = get_default_guardian()
    guardian.debt_ledger.history.clear()

    fleet_total_kw = sum(f.rated_power_kw for f in facilities)
    target_kw = fleet_total_kw * 0.40

    snapshots = {
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

    event = GridEvent(start_min=1080, duration_min=60, target_kw=target_kw)

    plan = plan_grid_event(
        event=event,
        fleet_state=snapshots,
        ambient_profile_c=35.0,
        guardian=guardian,
        facilities=facilities,
    )

    # 1. Table of Scheduled Actions & Exclusions
    print("\nDISPATCH ACTIONS & GUARDIAN BINDING CONSTRAINTS:")
    rows = []
    for f in facilities:
        act = next((a for a in plan.actions if a.facility_id == f.id), None)
        if act:
            rows.append({
                "Facility": f.id,
                "Product": f.product_type,
                "Status": "APPROVED",
                "Approved Min": act.duration_min,
                "Start Offset": f"{act.start_min - 1080}m",
                "Restart Delay": f"{act.restart_delay_min}m",
                "Cut P10 (kW)": f"{act.expected_kw_cut_p10:.1f}",
                "Cut Mean (kW)": f"{act.expected_kw_cut_mean:.1f}",
                "Reason": act.reason,
            })
        else:
            rows.append({
                "Facility": f.id,
                "Product": f.product_type,
                "Status": "EXCLUDED",
                "Approved Min": 0,
                "Start Offset": "0m",
                "Restart Delay": "0m",
                "Cut P10 (kW)": "0.0",
                "Cut Mean (kW)": "0.0",
                "Reason": plan.reasons.get(f.id, "N/A"),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # 2. Key Metrics
    total_cut_p10 = sum(a.expected_kw_cut_p10 for a in plan.actions)
    total_cut_mean = sum(a.expected_kw_cut_mean for a in plan.actions)

    print("\n" + "-" * 80)
    print("PLAN PERFORMANCE METRICS:")
    print(f"  • Target Reduction:           {target_kw:.1f} kW")
    print(f"  • Expected Fleet Cut (P10):   {total_cut_p10:.1f} kW (Conservative)")
    print(f"  • Expected Fleet Cut (Mean):  {total_cut_mean:.1f} kW")
    print(f"  • Shortfall below Target:     {plan.shortfall_kw:.1f} kW")
    print(f"  • Rebound Peak (Staggered):   {plan.rebound_peak_staggered:.1f} kW (Optimized)")
    print(f"  • Rebound Peak (Unstaggered): {plan.rebound_peak_unstaggered:.1f} kW (No Stagger)")
    print(f"  • Peak Reduction via Stagger: {plan.rebound_peak_unstaggered - plan.rebound_peak_staggered:+.1f} kW")
    print(f"  • Closed-Loop Verification:   Peak Err = {plan.library_vs_exact_error['mean_error_peak_kw']:.2f} kW, Energy Err = {plan.library_vs_exact_error['mean_error_energy_kwh']:.2f} kWh")
    print(f"  • Execution Latency:          {time.time() - t0:.2f} s (< 3.0s target)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo_part5()
