"""demo_part4.py - Part 4 Safety Guardian Live Demonstration.

Exercises all 6 facilities across cool, normal, and warm-phase states under 35 °C ambient,
requesting 30 minutes curtailment. Demonstrates Two Keys resolution, binding constraints,
fail-closed decisions, and actionable retry hints.

All telemetry and outputs are labeled simulated.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from core.contracts import CurtailmentAction, Facility, load_facilities_from_csv
from safety.rules import get_facility_tier
from safety.safety_guardian import get_default_guardian
from safety.types import FleetSnapshot
from simulation.thermal_model import THERMOSTAT_DEADBAND_C


def run_demo_part4() -> pd.DataFrame:
    """Run Part 4 live demonstration across all 6 facilities and 3 phase states."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    cal_csv = data_dir / "facilities_calibrated.csv"
    if not cal_csv.is_file():
        cal_csv = data_dir / "facilities.csv"
    facilities = load_facilities_from_csv(cal_csv)

    guardian = get_default_guardian()

    print("=" * 115)
    print("CHILL2GRID AI - PART 4: SAFETY GUARDIAN DEMONSTRATION")
    print("Operating Conditions: 35.0°C Ambient Outdoor Temperature | Requested Curtailment: 30.0 Minutes")
    print("=" * 115)

    rows = []
    phases = [
        ("Cool (Cut-out)", -THERMOSTAT_DEADBAND_C, False),
        ("Normal (Mid-cycle)", 0.0, True),
        ("Warm (Cut-in)", +THERMOSTAT_DEADBAND_C, True),
    ]

    for fac in facilities:
        tier = get_facility_tier(fac.id)
        setpoint = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0

        for phase_name, offset_c, comp_on in phases:
            curr_temp = setpoint + offset_c
            guardian.debt_ledger.reset()
            snap = FleetSnapshot(
                facility_id=fac.id,
                temp_c=curr_temp,
                outside_temp_c=35.0,
                humidity_pct=60.0,
                product_load_fraction=0.85,
                door_openings_per_hour=2.0,
                compressor_on=comp_on,
                sensor_ok=True,
                timestamp_min=120.0,
                minutes_in_current_state=30.0,
            )
            req = CurtailmentAction(fac.id, start_min=0.0, duration_min=30.0)
            verdict = guardian.evaluate_request(req, snap)

            ev = verdict.evidence
            k1_safe = ev.get("key1_safe_min", 0.0)
            k2_safe = ev.get("key2_safe_min", 0.0)
            retry_hint = verdict.wait_min

            rows.append({
                "Facility ID": fac.id,
                "Tier": tier.tier_name,
                "State": phase_name,
                "Sensor Air (°C)": f"{curr_temp:.2f}",
                "Decision": verdict.decision,
                "Approved": f"{verdict.approved_duration_min:.1f}m",
                "Binding Constraint": verdict.binding_constraint,
                "Key 1": f"{k1_safe:.1f}m",
                "Key 2": f"{k2_safe:.1f}m",
                "Retry Hint": f"{retry_hint:.1f}m" if retry_hint > 0 else "--",
            })

    df_demo = pd.DataFrame(rows)

    # Print formatted ASCII table
    header_fmt = "{:<12} {:<9} {:<18} {:<15} {:<18} {:<10} {:<24} {:<8} {:<8} {:<10}"
    print(header_fmt.format(
        "Facility ID", "Tier", "Phase State", "Sensor Air", "Decision", "Approved", "Binding Constraint", "Key 1", "Key 2", "Retry Hint"
    ))
    print("-" * 125)
    for _, r in df_demo.iterrows():
        print(header_fmt.format(
            r["Facility ID"],
            r["Tier"],
            r["State"],
            r["Sensor Air (°C)"],
            r["Decision"],
            r["Approved"],
            r["Binding Constraint"],
            r["Key 1"],
            r["Key 2"],
            r["Retry Hint"],
        ))
    print("=" * 125)
    print("Demonstration Status: ALL REQUESTS DETERMINISTICALLY GOVERNED (Simulated)\n")

    return df_demo


if __name__ == "__main__":
    run_demo_part4()
