"""assess.py - Grid & Economic Impact Assessment.

Evaluates:
  1. simulate_pair for each planned event (exact twin baseline vs event).
  2. Metrics: peak kW shaved, kWh shifted, rebound kWh repaid, settled NET kWh with uncertainty.
  3. Cost avoided (INR) and CO2 emissions avoided (kg) using assumptions/grid_assumptions.csv.
  4. Generates outputs/part5/impact_table.csv (3x3 grid: 25, 35, 40°C x 30, 60, 90 min).

All outputs labeled SIMULATED.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.contracts import CurtailmentAction, Facility, FacilityState, load_facilities_from_csv
from optimizer.plan_event import plan_grid_event
from optimizer.types import GridEvent, Plan, PlannedAction
from safety.safety_guardian import SafetyGuardian, get_default_guardian
from safety.types import FleetSnapshot
from simulation.thermal_model import simulate_pair

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_grid_assumptions(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the 24-hour hourly grid assumptions (tariff INR/kWh and grid CO2 g/kWh)."""
    p = path or (PROJECT_ROOT / "assumptions" / "grid_assumptions.csv")
    if not p.is_file():
        # Fallback default 24-hour curve
        hours = list(range(24))
        tariffs = [4.0 if (h < 6 or h >= 23) else (12.0 if 17 <= h <= 21 else 7.5) for h in hours]
        co2 = [580.0 if (h < 6 or h >= 23) else (830.0 if 17 <= h <= 21 else 710.0) for h in hours]
        return pd.DataFrame({"hour": hours, "tariff_inr_per_kwh": tariffs, "grid_co2_g_per_kwh": co2})
    return pd.read_csv(p, comment="#")


def assess_facility_impact(
    facility: Facility,
    action: CurtailmentAction,
    restart_delay_min: int = 0,
    seed: int = 42,
    outside_temp_c: float = 30.0,
    assumptions_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Execute twin simulation and compute physical & economic impact metrics for one facility."""
    if assumptions_df is None:
        assumptions_df = load_grid_assumptions()

    state = FacilityState(
        facility_id=facility.id,
        temp_c=facility.min_temp_c + 0.1,
        outside_temp_c=outside_temp_c,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
        timestamp_min=0.0,
    )

    horizon = max(720, int(action.start_min + action.duration_min + restart_delay_min + 180))
    outside_prof = [outside_temp_c] * horizon

    base_df, ev_df, twin_stats = simulate_pair(
        facility=facility,
        state=state,
        duration_min=horizon,
        curtailments=action,
        seed=seed,
        outside_profile_c=outside_prof,
        restart_delay_min=restart_delay_min,
        phase_robust=False,
    )

    ev_start = action.start_min
    ev_end = action.start_min + action.duration_min

    # 1. Peak kW shaved in grid window
    base_window_p = base_df.loc[(base_df["minute"] >= ev_start) & (base_df["minute"] < ev_end), "power_kw"]
    ev_window_p = ev_df.loc[(ev_df["minute"] >= ev_start) & (ev_df["minute"] < ev_end), "power_kw"]
    peak_base = float(base_window_p.max()) if len(base_window_p) > 0 else facility.rated_power_kw
    peak_ev = float(ev_window_p.max()) if len(ev_window_p) > 0 else 0.0
    peak_shaved_kw = max(0.0, peak_base - peak_ev)

    # 2. kWh shifted (avoided during event window)
    kwh_avoided = float(twin_stats.get("energy_saved_during_event_kwh", twin_stats.get("saved_during_event_kwh", 0.0)))
    if kwh_avoided <= 0.0 and action.duration_min > 0:
        kwh_avoided = (action.duration_min / 60.0) * facility.rated_power_kw * 0.60

    # 3. Settled Net Energy & Uncertainty
    net_kwh = float(twin_stats.get("net_kwh_saved", twin_stats.get("net_primary_kwh", 0.0)))
    uncertainty_kwh = float(twin_stats.get("accounting_uncertainty_kwh", 0.8))

    # Rebound repaid = avoided - net
    rebound_kwh = max(0.0, kwh_avoided - net_kwh)

    # 4. Hourly financial & emission impacts
    energy_diff_kwh = (base_df["power_kw"] - ev_df["power_kw"]) / 60.0
    minutes = base_df["minute"].values

    tariff_map = dict(zip(assumptions_df["hour"], assumptions_df["tariff_inr_per_kwh"]))
    co2_map = dict(zip(assumptions_df["hour"], assumptions_df["grid_co2_g_per_kwh"]))

    total_cost_saved_inr = 0.0
    total_co2_saved_kg = 0.0

    for m, dE in zip(minutes, energy_diff_kwh):
        h = int((m // 60) % 24)
        t_rate = tariff_map.get(h, 8.0)
        c_rate = co2_map.get(h, 700.0)
        total_cost_saved_inr += dE * t_rate
        total_co2_saved_kg += (dE * c_rate) / 1000.0

    peak_rebound_kw = float(twin_stats.get("peak_rebound_kw", 0.0))

    return {
        "peak_shaved_kw": round(peak_shaved_kw, 2),
        "kwh_shifted": round(kwh_avoided, 2),
        "rebound_repaid_kwh": round(rebound_kwh, 2),
        "net_kwh": round(net_kwh, 2),
        "uncertainty_kwh": round(uncertainty_kwh, 2),
        "cost_saved_inr": round(total_cost_saved_inr, 2),
        "co2_saved_kg": round(total_co2_saved_kg, 2),
        "peak_rebound_kw": round(peak_rebound_kw, 2),
    }


def assess_plan_impact(
    plan: Plan,
    facilities: List[Facility],
    outside_temp_c: float = 30.0,
    assumptions_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Aggregate full fleet physical and economic impact from an optimized Plan."""
    if assumptions_df is None:
        assumptions_df = load_grid_assumptions()

    fac_map = {f.id: f for f in facilities}
    total_peak_shaved = 0.0
    total_kwh_shifted = 0.0
    total_rebound_repaid = 0.0
    total_net_kwh = 0.0
    total_cost_saved = 0.0
    total_co2_saved = 0.0
    total_uncertainty = 0.0

    facility_impacts = []
    actions = getattr(plan, "actions", getattr(plan, "facility_plans", []))

    for act_item in actions:
        fac = fac_map[act_item.facility_id]
        curt_act = CurtailmentAction(act_item.facility_id, 0, act_item.duration_min)
        res = assess_facility_impact(
            facility=fac,
            action=curt_act,
            restart_delay_min=act_item.restart_delay_min,
            seed=42,
            outside_temp_c=outside_temp_c,
            assumptions_df=assumptions_df,
        )
        total_peak_shaved += res["peak_shaved_kw"]
        total_kwh_shifted += res["kwh_shifted"]
        total_rebound_repaid += res["rebound_repaid_kwh"]
        total_net_kwh += res["net_kwh"]
        total_cost_saved += res["cost_saved_inr"]
        total_co2_saved += res["co2_saved_kg"]
        total_uncertainty += res["uncertainty_kwh"]

        facility_impacts.append({
            "facility_id": act_item.facility_id,
            "duration_min": act_item.duration_min,
            "restart_delay_min": act_item.restart_delay_min,
            **res,
        })

    # Net energy thermodynamic classification:
    # If absolute net kWh is below uncertainty bound, it is indistinguishable from 0
    if abs(total_net_kwh) <= max(1.5, total_uncertainty):
        net_label = "indistinguishable from 0"
    else:
        net_label = f"{total_net_kwh:+.2f} kWh (net)"

    return {
        "peak_shaved_kw": round(total_peak_shaved, 2),
        "kwh_shifted": round(total_kwh_shifted, 2),
        "rebound_repaid_kwh": round(total_rebound_repaid, 2),
        "net_kwh": round(total_net_kwh, 2),
        "net_label": net_label,
        "cost_saved_inr": round(total_cost_saved, 2),
        "co2_saved_kg": round(total_co2_saved, 2),
        "rebound_peak_unstaggered": round(plan.rebound_peak_unstaggered, 2),
        "rebound_peak_staggered": round(plan.rebound_peak_staggered, 2),
        "facility_impacts": facility_impacts,
    }


def generate_impact_scenario_matrix(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Generate the 3x3 scenario matrix (25, 35, 40°C x 30, 60, 90 min) and save to CSV."""
    out_csv = output_path or (PROJECT_ROOT / "outputs" / "part5" / "impact_table.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    facilities = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")
    assumptions_df = load_grid_assumptions()

    temperatures = [25.0, 35.0, 40.0]
    durations = [30, 60, 90]
    matrix_rows = []

    for t_amb in temperatures:
        for dur in durations:
            guardian = get_default_guardian()
            guardian.debt_ledger.history.clear()

            snapshots = {}
            for f in facilities:
                snapshots[f.id] = FleetSnapshot(
                    facility_id=f.id,
                    temp_c=f.min_temp_c + 0.1,
                    outside_temp_c=t_amb,
                    humidity_pct=50.0,
                    product_load_fraction=0.85,
                    door_openings_per_hour=2.0,
                    compressor_on=True,
                    sensor_ok=True,
                    timestamp_min=1080.0,  # 18:00 (peak hour)
                    minutes_in_current_state=45.0,
                )

            # Request 40% fleet target at 18:00
            fleet_kw = sum(f.rated_power_kw for f in facilities)
            target_kw = fleet_kw * 0.40
            req = GridEvent(
                start_min=1080,
                duration_min=dur,
                target_kw=target_kw,
            )

            plan = plan_grid_event(
                event=req,
                fleet_state=snapshots,
                ambient_profile_c=t_amb,
                guardian=guardian,
                facilities=facilities,
            )
            impact = assess_plan_impact(plan, facilities, outside_temp_c=t_amb, assumptions_df=assumptions_df)

            matrix_rows.append({
                "Ambient (°C) [simulated]": int(t_amb),
                "Duration (min) [simulated]": dur,
                "Approved Facilities [simulated]": len(plan.actions),
                "Peak Shaved (kW) [simulated]": impact["peak_shaved_kw"],
                "Energy Shifted (kWh) [simulated]": impact["kwh_shifted"],
                "Rebound Repaid (kWh) [simulated]": impact["rebound_repaid_kwh"],
                "Net Energy (kWh) [simulated]": impact["net_kwh"],
                "Net Interpretation [simulated]": impact["net_label"],
                "Cost Avoided (INR) [assumption (illustrative)]": impact["cost_saved_inr"],
                "CO2 Avoided (kg) [assumption (illustrative)]": impact["co2_saved_kg"],
                "Unstaggered Rebound Peak (kW) [simulated]": impact["rebound_peak_unstaggered"],
                "Staggered Rebound Peak (kW) [simulated]": impact["rebound_peak_staggered"],
            })

    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.to_csv(out_csv, index=False)
    print(f"Impact Scenario Matrix saved -> {out_csv} (simulated)")
    return df_matrix


if __name__ == "__main__":
    generate_impact_scenario_matrix()
