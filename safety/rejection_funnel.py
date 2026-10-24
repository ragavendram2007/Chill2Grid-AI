"""rejection_funnel.py - Rejection Funnel Analysis for Safety Guardian Decisions.

Analyzes curtailment requests across:
  - Risk Tiers: CRITICAL, HIGH, STANDARD
  - Ambient Bins: 20-26, 26-32, 32-38, 38-44 (°C)
  - Thermodynamic Phases: cut_in, mid_cycle, cut_out

Measures the exact share of requests killed or restricted by each gate and key:
  - G0: Sanity checks (maintenance hold, duration <= 0)
  - G1: Telemetry faults / stuck sensor / dropouts / impossible readings
  - G2: Equipment interlocks (min_on_min lockout, min_off_min anti-short-cycle)
  - G3: Thermal headroom (already over limit, ambient heat tripwire)
  - G4: Capacity headroom (steady-state duty cycle >= 90%, recovery cap)
  - G5: Thermal debt ledger (inter-event cooldown, 24h budget)
  - Key 1: ML conformal safe window below minimum actionable window
  - Key 2: Physics twin safe window below minimum actionable window
  - Reserve: Dynamic restart overshoot subtraction pushing window below minimum actionable window
  - Window under min: Other constraints pushing window below minimum actionable window
  - Tier Cap: Approved, but shortened by tier single-event maximum duration
  - Approved: Approved for full requested duration

All shares across terminal categories sum to exactly 1.0 (100% of requests accounted for).
All outputs and telemetry are labeled simulated.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.contracts import CurtailmentAction, Facility
from safety.key2_physics import compute_restart_overshoot_min, simulate_key2_worst_case
from safety.rules import FACILITY_TIER_MAP, RISK_TIERS, RiskTierConfig
from safety.safety_guardian import SafetyGuardian, get_default_guardian
from safety.types import FleetSnapshot
from simulation.thermal_model import steady_state_duty_cycle


AMBIENT_BINS = ["20-26", "26-32", "32-38", "38-44"]
PHASES = ["cut_in", "mid_cycle", "cut_out"]
TIERS = ["CRITICAL", "HIGH", "STANDARD"]

FUNNEL_SHARE_COLUMNS = [
    "share_g0",
    "share_g1",
    "share_g2",
    "share_g3",
    "share_g4",
    "share_g5",
    "share_key1",
    "share_key2",
    "share_reserve",
    "share_window_under_min",
    "share_tier_cap",
    "share_approved",
]


def assign_ambient_bin(outside_temp_c: float) -> str:
    """Bin outside temperature into standard 6°C evaluation intervals."""
    if outside_temp_c < 26.0:
        return "20-26"
    elif outside_temp_c < 32.0:
        return "26-32"
    elif outside_temp_c < 38.0:
        return "32-38"
    else:
        return "38-44"


def evaluate_funnel_disposition(
    guardian: SafetyGuardian,
    facility: Facility,
    tier: RiskTierConfig,
    snapshot: FleetSnapshot,
    requested_duration_min: float = 60.0,
) -> str:
    """Evaluate a single snapshot through the guardian pipeline and classify its disposition.

    Terminal Dispositions:
      - 'G0': Sanity / duration <= 0
      - 'G1': Telemetry fault / sensor dropout / stuck / jump
      - 'G2': Equipment min_on / min_off interlock
      - 'G3': Thermal state (already over limit / tripwire lockout)
      - 'G4': Capacity limited (duty >= 90% or recovery exceeds cap)
      - 'G5': Thermal debt ledger cooldown or budget exhausted
      - 'Key1': Passed G0-G5, but Key 1 safe window < min_actionable_window_min
      - 'Key2': Passed G0-G5, Key 1 >= min_window, but Key 2 safe window < min_actionable_window_min
      - 'reserve': Passed G0-G5, Key 1 & Key 2 >= min_window, but window after reserve < min_window
      - 'window_under_min': Window < min_actionable_window_min after other constraints
      - 'tier_cap': Approved, but capped by tier.max_single_curtailment_min
      - 'approved': Approved for full requested duration
    """
    req = CurtailmentAction(facility.id, start_min=0, duration_min=requested_duration_min)

    # Gate 0
    if req.duration_min <= 0:
        return "G0"

    # Gate 1
    if not snapshot.sensor_ok or snapshot.n_missing_last_30 > 5 or snapshot.max_gap_last_30 > 5:
        return "G1"
    if snapshot.temp_c is None or math.isnan(snapshot.temp_c) or snapshot.temp_c < -60.0 or snapshot.temp_c > 80.0:
        return "G1"
    if snapshot.recent_temperatures and len(snapshot.recent_temperatures) >= 15:
        if float(np.var(snapshot.recent_temperatures[-15:])) < 1e-4:
            return "G1"
    if snapshot.recent_temperatures and len(snapshot.recent_temperatures) >= 2:
        if np.any(np.abs(np.diff(snapshot.recent_temperatures[-5:])) > 2.0):
            return "G1"

    # Gate 2
    if snapshot.compressor_on and snapshot.minutes_in_current_state < facility.min_on_min:
        return "G2"
    if not snapshot.compressor_on and snapshot.minutes_in_current_state < facility.min_off_min:
        return "G2"

    # Gate 3
    limit_threshold = facility.max_safe_temp_c - tier.guard_margin_c
    if snapshot.temp_c >= limit_threshold:
        return "G3"
    if tier.tripwire_lockout_ambient_c is not None and snapshot.outside_temp_c >= tier.tripwire_lockout_ambient_c:
        return "G3"

    # Gate 4
    duty = steady_state_duty_cycle(facility, snapshot)
    if duty >= tier.duty_cycle_max_threshold:
        return "G4"

    # Gate 5
    g5_allowed, g5_budget_dur, _, _ = guardian.debt_ledger.evaluate_gate5(
        facility, tier, snapshot.timestamp_min, req.duration_min
    )
    if not g5_allowed:
        return "G5"

    # Passed G0 to G5 -> Evaluate Keys & Margins
    # Detect phase
    setpoint = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0
    from simulation.thermal_model import THERMOSTAT_DEADBAND_C
    upper_band = setpoint + THERMOSTAT_DEADBAND_C
    phase_tag = (
        "cut_in"
        if snapshot.temp_c >= upper_band - 0.20
        else "cut_out"
        if snapshot.temp_c <= setpoint - THERMOSTAT_DEADBAND_C + 0.20
        else "mid_cycle"
    )

    is_ood, _ = guardian.ood_guard.is_ood(
        features={"lag_0": snapshot.temp_c, "outside_temp_c": snapshot.outside_temp_c}
    )

    # Key 2
    lam = 1.0 if is_ood else tier.calibrated_lambda
    k2_safe = simulate_key2_worst_case(facility, snapshot, tier, lambda_pessimism=lam, is_ood=is_ood)

    # Key 1
    if is_ood:
        k1_safe = k2_safe
    else:
        k1_safe = guardian._predict_key1_safe_window(facility, snapshot, tier, phase_tag)

    min_window = tier.min_actionable_window_min

    # Key 1 check
    if k1_safe < min_window and k1_safe <= k2_safe:
        return "Key1"

    # Key 2 check
    if k2_safe < min_window and k2_safe < k1_safe:
        return "Key2"

    raw_window = min(k1_safe, k2_safe)
    if raw_window < min_window:
        return "Key2" if k2_safe < k1_safe else "Key1"

    # Dynamic restart reserve
    reserve_min, _ = compute_restart_overshoot_min(
        facility, snapshot.outside_temp_c, tier, lambda_pessimism=lam
    )
    after_reserve = max(0.0, raw_window - reserve_min)

    if after_reserve < min_window:
        return "reserve"

    # Capacity recovery duration cap
    from simulation.thermal_model import cop
    cop_val = cop(snapshot.outside_temp_c)
    cooling_rated_th_kw = facility.rated_power_kw * cop_val
    cooling_spare_th_kw = max(1.0, cooling_rated_th_kw * (1.0 - duty))
    max_recoverable_th_kwh = (tier.max_recovery_min / 60.0) * cooling_spare_th_kw
    max_duration_by_recovery = (max_recoverable_th_kwh / cooling_rated_th_kw) * 60.0

    allowed_duration = min(
        req.duration_min,
        after_reserve,
        tier.max_single_curtailment_min,
        max_duration_by_recovery,
        g5_budget_dur,
    )

    if allowed_duration < min_window:
        return "window_under_min"

    # Tier cap check
    if allowed_duration < req.duration_min and allowed_duration == tier.max_single_curtailment_min:
        return "tier_cap"

    return "approved"


def compute_rejection_funnel(
    df_data: pd.DataFrame,
    guardian: Optional[SafetyGuardian] = None,
    requested_duration_min: float = 60.0,
) -> pd.DataFrame:
    """Compute the rejection funnel table across tiers, ambient bins, and phases.

    Returns a DataFrame with 36 rows and shares summing to 1.0 per row.
    """
    if guardian is None:
        guardian = get_default_guardian()

    # Extract unique decision points (horizon == 5 is first row of record)
    if "horizon_min" in df_data.columns:
        rec_df = df_data[df_data["horizon_min"] == 5].copy()
    else:
        rec_df = df_data.copy()

    rec_df["tier"] = rec_df["facility_id"].map(FACILITY_TIER_MAP)
    rec_df["ambient_bin"] = rec_df["outside_temp_c"].apply(assign_ambient_bin)
    if "phase_tag" in rec_df.columns:
        rec_df["phase"] = rec_df["phase_tag"]
    elif "phase" not in rec_df.columns:
        rec_df["phase"] = "mid_cycle"

    # Evaluate dispositions
    dispositions = []
    for _, row in rec_df.iterrows():
        fac = guardian.facilities_map[row["facility_id"]]
        tier = RISK_TIERS[row["tier"]]
        snap = FleetSnapshot(
            facility_id=fac.id,
            temp_c=float(row["lag_0"]),
            outside_temp_c=float(row["outside_temp_c"]),
            humidity_pct=float(row["humidity_pct"]),
            product_load_fraction=float(row["product_load_fraction"]),
            door_openings_per_hour=float(row["door_rate"]),
            compressor_on=bool(row["compressor_on"]),
            sensor_ok=True,
            minutes_in_current_state=float(row["minutes_in_compressor_state"]),
            timestamp_min=0.0,
        )
        disp = evaluate_funnel_disposition(
            guardian, fac, tier, snap, requested_duration_min=requested_duration_min
        )
        dispositions.append(disp)

    rec_df["disposition"] = dispositions

    # Aggregate by (tier, ambient_bin, phase)
    funnel_rows = []
    for tier_name in TIERS:
        for amb_bin in AMBIENT_BINS:
            for ph in PHASES:
                sub = rec_df[
                    (rec_df["tier"] == tier_name)
                    & (rec_df["ambient_bin"] == amb_bin)
                    & (rec_df["phase"] == ph)
                ]
                n_total = len(sub)
                if n_total == 0:
                    counts = {col: 0.0 for col in FUNNEL_SHARE_COLUMNS}
                    row_dict = {
                        "tier": tier_name,
                        "ambient_bin": amb_bin,
                        "phase": ph,
                        "total_requests": 0,
                    }
                    row_dict.update(counts)
                    funnel_rows.append(row_dict)
                    continue

                vc = sub["disposition"].value_counts().to_dict()

                s_g0 = vc.get("G0", 0) / n_total
                s_g1 = vc.get("G1", 0) / n_total
                s_g2 = vc.get("G2", 0) / n_total
                s_g3 = vc.get("G3", 0) / n_total
                s_g4 = vc.get("G4", 0) / n_total
                s_g5 = vc.get("G5", 0) / n_total
                s_k1 = vc.get("Key1", 0) / n_total
                s_k2 = vc.get("Key2", 0) / n_total
                s_res = vc.get("reserve", 0) / n_total
                s_win = vc.get("window_under_min", 0) / n_total
                s_cap = vc.get("tier_cap", 0) / n_total
                s_app = vc.get("approved", 0) / n_total

                # Normalize rounding so that shares sum to exactly 1.0
                shares_arr = np.array([
                    s_g0, s_g1, s_g2, s_g3, s_g4, s_g5,
                    s_k1, s_k2, s_res, s_win, s_cap, s_app,
                ])
                shares_rounded = np.round(shares_arr, 4)
                diff = round(1.0 - float(np.sum(shares_rounded)), 4)
                if abs(diff) > 1e-6:
                    # Adjust largest non-zero share to guarantee sum == 1.0 exactly
                    max_idx = int(np.argmax(shares_rounded))
                    shares_rounded[max_idx] = round(shares_rounded[max_idx] + diff, 4)

                funnel_rows.append({
                    "tier": tier_name,
                    "ambient_bin": amb_bin,
                    "phase": ph,
                    "total_requests": n_total,
                    "share_g0": shares_rounded[0],
                    "share_g1": shares_rounded[1],
                    "share_g2": shares_rounded[2],
                    "share_g3": shares_rounded[3],
                    "share_g4": shares_rounded[4],
                    "share_g5": shares_rounded[5],
                    "share_key1": shares_rounded[6],
                    "share_key2": shares_rounded[7],
                    "share_reserve": shares_rounded[8],
                    "share_window_under_min": shares_rounded[9],
                    "share_tier_cap": shares_rounded[10],
                    "share_approved": shares_rounded[11],
                })

    return pd.DataFrame(funnel_rows)


def run_and_save_funnel(
    data_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run rejection funnel pipeline and save CSV."""
    project_root = Path(__file__).resolve().parent.parent
    data_path = data_path or (project_root / "data" / "safety" / "calib_extra.csv")
    output_path = output_path or (project_root / "outputs" / "part4" / "rejection_funnel.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {data_path} for rejection funnel analysis...")
    df_calib = pd.read_csv(data_path)
    guardian = get_default_guardian()

    print("Computing rejection funnel across 36 cells (tiers x ambient bins x phases)...")
    funnel_df = compute_rejection_funnel(df_calib, guardian=guardian)
    funnel_df.to_csv(output_path, index=False)
    print(f"Rejection funnel saved -> {output_path} (simulated)")

    return funnel_df


if __name__ == "__main__":
    run_and_save_funnel()
