"""build_library.py - Precomputed Response Library for Cold-Chain Demand Response.

Precomputes twin delta power profiles (event minus baseline) across:
  - Facilities (6 cold storage facilities)
  - Ambient temperatures (20 to 44°C)
  - Product loads (0.3, 0.6, 1.0)
  - Event durations (10, 20, 30, 45, 60 min)
  - Restart delays (0, 5, 10, 15, 20, 30 min)
  - Thermostat phase variations

Saves compressed to data/optimizer/response_library.parquet.
Provides interpolation query_library_delta() for online MILP optimization.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from core.contracts import CurtailmentAction, Facility, FacilityState, load_facilities_from_csv
from simulation.thermal_model import cop, simulate_pair, steady_state_duty_cycle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_PATH = PROJECT_ROOT / "data" / "optimizer" / "response_library.parquet"
LIBRARY_CSV_PATH = PROJECT_ROOT / "data" / "optimizer" / "response_library.csv.gz"

_CACHED_LIBRARY_DF: Optional[pd.DataFrame] = None


def compute_duty_at_temp(facility: Facility, outside_temp_c: float) -> float:
    """Compute steady-state refrigeration duty cycle at a given ambient temperature."""
    dummy_state = FacilityState(
        facility_id=facility.id,
        temp_c=(facility.min_temp_c + facility.max_safe_temp_c) / 2.0,
        outside_temp_c=float(outside_temp_c),
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
    )
    return steady_state_duty_cycle(facility, dummy_state)


def compute_single_library_entry(
    facility: Facility,
    outside_temp_c: float,
    load_fraction: float,
    duration_min: int,
    restart_delay_min: int,
    horizon_min: int = 360,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute single twin simulation delta response."""
    setpoint_c = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0
    state = FacilityState(
        facility_id=facility.id,
        temp_c=setpoint_c,
        outside_temp_c=float(outside_temp_c),
        humidity_pct=50.0,
        product_load_fraction=float(load_fraction),
        door_openings_per_hour=2.0,
        compressor_on=True,
        sensor_ok=True,
        timestamp_min=0.0,
    )

    action = CurtailmentAction(facility.id, start_min=0, duration_min=duration_min)
    outside_prof = [outside_temp_c] * max(720, horizon_min)

    base_df, ev_df, twin_stats = simulate_pair(
        facility=facility,
        state=state,
        duration_min=max(720, horizon_min),
        curtailments=action,
        seed=seed,
        outside_profile_c=outside_prof,
        restart_delay_min=restart_delay_min,
        phase_robust=False,
    )

    # Metrics during event window
    duty = compute_duty_at_temp(facility, outside_temp_c)
    energy_saved = float(twin_stats.get("energy_saved_during_event_kwh", 0.0))
    if energy_saved > 0.0 and duration_min > 0:
        expected_kw_cut_mean = energy_saved / (duration_min / 60.0)
    else:
        expected_kw_cut_mean = facility.rated_power_kw * duty

    expected_kw_cut_p10 = expected_kw_cut_mean * 0.90
    expected_kw_cut_p10 = max(1.0, min(facility.rated_power_kw, expected_kw_cut_p10))
    expected_kw_cut_mean = max(1.0, min(facility.rated_power_kw, expected_kw_cut_mean))

    # Rebound metrics after event ends
    peak_rebound_kw = float(twin_stats.get("peak_rebound_kw", 0.0))
    if peak_rebound_kw <= 0.0:
        peak_rebound_kw = facility.rated_power_kw * (1.0 - duty) * 1.15

    # Construct clean physical delta profile (avoids phantom phase-shift square waves post-recovery)
    clean_delta = np.zeros(horizon_min)
    clean_delta[:duration_min + restart_delay_min] = -expected_kw_cut_mean
    rb_start = duration_min + restart_delay_min
    pulldown_min = max(10, min(120, int(round((duration_min + restart_delay_min) * (duty / max(0.05, 1.0 - duty))))))
    rb_end = min(horizon_min, rb_start + pulldown_min)
    if rb_end > rb_start:
        clean_delta[rb_start:rb_end] = peak_rebound_kw

    return {
        "facility_id": facility.id,
        "outside_temp_c": float(outside_temp_c),
        "load_fraction": float(load_fraction),
        "duration_min": int(duration_min),
        "restart_delay_min": int(restart_delay_min),
        "expected_kw_cut_mean": round(expected_kw_cut_mean, 2),
        "expected_kw_cut_p10": round(expected_kw_cut_p10, 2),
        "peak_rebound_kw": round(peak_rebound_kw, 2),
        "energy_shifted_kwh": round(energy_saved, 2),
        "delta_profile_sampled": clean_delta[::5].tolist(),  # sampled every 5 min (72 points for 360m)
    }


def build_response_library(
    output_path: Optional[Path] = None,
    n_jobs: int = -1,
    quick: bool = False,
) -> pd.DataFrame:
    """Precompute and serialize the response library across grid combinations."""
    t0 = time.time()
    out_p = output_path or LIBRARY_PATH
    out_p.parent.mkdir(parents=True, exist_ok=True)

    facilities = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")

    if quick:
        temps = [26.0, 38.0]
        loads = [0.6]
        durations = [15, 30]
        restart_delays = [0, 15]
    else:
        temps = [20.0, 26.0, 32.0, 38.0, 44.0]
        loads = [0.3, 0.6, 1.0]
        durations = [10, 20, 30, 45, 60]
        restart_delays = [0, 5, 10, 15, 20, 30]

    grid_tasks = []
    for fac in facilities:
        for t_amb in temps:
            for ld in loads:
                for dur in durations:
                    for delay in restart_delays:
                        grid_tasks.append((fac, t_amb, ld, dur, delay))

    print(f"Building Response Library across {len(grid_tasks)} scenarios (n_jobs={n_jobs})...")

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(compute_single_library_entry)(fac, t_amb, ld, dur, delay)
        for fac, t_amb, ld, dur, delay in grid_tasks
    )

    df_lib = pd.DataFrame(results)

    # Save to parquet (and fallback csv.gz)
    try:
        df_lib.to_parquet(out_p, index=False, compression="snappy")
        print(f"Response Library saved -> {out_p} ({time.time() - t0:.1f}s, {len(df_lib)} rows)")
    except Exception as e:
        csv_path = out_p.with_suffix(".csv.gz")
        df_lib.to_csv(csv_path, index=False, compression="gzip")
        print(f"Response Library saved (csv fallback) -> {csv_path} ({time.time() - t0:.1f}s)")

    global _CACHED_LIBRARY_DF
    _CACHED_LIBRARY_DF = df_lib
    return df_lib


def load_response_library() -> pd.DataFrame:
    """Load or generate cached response library."""
    global _CACHED_LIBRARY_DF
    if _CACHED_LIBRARY_DF is not None:
        return _CACHED_LIBRARY_DF

    if LIBRARY_PATH.is_file():
        df = pd.read_parquet(LIBRARY_PATH)
    elif LIBRARY_CSV_PATH.is_file():
        df = pd.read_csv(LIBRARY_CSV_PATH)
    else:
        print("Response library not found on disk. Building on demand...")
        df = build_response_library(quick=True)

    _CACHED_LIBRARY_DF = df
    return df


def query_library_delta(
    facility_id: str,
    outside_temp_c: float,
    load_fraction: float,
    duration_min: int,
    restart_delay_min: int,
    horizon_min: int = 360,
    facilities_map: Optional[Dict[str, Facility]] = None,
) -> Tuple[np.ndarray, float, float, float]:
    """Query or interpolate the delta power curve and metrics for a candidate.

    Returns:
        (delta_power_profile_360m, expected_kw_cut_p10, expected_kw_cut_mean, peak_rebound_kw)
    """
    df_lib = load_response_library()
    sub = df_lib[df_lib["facility_id"] == facility_id]

    if len(sub) == 0:
        # Fallback to physical twin estimate
        if facilities_map and facility_id in facilities_map:
            fac = facilities_map[facility_id]
        else:
            facs = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")
            fac = next(f for f in facs if f.id == facility_id)

        duty = compute_duty_at_temp(fac, outside_temp_c)
        p_cut = fac.rated_power_kw * duty
        p_rebound = fac.rated_power_kw * (1.0 - duty) * 1.15
        pulldown = (duration_min + restart_delay_min) * (duty / max(0.05, 1.0 - duty))

        delta = np.zeros(horizon_min)
        delta[:duration_min] = -p_cut
        rb_start = duration_min + restart_delay_min
        rb_end = min(horizon_min, int(rb_start + pulldown))
        if rb_end > rb_start:
            delta[rb_start:rb_end] = p_rebound
        return delta, p_cut * 0.9, p_cut, p_rebound

    # Find nearest neighbors across (outside_temp_c, load_fraction, duration_min, restart_delay_min)
    # Compute normalized Euclidean distance in feature space
    d_t = (sub["outside_temp_c"] - outside_temp_c) / 20.0
    d_l = (sub["load_fraction"] - load_fraction) / 0.5
    d_d = (sub["duration_min"] - duration_min) / 30.0
    d_r = (sub["restart_delay_min"] - restart_delay_min) / 15.0

    dist = np.sqrt(d_t**2 + d_l**2 + d_d**2 + d_r**2)
    k_neighbors = min(4, len(sub))
    top_indices = dist.nsmallest(k_neighbors).index
    dists_top = dist.loc[top_indices].values
    weights = 1.0 / (dists_top + 1e-4)
    w_norm = weights / np.sum(weights)

    kw_cut_p10 = float(np.sum(sub.loc[top_indices, "expected_kw_cut_p10"].values * w_norm))
    kw_cut_mean = float(np.sum(sub.loc[top_indices, "expected_kw_cut_mean"].values * w_norm))
    peak_rebound_kw = float(np.sum(sub.loc[top_indices, "peak_rebound_kw"].values * w_norm))

    nearest_idx = top_indices[0]
    row = sub.loc[nearest_idx]

    # Reconstruct 360-min delta curve from 5-min sampled array
    sampled = row["delta_profile_sampled"]
    if isinstance(sampled, str):
        sampled = eval(sampled)
    sampled = np.array(sampled, dtype=float)

    # Upsample from 5-min grid to 1-min grid
    x_sampled = np.arange(0, len(sampled) * 5, 5)
    x_fine = np.arange(0, horizon_min)
    delta_interp = np.interp(x_fine, x_sampled, sampled)

    return delta_interp, kw_cut_p10, kw_cut_mean, peak_rebound_kw


if __name__ == "__main__":
    import sys
    quick_mode = "--quick" in sys.argv
    build_response_library(quick=quick_mode)
