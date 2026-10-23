"""scenario_generator.py - Stochastic Reality-Gap Scenario & Dataset Generator.

Generates realistic cold-storage operational datasets across the 6 calibrated facilities.
Each scenario models the true physical asset with perturbed parameters (reality gap),
ambient weather drift, stochastic loading dock door events, sensor noise, and internal disturbances.

For each decision point ('now'), only observable telemetry is recorded as features.
Ground truth temperature trajectories are obtained by branching the true world state
under a 120-minute forced curtailment command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

from core.contracts import CurtailmentAction, Facility, FacilityState, load_facilities_from_csv
from simulation.reality_gap import make_disturbance, make_world_facility
from simulation.thermal_model import read_sensor, simulate_temperature


# ============================================================================
# HORIZONS & FEATURE CONSTANTS (NO LEAKAGE RULE)
# ============================================================================

FORECAST_HORIZONS: List[int] = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 120]

OBSERVABLE_FEATURES: Tuple[str, ...] = (
    "lag_0",
    "lag_1",
    "lag_2",
    "lag_5",
    "lag_10",
    "lag_15",
    "lag_30",
    "lag_60",
    "slope_5min",
    "slope_15min",
    "min_30min",
    "max_30min",
    "compressor_on",
    "minutes_in_compressor_state",
    "minutes_since_curtailment",
    "outside_temp_c",
    "humidity_pct",
    "product_load_fraction",
    "door_rate",
    "rated_power_kw",
    "setpoint_c",
    "max_safe_temp_c",
    "min_temp_c",
    "thermal_mass",
    "insulation_r",
    "headroom_c",
)

M1_FEATURES: Tuple[str, ...] = OBSERVABLE_FEATURES + ("horizon_min",)
M2_FEATURES: Tuple[str, ...] = (
    OBSERVABLE_FEATURES + ("horizon_min", "physics_temp_c", "physics_minutes_to_limit")
)

FORBIDDEN_FEATURES: set[str] = {
    "true_temp_c",
    "residual_c",
    "true_minutes_to_limit",
    "scenario_id",
    "record_id",
    "facility_id",
    "initial_product_temp_c",
    "disturbance",
    "extra_heat_kw",
}


# ============================================================================
# SINGLE SCENARIO SIMULATION & FEATURE EXTRACTION
# ============================================================================

def generate_single_scenario(
    scenario_id: int,
    nominal_facility: Facility,
    seed: int,
    records_per_scenario: int = 30,
) -> List[Dict[str, object]]:
    """Simulate a single 480-minute scenario in the true world and extract branch records.

    Args:
        scenario_id: Unique integer identifier for the scenario.
        nominal_facility: Calibrated nominal Facility instance.
        seed: Unique random seed for repeatability.
        records_per_scenario: Number of 'now' decision moments to sample (default 30).

    Returns:
        List of long-format row dictionaries.
    """
    rng = np.random.default_rng(seed)

    # 1. Randomize operational & weather boundary conditions
    outside_base_c = float(rng.uniform(18.0, 44.0))
    swing = float(rng.uniform(3.0, 6.0))
    humidity_pct = float(rng.uniform(30.0, 90.0))
    product_load_fraction = float(rng.uniform(0.30, 1.0))
    door_rate = float(rng.uniform(0.0, 12.0))
    sensor_noise_std = float(rng.uniform(0.05, 0.20))

    # Daily outdoor temperature profile over 480 minutes
    outside_profile = [
        round(
            float(
                outside_base_c
                + swing * np.sin(2.0 * np.pi * (m - 120) / 1440.0)
                + rng.normal(0.0, 0.2)
            ),
            2,
        )
        for m in range(480)
    ]

    # Past curtailments (0 to 2 historical events between min 60 and 300)
    past_curtailments: List[CurtailmentAction] = []
    n_past = int(rng.integers(0, 3))
    if n_past > 0:
        c1_start = int(rng.integers(60, 180))
        c1_dur = int(rng.integers(10, 45))
        past_curtailments.append(CurtailmentAction(nominal_facility.id, c1_start, c1_dur))
        if n_past == 2:
            c2_start = int(rng.integers(c1_start + c1_dur + 20, 300))
            c2_dur = int(rng.integers(10, 45))
            past_curtailments.append(CurtailmentAction(nominal_facility.id, c2_start, c2_dur))

    # 2. Perturb facility to create the 'True Physical World'
    world_facility = make_world_facility(nominal_facility, rng)

    setpoint_c = (nominal_facility.min_temp_c + nominal_facility.max_safe_temp_c) / 2.0
    mean_wall_heat = max(
        5.0,
        (outside_base_c - setpoint_c) / nominal_facility.insulation_r_deg_c_per_kw,
    )
    disturbance = make_disturbance(rng, 480, mean_wall_heat)

    # 3. Simulate true full 480-minute world history
    initial_state = FacilityState(
        facility_id=world_facility.id,
        temp_c=setpoint_c,
        outside_temp_c=outside_profile[0],
        humidity_pct=humidity_pct,
        product_load_fraction=product_load_fraction,
        door_openings_per_hour=door_rate,
        compressor_on=True,
        sensor_ok=True,
    )

    true_world_df = simulate_temperature(
        facility=world_facility,
        state=initial_state,
        duration_min=480,
        curtailments=past_curtailments,
        seed=seed,
        outside_profile_c=outside_profile,
        extra_heat_kw=disturbance,
    )

    # Generate noisy sensor air temperature telemetry
    true_air_series = true_world_df["air_temp_c"].values
    sensor_telemetry = [
        read_sensor(t, rng=rng, noise_std_c=sensor_noise_std)
        for t in true_air_series
    ]

    # 4. Sample decision times 'now' between minute 180 and 360
    now_times = rng.choice(np.arange(180, 360), size=records_per_scenario, replace=False)
    now_times.sort()

    rows: List[Dict[str, object]] = []

    for rec_idx, now_min in enumerate(now_times):
        # A. Extract observable features (what the operator actually sees)
        sensor_now = sensor_telemetry[now_min]
        lag_0 = sensor_now
        lag_1 = sensor_telemetry[now_min - 1]
        lag_2 = sensor_telemetry[now_min - 2]
        lag_5 = sensor_telemetry[now_min - 5]
        lag_10 = sensor_telemetry[now_min - 10]
        lag_15 = sensor_telemetry[now_min - 15]
        lag_30 = sensor_telemetry[now_min - 30]
        lag_60 = sensor_telemetry[now_min - 60]

        slope_5min = (lag_0 - lag_5) / 5.0
        slope_15min = (lag_0 - lag_15) / 15.0

        past_30_window = sensor_telemetry[now_min - 30 : now_min + 1]
        min_30min = min(past_30_window)
        max_30min = max(past_30_window)

        comp_on_val = 1.0 if true_world_df.loc[now_min, "compressor_on"] else 0.0

        # Minutes in current compressor state
        mins_in_state = 0
        for back_m in range(now_min, -1, -1):
            if true_world_df.loc[back_m, "compressor_on"] == bool(comp_on_val):
                mins_in_state += 1
            else:
                break

        # Minutes since last curtailment ended
        last_curt_m = -999
        for back_m in range(now_min, -1, -1):
            if true_world_df.loc[back_m, "forced_off"]:
                last_curt_m = back_m
                break
        mins_since_curt = float(now_min - last_curt_m) if last_curt_m >= 0 else 600.0

        outside_now = outside_profile[now_min]
        headroom_c = nominal_facility.max_safe_temp_c - sensor_now

        # Pack observable feature map
        feat_map = {
            "lag_0": round(lag_0, 3),
            "lag_1": round(lag_1, 3),
            "lag_2": round(lag_2, 3),
            "lag_5": round(lag_5, 3),
            "lag_10": round(lag_10, 3),
            "lag_15": round(lag_15, 3),
            "lag_30": round(lag_30, 3),
            "lag_60": round(lag_60, 3),
            "slope_5min": round(slope_5min, 4),
            "slope_15min": round(slope_15min, 4),
            "min_30min": round(min_30min, 3),
            "max_30min": round(max_30min, 3),
            "compressor_on": comp_on_val,
            "minutes_in_compressor_state": mins_in_state,
            "minutes_since_curtailment": mins_since_curt,
            "outside_temp_c": round(outside_now, 2),
            "humidity_pct": round(humidity_pct, 1),
            "product_load_fraction": round(product_load_fraction, 2),
            "door_rate": round(door_rate, 1),
            "rated_power_kw": nominal_facility.rated_power_kw,
            "setpoint_c": round(setpoint_c, 2),
            "max_safe_temp_c": nominal_facility.max_safe_temp_c,
            "min_temp_c": nominal_facility.min_temp_c,
            "thermal_mass": nominal_facility.thermal_mass_kwh_per_c,
            "insulation_r": nominal_facility.insulation_r_deg_c_per_kw,
            "headroom_c": round(headroom_c, 3),
        }

        # B. Branch TRUE world forward 120 minutes under forced curtailment
        branch_curtail = CurtailmentAction(world_facility.id, start_min=0, duration_min=120)
        true_air_now = true_world_df.loc[now_min, "air_temp_c"]
        true_prod_now = true_world_df.loc[now_min, "product_temp_c"]
        branch_outside = outside_profile[now_min : now_min + 120]
        branch_dist = disturbance[now_min : now_min + 120]

        branch_state = FacilityState(
            facility_id=world_facility.id,
            temp_c=true_air_now,
            outside_temp_c=branch_outside[0],
            humidity_pct=humidity_pct,
            product_load_fraction=product_load_fraction,
            door_openings_per_hour=door_rate,
            compressor_on=False,
            sensor_ok=True,
        )

        true_branch_df = simulate_temperature(
            facility=world_facility,
            state=branch_state,
            duration_min=120,
            curtailments=[branch_curtail],
            seed=seed + 1000 + rec_idx,
            outside_profile_c=branch_outside,
            extra_heat_kw=branch_dist,
            initial_air_temp_c=true_air_now,
            initial_product_temp_c=true_prod_now,
            initial_compressor_on=False,
            initial_minutes_in_state=0,
        )

        # First minute true temperature breaches max safe temp
        breached_true = true_branch_df["air_temp_c"] >= nominal_facility.max_safe_temp_c
        if breached_true.any():
            true_min_to_limit = float(true_branch_df.loc[breached_true, "minute"].iloc[0])
        else:
            true_min_to_limit = 120.0

        # C. Compute physics model prediction from nominal beliefs
        # Naive state guess: air = product = sensor_now (no hidden sensor access)
        physics_state = FacilityState(
            facility_id=nominal_facility.id,
            temp_c=sensor_now,
            outside_temp_c=branch_outside[0],
            humidity_pct=humidity_pct,
            product_load_fraction=product_load_fraction,
            door_openings_per_hour=door_rate,
            compressor_on=False,
            sensor_ok=True,
        )
        physics_branch_df = simulate_temperature(
            facility=nominal_facility,
            state=physics_state,
            duration_min=120,
            curtailments=[branch_curtail],
            seed=0,
            outside_profile_c=branch_outside,
            expected_door_heat=True,
            extra_heat_kw=None,
            initial_air_temp_c=sensor_now,
            initial_product_temp_c=sensor_now,
            initial_compressor_on=False,
            initial_minutes_in_state=0,
        )

        breached_phys = physics_branch_df["air_temp_c"] >= nominal_facility.max_safe_temp_c
        if breached_phys.any():
            physics_min_to_limit = float(physics_branch_df.loc[breached_phys, "minute"].iloc[0])
        else:
            physics_min_to_limit = 120.0

        # D. Assemble long-format records across all evaluation horizons
        for h in FORECAST_HORIZONS:
            step_idx = h - 1
            true_t = round(float(true_branch_df.loc[step_idx, "air_temp_c"]), 3)
            phys_t = round(float(physics_branch_df.loc[step_idx, "air_temp_c"]), 3)
            res_t = round(float(true_t - phys_t), 3)

            row_data = {
                "scenario_id": scenario_id,
                "record_id": rec_idx,
                "facility_id": nominal_facility.id,
                "outside_base_c": outside_base_c,
                "horizon_min": h,
                "true_temp_c": true_t,
                "physics_temp_c": phys_t,
                "residual_c": res_t,
                "true_minutes_to_limit": true_min_to_limit,
                "physics_minutes_to_limit": physics_min_to_limit,
            }
            row_data.update(feat_map)
            rows.append(row_data)

    return rows


# ============================================================================
# DATASET GENERATION PIPELINE
# ============================================================================

def generate_full_dataset(
    calibrated_facilities: List[Facility],
    n_scenarios: int = 500,
    records_per_scenario: int = 30,
    base_seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Generate multi-facility reality-gap dataset in parallel using joblib.

    Args:
        calibrated_facilities: List of calibrated Facility objects.
        n_scenarios: Total number of scenarios to simulate across the fleet.
        records_per_scenario: Decision points per scenario (default 30).
        base_seed: Master seed for exact dataset reproducibility.
        n_jobs: Number of parallel worker cores (default -1 = all cores).

    Returns:
        DataFrame in long format containing all scenario records across horizons.
    """
    n_fac = len(calibrated_facilities)
    jobs = []
    for sc_id in range(n_scenarios):
        fac = calibrated_facilities[sc_id % n_fac]
        seed = base_seed + sc_id * 100
        jobs.append((sc_id, fac, seed, records_per_scenario))

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(generate_single_scenario)(sc_id, fac, seed, recs)
        for sc_id, fac, seed, recs in jobs
    )

    all_rows = [row for scenario_rows in results for row in scenario_rows]
    df = pd.DataFrame(all_rows)
    return df


def split_dataset_by_scenario(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    calib_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset BY SCENARIO (never by row), stratified across facilities.

    Guarantees that no scenario appears in multiple splits.

    Returns:
        (train_df, calib_df, test_df)
    """
    rng = np.random.default_rng(seed)
    train_scenarios = []
    calib_scenarios = []
    test_scenarios = []

    # Stratify scenario IDs by facility
    facilities = df["facility_id"].unique()
    for fac_id in facilities:
        fac_scenarios = df.loc[df["facility_id"] == fac_id, "scenario_id"].unique()
        rng.shuffle(fac_scenarios)
        n_tot = len(fac_scenarios)
        n_tr = int(round(train_frac * n_tot))
        n_cal = int(round(calib_frac * n_tot))

        train_scenarios.extend(fac_scenarios[:n_tr])
        calib_scenarios.extend(fac_scenarios[n_tr : n_tr + n_cal])
        test_scenarios.extend(fac_scenarios[n_tr + n_cal :])

    train_set = set(train_scenarios)
    calib_set = set(calib_scenarios)
    test_set = set(test_scenarios)

    # Disjointness assertions
    assert train_set.isdisjoint(calib_set), "Train and Calibration splits overlap!"
    assert train_set.isdisjoint(test_set), "Train and Test splits overlap!"
    assert calib_set.isdisjoint(test_set), "Calibration and Test splits overlap!"

    train_df = df[df["scenario_id"].isin(train_set)].copy()
    calib_df = df[df["scenario_id"].isin(calib_set)].copy()
    test_df = df[df["scenario_id"].isin(test_set)].copy()

    return train_df, calib_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Chill2Grid reality-gap dataset.")
    parser.add_argument("--quick", action="store_true", help="Generate small fast dataset (60 scenarios)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cal_csv = project_root / "data" / "facilities_calibrated.csv"
    if not cal_csv.is_file():
        # Fall back to base catalog if calibrated has not been saved yet
        cal_csv = project_root / "data" / "facilities.csv"

    facilities = load_facilities_from_csv(cal_csv)
    n_sc = 60 if args.quick else 500

    print(f"Generating {n_sc} scenarios ({'quick mode' if args.quick else 'full mode'}) across {len(facilities)} facilities...")
    df = generate_full_dataset(facilities, n_scenarios=n_sc, n_jobs=-1)
    print(f"Generated {len(df)} total rows across {len(FORECAST_HORIZONS)} horizons.")

    # Save a 20k-row sample to data/training/sample.csv
    train_dir = project_root / "data" / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    sample_df = df.sample(n=min(20000, len(df)), random_state=42)
    sample_path = train_dir / "sample.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Saved {len(sample_df)} sample rows to: {sample_path}")

    # Save config
    config = {
        "n_scenarios": n_sc,
        "total_rows": len(df),
        "horizons": FORECAST_HORIZONS,
        "n_observable_features": len(OBSERVABLE_FEATURES),
        "base_seed": 42,
    }
    with open(train_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
