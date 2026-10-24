"""temperature_predictor.py - Inference Engine for Chill2Grid AI.

Provides high-speed, non-leaking temperature trajectory forecasting and conservative
safe curtailment window estimation using the trained hybrid AI and split-conformal bounds.
Includes fail-safe sensor health diagnostics and batch inference for the real-time VPP dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from core.contracts import CurtailmentAction, Facility, FacilityState
from simulation.thermal_model import THERMOSTAT_DEADBAND_C, simulate_temperature
from training.scenario_generator import FORECAST_HORIZONS, M1_FEATURES, M2_FEATURES, OBSERVABLE_FEATURES


# Global in-memory artifact cache
_PREDICTOR_CACHE: Optional[Dict[str, object]] = None

CONFIDENCE_LEVELS: List[float] = [0.50, 0.80, 0.90, 0.95, 0.99]


@dataclass
class TemperatureForecast:
    """Multi-horizon trajectory forecast for an unpowered facility."""

    horizons_min: List[int]
    physics_c: List[float]
    mean_c: List[float]
    upper_c: Dict[float, List[float]]  # confidence level -> upper bound trajectory
    valid: bool = True
    reason: str = "Nominal trajectory prediction"


# ============================================================================
# ARTIFACT LOADING & VALIDATION
# ============================================================================

def load_predictor(artifacts_dir: Optional[str | Path] = None) -> Dict[str, object]:
    """Load model artifacts and metadata into memory with verification and caching.

    Raises:
        FileNotFoundError: If model artifacts have not been trained yet.
        ValueError: If metadata feature definitions do not match expectations.
    """
    global _PREDICTOR_CACHE
    if _PREDICTOR_CACHE is not None and artifacts_dir is None:
        return _PREDICTOR_CACHE

    if artifacts_dir is None:
        art_dir = Path(__file__).resolve().parent / "artifacts"
    else:
        art_dir = Path(artifacts_dir)

    m1_path = art_dir / "m1_ml_only.joblib"
    m2_path = art_dir / "m2_hybrid.joblib"
    m3_path = art_dir / "m3_quantile_models.joblib"
    meta_path = art_dir / "metadata.json"
    conf_path = art_dir / "conformal_offsets.json"

    if not (m1_path.is_file() and m2_path.is_file() and m3_path.is_file() and meta_path.is_file()):
        raise FileNotFoundError(
            f"Trained model artifacts not found at: {art_dir.resolve()}.\n"
            f"Please run the training command first: python -m training.train_model --quick"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Validate feature consistency
    expected_m2 = list(M2_FEATURES)
    if metadata.get("m2_features") != expected_m2:
        raise ValueError(
            f"Metadata feature mismatch in {meta_path.name}. "
            f"Expected {len(expected_m2)} features, found {len(metadata.get('m2_features', []))}."
        )

    with open(conf_path, "r", encoding="utf-8") as f:
        conformal_offsets = json.load(f)

    # Convert string keys to floats and ints
    cleaned_offsets: Dict[float, Dict[int, float]] = {}
    phase_offsets: Dict[str, Dict[float, Dict[int, float]]] = {}

    if "pooled" in conformal_offsets and "phase" in conformal_offsets:
        for lvl_str, h_dict in conformal_offsets["pooled"].items():
            lvl = float(lvl_str)
            cleaned_offsets[lvl] = {int(h): float(val) for h, val in h_dict.items()}
        for ph_str, lvl_dict in conformal_offsets["phase"].items():
            phase_offsets[ph_str] = {}
            for lvl_str, h_dict in lvl_dict.items():
                lvl = float(lvl_str)
                phase_offsets[ph_str][lvl] = {int(h): float(val) for h, val in h_dict.items()}
    else:
        for lvl_str, h_dict in conformal_offsets.items():
            try:
                lvl = float(lvl_str)
                cleaned_offsets[lvl] = {int(h): float(val) for h, val in h_dict.items()}
            except (ValueError, TypeError):
                continue

    m1_mod = joblib.load(m1_path)
    m2_mod = joblib.load(m2_path)
    m1_mod.set_params(n_jobs=1)
    m2_mod.set_params(n_jobs=1)

    _PREDICTOR_CACHE = {
        "m1_model": m1_mod,
        "m2_model": m2_mod,
        "m3_models": joblib.load(m3_path),
        "conformal_offsets": cleaned_offsets,
        "phase_conformal_offsets": phase_offsets,
        "metadata": metadata,
    }
    return _PREDICTOR_CACHE


# ============================================================================
# FEATURE EXTRACTION FROM TELEMETRY
# ============================================================================

def build_features_from_telemetry(
    facility: Facility,
    state: FacilityState,
    sensor_history_c: List[Optional[float]],
    minutes_in_compressor_state: int,
    minutes_since_curtailment: float,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Construct observable feature vector from operational state and sensor history.

    Enforces fail-safe sensor checks:
      - Validates history length >= 60.
      - Checks for sensor dropouts (>5 consecutive None readings).
    """
    if len(sensor_history_c) < 60:
        return None, f"Insufficient sensor history: {len(sensor_history_c)} readings provided (minimum 60 required)."

    # Check consecutive dropouts
    consecutive_none = 0
    max_consecutive_none = 0
    clean_history: List[float] = []

    for idx, val in enumerate(sensor_history_c):
        if val is None:
            consecutive_none += 1
            max_consecutive_none = max(max_consecutive_none, consecutive_none)
            # Impute single short dropouts with forward fill if possible
            if clean_history:
                clean_history.append(clean_history[-1])
            else:
                clean_history.append(state.temp_c)
        else:
            consecutive_none = 0
            clean_history.append(float(val))

    if max_consecutive_none > 5:
        return None, f"Sensor dropout fault: {max_consecutive_none} consecutive missing readings detected."

    # Lags from end of clean history
    lag_0 = clean_history[-1]
    lag_1 = clean_history[-2]
    lag_2 = clean_history[-3]
    lag_5 = clean_history[-6]
    lag_10 = clean_history[-11]
    lag_15 = clean_history[-16]
    lag_30 = clean_history[-31]
    lag_60 = clean_history[-61] if len(clean_history) > 60 else clean_history[0]

    slope_5 = (lag_0 - lag_5) / 5.0
    slope_15 = (lag_0 - lag_15) / 15.0

    past_30 = clean_history[-31:]
    min_30 = min(past_30)
    max_30 = max(past_30)

    setpoint_c = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0
    headroom_c = facility.max_safe_temp_c - lag_0

    # Missing telemetry accounting in last 30 readings
    past_raw_30 = sensor_history_c[-31:]
    n_missing_last_30 = sum(1 for v in past_raw_30 if v is None)
    max_gap_30 = 0
    cur_gap = 0
    for v in past_raw_30:
        if v is None:
            cur_gap += 1
            if cur_gap > max_gap_30:
                max_gap_30 = cur_gap
        else:
            cur_gap = 0
    max_gap_last_30 = max_gap_30

    temp_over_deadband = (lag_0 - setpoint_c) / THERMOSTAT_DEADBAND_C

    feat_dict = {
        "lag_0": lag_0,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_5": lag_5,
        "lag_10": lag_10,
        "lag_15": lag_15,
        "lag_30": lag_30,
        "lag_60": lag_60,
        "slope_5min": slope_5,
        "slope_15min": slope_15,
        "min_30min": min_30,
        "max_30min": max_30,
        "compressor_on": 1.0 if state.compressor_on else 0.0,
        "minutes_in_compressor_state": float(minutes_in_compressor_state),
        "minutes_since_curtailment": float(minutes_since_curtailment),
        "outside_temp_c": float(state.outside_temp_c),
        "humidity_pct": float(state.humidity_pct),
        "product_load_fraction": float(state.product_load_fraction),
        "door_rate": float(state.door_openings_per_hour),
        "rated_power_kw": float(facility.rated_power_kw),
        "setpoint_c": setpoint_c,
        "max_safe_temp_c": float(facility.max_safe_temp_c),
        "min_temp_c": float(facility.min_temp_c),
        "thermal_mass": float(facility.thermal_mass_kwh_per_c),
        "insulation_r": float(facility.insulation_r_deg_c_per_kw),
        "headroom_c": headroom_c,
        "(sensor_air - setpoint)/deadband": float(temp_over_deadband),
        "n_missing_last_30": int(n_missing_last_30),
        "max_gap_last_30": int(max_gap_last_30),
    }
    return feat_dict, None


# ============================================================================
# PRIMARY INFERENCE API
# ============================================================================

def predict_trajectory(
    facility: Facility,
    state: FacilityState,
    sensor_history_c: Optional[List[Optional[float]]],
    minutes_in_compressor_state: int = 20,
    minutes_since_curtailment: float = 600.0,
    horizons: Optional[List[int]] = None,
) -> TemperatureForecast:
    """Generate multi-horizon temperature forecasts and conservative prediction intervals.

    Args:
        facility: Calibrated Facility configuration.
        state: Current telemetry state.
        sensor_history_c: At least 60 per-minute past sensor readings (newest last).
        minutes_in_compressor_state: Anti-cycling tracker duration.
        minutes_since_curtailment: Time since previous curtailment event.
        horizons: Target horizons (default: FORECAST_HORIZONS).

    Returns:
        TemperatureForecast instance.
    """
    target_horizons = horizons or FORECAST_HORIZONS
    cache = load_predictor()

    # Fail-safe check 1: If history is not provided, impute with current temperature
    is_imputed_history = False
    if sensor_history_c is None or len(sensor_history_c) == 0:
        sensor_history_c = [state.temp_c] * 61
        is_imputed_history = True

    # Feature extraction & sensor failure check
    feat_dict, fault_reason = build_features_from_telemetry(
        facility=facility,
        state=state,
        sensor_history_c=sensor_history_c,
        minutes_in_compressor_state=minutes_in_compressor_state,
        minutes_since_curtailment=minutes_since_curtailment,
    )

    if feat_dict is None:
        return TemperatureForecast(
            horizons_min=target_horizons,
            physics_c=[],
            mean_c=[],
            upper_c={},
            valid=False,
            reason=fault_reason or "Telemetry extraction failure",
        )

    # 1. Compute Physics Baseline from operator's naive state guess
    curtail_act = CurtailmentAction(facility.id, start_min=0, duration_min=max(target_horizons))
    physics_state = FacilityState(
        facility_id=facility.id,
        temp_c=feat_dict["lag_0"],
        outside_temp_c=state.outside_temp_c,
        humidity_pct=state.humidity_pct,
        product_load_fraction=state.product_load_fraction,
        door_openings_per_hour=state.door_openings_per_hour,
        compressor_on=False,
        sensor_ok=True,
    )

    phys_df = simulate_temperature(
        facility=facility,
        state=physics_state,
        duration_min=max(target_horizons),
        curtailments=[curtail_act],
        seed=0,
        expected_door_heat=True,
        initial_air_temp_c=feat_dict["lag_0"],
        initial_product_temp_c=feat_dict["lag_0"],
        initial_compressor_on=False,
        initial_minutes_in_state=0,
    )

    # Compute physics minutes to limit
    breached_p = phys_df["air_temp_c"] >= facility.max_safe_temp_c
    phys_min_to_limit = float(phys_df.loc[breached_p, "minute"].iloc[0]) if breached_p.any() else 120.0

    physics_trajectory = [float(phys_df.loc[h - 1, "air_temp_c"]) for h in target_horizons]

    # 2. Build model input records for each horizon
    rows = []
    for idx, h in enumerate(target_horizons):
        r = dict(feat_dict)
        r["horizon_min"] = h
        r["physics_temp_c"] = physics_trajectory[idx]
        r["physics_minutes_to_limit"] = phys_min_to_limit
        rows.append(r)

    df_infer = pd.DataFrame(rows)
    X_m2 = df_infer[list(M2_FEATURES)]

    # 3. Predict Hybrid Mean (M2)
    m2_model: RandomForestRegressor = cache["m2_model"]
    pred_res = m2_model.predict(X_m2)
    hybrid_mean = [round(float(physics_trajectory[i] + pred_res[i]), 3) for i in range(len(target_horizons))]

    # 4. Predict Conservative Upper Bounds (M3)
    m3_models: Dict[float, HistGradientBoostingRegressor] = cache["m3_models"]
    conformal_offsets: Dict[float, Dict[int, float]] = cache["conformal_offsets"]
    phase_conformal_offsets: Dict[str, Dict[float, Dict[int, float]]] = cache.get("phase_conformal_offsets", {})

    # Detect current cycle phase tag for phase-conditional calibration
    current_phase = "mid_cycle"
    if state.compressor_on and minutes_in_compressor_state <= 2:
        current_phase = "cut_in"
    elif not state.compressor_on and minutes_in_compressor_state <= 2:
        current_phase = "cut_out"

    upper_dict: Dict[float, List[float]] = {}
    for level in CONFIDENCE_LEVELS:
        q_mod = m3_models[level]
        raw_upper = q_mod.predict(X_m2)
        calib_upper = []
        for i, h in enumerate(target_horizons):
            if (
                current_phase in phase_conformal_offsets
                and level in phase_conformal_offsets[current_phase]
                and h in phase_conformal_offsets[current_phase][level]
            ):
                c_offset = phase_conformal_offsets[current_phase][level][h]
            else:
                c_offset = conformal_offsets.get(level, {}).get(h, 0.0)

            up_val = float(physics_trajectory[i] + raw_upper[i] + c_offset)
            # Enforce upper >= hybrid mean
            calib_upper.append(max(up_val, hybrid_mean[i]))

        # Enforce non-decreasing monotonic upper bound trajectory
        monotonic_upper = [round(float(v), 3) for v in np.maximum.accumulate(calib_upper)]
        upper_dict[level] = monotonic_upper

    reason_str = "Nominal trajectory prediction"
    if is_imputed_history:
        reason_str = "Imputed flat sensor history (current temperature repeated)"

    return TemperatureForecast(
        horizons_min=target_horizons,
        physics_c=[round(v, 3) for v in physics_trajectory],
        mean_c=hybrid_mean,
        upper_c=upper_dict,
        valid=True,
        reason=reason_str,
    )


def compute_safe_minutes_from_trajectory(
    horizons: List[int],
    upper_bounds: List[float] | np.ndarray,
    max_safe_temp_c: float,
) -> float:
    """Compute safe curtailment duration (minutes) using linear interpolation on upper bound."""
    if len(horizons) == 0 or len(upper_bounds) == 0:
        return 0.0

    if upper_bounds[0] >= max_safe_temp_c:
        return 0.0

    for i in range(1, len(horizons)):
        if upper_bounds[i] >= max_safe_temp_c:
            t_prev, t_curr = horizons[i - 1], horizons[i]
            y_prev, y_curr = upper_bounds[i - 1], upper_bounds[i]
            if y_curr > y_prev:
                frac = (max_safe_temp_c - y_prev) / (y_curr - y_prev)
                return float(round(t_prev + frac * (t_curr - t_prev), 2))
            return float(t_prev)

    return float(horizons[-1])


def predict_temperature(
    facility: Facility,
    state: FacilityState,
    minutes: int,
    sensor_history_c: Optional[List[Optional[float]]] = None,
    confidence: float = 0.90,
) -> Tuple[float, float]:
    """Predict point temperature and conservative upper bound at a specific horizon.

    Returns:
        (mean_c, upper_c)
    """
    forecast = predict_trajectory(
        facility=facility,
        state=state,
        sensor_history_c=sensor_history_c,
        horizons=[minutes],
    )
    if not forecast.valid:
        return float(state.temp_c), float(state.temp_c)

    mean_val = forecast.mean_c[0]
    # Match closest available confidence level
    closest_lvl = min(CONFIDENCE_LEVELS, key=lambda l: abs(l - confidence))
    upper_val = forecast.upper_c[closest_lvl][0]
    return mean_val, upper_val


def estimate_safe_minutes(
    facility: Facility,
    state: FacilityState,
    sensor_history_c: Optional[List[Optional[float]]],
    confidence: float = 0.90,
    guard_margin_c: float = 0.20,
) -> Tuple[float, str]:
    """Estimate provably safe curtailment window with hard fail-safe bounds.

    Returns:
        (safe_minutes, reason)
    """
    try:
        # Fail-safe check 1: Current temp already at or near safe limit
        if state.temp_c >= (facility.max_safe_temp_c - guard_margin_c):
            return 0.0, f"Current temperature ({state.temp_c:.2f}°C) within safety margin of limit ({facility.max_safe_temp_c}°C)."

        # Generate trajectory forecast
        forecast = predict_trajectory(
            facility=facility,
            state=state,
            sensor_history_c=sensor_history_c,
            horizons=FORECAST_HORIZONS,
        )

        if not forecast.valid:
            return 0.0, f"Fail-safe veto: {forecast.reason}"

        closest_lvl = min(CONFIDENCE_LEVELS, key=lambda l: abs(l - confidence))
        upper_traj = forecast.upper_c[closest_lvl]

        effective_limit = facility.max_safe_temp_c - guard_margin_c
        safe_mins = compute_safe_minutes_from_trajectory(
            forecast.horizons_min,
            upper_traj,
            effective_limit,
        )
        return safe_mins, f"Safe window certified at {int(closest_lvl*100)}% conservative confidence."
    except Exception as err:
        return 0.0, f"Inference failure fallback: {err}"


def predict_trajectory_batch(
    facilities: List[Facility],
    states: List[FacilityState],
    sensor_histories: List[Optional[List[Optional[float]]]],
) -> List[TemperatureForecast]:
    """High-speed batch trajectory prediction for fleet dashboard (<200 ms)."""
    cache = load_predictor()
    n_fac = len(facilities)

    # 1. Physics ODE runs and telemetry checks
    physics_trajectories: List[Tuple[List[float], float]] = []
    feat_dicts: List[Dict[str, float]] = []
    valid_flags: List[bool] = []
    fault_reasons: List[str] = []

    for fac, st, hist in zip(facilities, states, sensor_histories):
        if hist is None or len(hist) == 0:
            hist = [st.temp_c] * 61

        fd, reason = build_features_from_telemetry(fac, st, hist, 20, 600.0)
        if fd is None:
            valid_flags.append(False)
            fault_reasons.append(reason or "Telemetry error")
            physics_trajectories.append(([0.0] * len(FORECAST_HORIZONS), 0.0))
            feat_dicts.append({})
            continue

        valid_flags.append(True)
        fault_reasons.append("Nominal trajectory prediction")
        feat_dicts.append(fd)

        curt = CurtailmentAction(fac.id, 0, 120)
        p_st = FacilityState(
            facility_id=fac.id,
            temp_c=fd["lag_0"],
            outside_temp_c=st.outside_temp_c,
            humidity_pct=st.humidity_pct,
            product_load_fraction=st.product_load_fraction,
            door_openings_per_hour=st.door_openings_per_hour,
            compressor_on=False,
            sensor_ok=True,
        )
        p_df = simulate_temperature(
            fac,
            p_st,
            120,
            [curt],
            expected_door_heat=True,
            initial_air_temp_c=fd["lag_0"],
            initial_product_temp_c=fd["lag_0"],
            initial_compressor_on=False,
            initial_minutes_in_state=0,
        )

        breached = p_df["air_temp_c"] >= fac.max_safe_temp_c
        p_lim = float(p_df.loc[breached, "minute"].iloc[0]) if breached.any() else 120.0
        p_traj = [float(p_df.loc[h - 1, "air_temp_c"]) for h in FORECAST_HORIZONS]
        physics_trajectories.append((p_traj, p_lim))

    # 2. Build batched feature matrix
    matrix_rows = []
    row_mapping = []
    for f_idx in range(n_fac):
        if not valid_flags[f_idx]:
            continue
        fd = feat_dicts[f_idx]
        p_traj, p_lim = physics_trajectories[f_idx]
        for h_idx, h in enumerate(FORECAST_HORIZONS):
            row_dict = dict(fd)
            row_dict["horizon_min"] = h
            row_dict["physics_temp_c"] = p_traj[h_idx]
            row_dict["physics_minutes_to_limit"] = p_lim
            matrix_rows.append(row_dict)
            row_mapping.append((f_idx, h_idx))

    if not matrix_rows:
        return [
            TemperatureForecast(
                horizons_min=FORECAST_HORIZONS,
                physics_c=[],
                mean_c=[],
                upper_c={},
                valid=False,
                reason=fault_reasons[i],
            )
            for i in range(n_fac)
        ]

    df_batch = pd.DataFrame(matrix_rows)
    X_mat = df_batch[list(M2_FEATURES)]

    # 3. Predict M2 once
    m2_model: RandomForestRegressor = cache["m2_model"]
    pred_res_all = m2_model.predict(X_mat)

    # 4. Predict M3 once per confidence level
    m3_models: Dict[float, HistGradientBoostingRegressor] = cache["m3_models"]
    conformal_offsets: Dict[float, Dict[int, float]] = cache["conformal_offsets"]
    q_preds_all = {lvl: m3_models[lvl].predict(X_mat) for lvl in CONFIDENCE_LEVELS}

    # 5. Distribute predictions back to facilities
    results: List[TemperatureForecast] = []
    n_horizons = len(FORECAST_HORIZONS)
    row_cursor = 0

    for f_idx in range(n_fac):
        if not valid_flags[f_idx]:
            results.append(
                TemperatureForecast(
                    horizons_min=FORECAST_HORIZONS,
                    physics_c=[],
                    mean_c=[],
                    upper_c={},
                    valid=False,
                    reason=fault_reasons[f_idx],
                )
            )
            continue

        p_traj, _ = physics_trajectories[f_idx]
        res_slice = pred_res_all[row_cursor : row_cursor + n_horizons]
        hybrid_mean = [
            round(float(p_traj[i] + res_slice[i]), 3) for i in range(n_horizons)
        ]

        upper_dict = {}
        for lvl in CONFIDENCE_LEVELS:
            raw_q_slice = q_preds_all[lvl][row_cursor : row_cursor + n_horizons]
            calib_up = []
            for i, h in enumerate(FORECAST_HORIZONS):
                c_off = conformal_offsets.get(lvl, {}).get(h, 0.0)
                up_v = float(p_traj[i] + raw_q_slice[i] + c_off)
                calib_up.append(max(up_v, hybrid_mean[i]))
            upper_dict[lvl] = [round(float(v), 3) for v in np.maximum.accumulate(calib_up)]

        results.append(
            TemperatureForecast(
                horizons_min=FORECAST_HORIZONS,
                physics_c=[round(v, 3) for v in p_traj],
                mean_c=hybrid_mean,
                upper_c=upper_dict,
                valid=True,
                reason=fault_reasons[f_idx],
            )
        )
        row_cursor += n_horizons

    return results
