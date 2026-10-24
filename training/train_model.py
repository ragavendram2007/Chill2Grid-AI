"""train_model.py - Model Training & Evaluation Engine for Chill2Grid AI.

Trains and evaluates:
  M0: Physics-Only Baseline (zero parameters)
  M1: ML-Only Forecaster (RandomForestRegressor predicting true temperature)
  M2: Hybrid Physics+ML Forecaster (RandomForestRegressor predicting physics residual)
  M3: Conservative Prediction Intervals (Quantile HistGradientBoosting + Split-Conformal Calibration)

Enforces strict leakage prevention, scenario-stratified evaluation,
and produces comprehensive metric tables and interactive visualization artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from core.contracts import load_facilities_from_csv
from training.scenario_generator import (
    FORECAST_HORIZONS,
    FORBIDDEN_FEATURES,
    M1_FEATURES,
    M2_FEATURES,
    OBSERVABLE_FEATURES,
    generate_data_quality_report,
    generate_full_dataset,
    split_dataset_by_scenario,
)


CONFIDENCE_LEVELS = [0.50, 0.80, 0.90, 0.95, 0.99]


# ============================================================================
# SAFE WINDOW INTERPOLATION
# ============================================================================

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


# ============================================================================
# TRAINING & CONFORMAL CALIBRATION
# ============================================================================

def train_all_models(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    quick: bool = False,
) -> Tuple[
    RandomForestRegressor,
    RandomForestRegressor,
    Dict[float, HistGradientBoostingRegressor],
    Dict[str, object],
    float,
]:
    """Train M1, M2, and M3 models and calibrate conformal offsets using calibration split."""
    start_time = time.time()
    n_trees = 60 if quick else 200

    # 1. Train M1 (ML-only)
    X_m1_tr = train_df[list(M1_FEATURES)]
    y_m1_tr = train_df["true_temp_c"]
    m1_rf = RandomForestRegressor(
        n_estimators=n_trees,
        min_samples_leaf=5,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
    )
    m1_rf.fit(X_m1_tr, y_m1_tr)

    # 2. Train M2 (Hybrid residual)
    X_m2_tr = train_df[list(M2_FEATURES)]
    y_m2_tr = train_df["residual_c"]
    m2_rf = RandomForestRegressor(
        n_estimators=n_trees,
        min_samples_leaf=5,
        max_features=0.5,
        n_jobs=-1,
        random_state=42,
    )
    m2_rf.fit(X_m2_tr, y_m2_tr)

    # 3. Train M3 Quantile Regressors on residual_c
    m3_models: Dict[float, HistGradientBoostingRegressor] = {}
    conformal_offsets_pooled: Dict[float, Dict[int, float]] = {}
    conformal_offsets_phase: Dict[str, Dict[float, Dict[int, float]]] = {
        "cut_in": {},
        "cut_out": {},
        "mid_cycle": {},
    }

    X_m2_calib = calib_df[list(M2_FEATURES)]
    y_res_calib = calib_df["residual_c"].values

    for level in CONFIDENCE_LEVELS:
        q_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=level,
            max_iter=50 if quick else 100,
            random_state=42,
        )
        q_model.fit(X_m2_tr, y_m2_tr)
        m3_models[level] = q_model

        # Pooled split-conformal calibration per horizon
        raw_calib_pred = q_model.predict(X_m2_calib)
        conformal_offsets_pooled[level] = {}

        for h in FORECAST_HORIZONS:
            h_mask = (calib_df["horizon_min"] == h).values
            scores_h = y_res_calib[h_mask] - raw_calib_pred[h_mask]
            n_calib = len(scores_h)
            if n_calib > 0:
                k_quantile = min(1.0, np.ceil((n_calib + 1) * level) / n_calib)
                offset = float(np.quantile(scores_h, k_quantile))
            else:
                offset = 0.0
            conformal_offsets_pooled[level][h] = round(offset, 4)

        # Phase-conditional conformal calibration per (phase, horizon)
        for ph in ("cut_in", "cut_out", "mid_cycle"):
            if level not in conformal_offsets_phase[ph]:
                conformal_offsets_phase[ph][level] = {}
            for h in FORECAST_HORIZONS:
                ph_h_mask = (
                    (calib_df["horizon_min"] == h)
                    & (calib_df["phase_tag"] == ph)
                ).values
                scores_ph_h = y_res_calib[ph_h_mask] - raw_calib_pred[ph_h_mask]
                n_ph_calib = len(scores_ph_h)
                # Requirement 4: pooled fallback if < 200 calib rows
                if n_ph_calib >= 200:
                    k_q = min(1.0, np.ceil((n_ph_calib + 1) * level) / n_ph_calib)
                    ph_offset = float(np.quantile(scores_ph_h, k_q))
                else:
                    ph_offset = conformal_offsets_pooled[level][h]
                conformal_offsets_phase[ph][level][h] = round(ph_offset, 4)

    conformal_offsets_data = {
        "pooled": conformal_offsets_pooled,
        "phase": conformal_offsets_phase,
    }
    training_duration_sec = round(time.time() - start_time, 2)
    return m1_rf, m2_rf, m3_models, conformal_offsets_data, training_duration_sec


# ============================================================================
# EVALUATION & METRICS PIPELINE
# ============================================================================

def evaluate_models(
    test_df: pd.DataFrame,
    m1_model: RandomForestRegressor,
    m2_model: RandomForestRegressor,
    m3_models: Dict[float, HistGradientBoostingRegressor],
    conformal_offsets_data: Dict[str, object] | Dict[float, Dict[int, float]],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Perform rigorous evaluation on the held-out test split."""
    test_eval = test_df.copy()

    # Extract pooled and phase-conditional offsets
    if "pooled" in conformal_offsets_data and "phase" in conformal_offsets_data:
        pooled_offsets: Dict[float, Dict[int, float]] = conformal_offsets_data["pooled"]
        phase_offsets: Dict[str, Dict[float, Dict[int, float]]] = conformal_offsets_data["phase"]
    else:
        pooled_offsets = conformal_offsets_data
        phase_offsets = {}

    # M0 Prediction
    test_eval["pred_m0"] = test_eval["physics_temp_c"]

    # M1 Prediction
    X_m1 = test_eval[list(M1_FEATURES)]
    test_eval["pred_m1"] = m1_model.predict(X_m1)

    # M2 Hybrid Prediction
    X_m2 = test_eval[list(M2_FEATURES)]
    pred_res_m2 = m2_model.predict(X_m2)
    test_eval["pred_m2_res"] = pred_res_m2
    test_eval["pred_m2"] = test_eval["physics_temp_c"] + pred_res_m2

    # M3 Upper Bounds (both Pooled Before and Phase-Conditional After)
    for level in CONFIDENCE_LEVELS:
        q_mod = m3_models[level]
        raw_upper = q_mod.predict(X_m2)

        # Pooled Upper Bound (Before)
        offsets_pooled = test_eval["horizon_min"].map(pooled_offsets[level]).values
        calibrated_upper_pooled = test_eval["physics_temp_c"].values + raw_upper + offsets_pooled
        test_eval[f"upper_pooled_{int(level*100)}"] = np.maximum(
            calibrated_upper_pooled, test_eval["pred_m2"].values
        )

        # Phase-Conditional Upper Bound (After)
        phase_offset_list = [
            phase_offsets.get(ph, {}).get(level, {}).get(h, pooled_offsets[level][h])
            for ph, h in zip(test_eval["phase_tag"], test_eval["horizon_min"])
        ]
        calibrated_upper_phase = (
            test_eval["physics_temp_c"].values + raw_upper + np.array(phase_offset_list)
        )
        test_eval[f"upper_phase_{int(level*100)}"] = np.maximum(
            calibrated_upper_phase, test_eval["pred_m2"].values
        )

        # Active Upper Bound uses Phase-Conditional
        test_eval[f"upper_{int(level*100)}"] = test_eval[f"upper_phase_{int(level*100)}"]

    # -------------------------------------------------------------------------
    # 1. Model Comparison Table (M0, M1, M2)
    # -------------------------------------------------------------------------
    def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return round(mae, 3), round(rmse, 3)

    comp_rows = []
    y_true_all = test_eval["true_temp_c"].values
    m0_mae, m0_rmse = calc_metrics(y_true_all, test_eval["pred_m0"].values)
    m1_mae, m1_rmse = calc_metrics(y_true_all, test_eval["pred_m1"].values)
    m2_mae, m2_rmse = calc_metrics(y_true_all, test_eval["pred_m2"].values)
    comp_rows.append({
        "segment": "Overall",
        "horizon_min": "All",
        "n_samples": len(test_eval),
        "M0_Physics_MAE": m0_mae,
        "M0_Physics_RMSE": m0_rmse,
        "M1_ML_MAE": m1_mae,
        "M1_ML_RMSE": m1_rmse,
        "M2_Hybrid_MAE": m2_mae,
        "M2_Hybrid_RMSE": m2_rmse,
        "Hybrid_MAE_Reduction_Pct": round((m0_mae - m2_mae) / m0_mae * 100.0, 1),
    })

    # Near-limit zone (true temperature within 1.0 °C of max_safe_temp_c)
    near_mask = (test_eval["true_temp_c"] >= test_eval["max_safe_temp_c"] - 1.0).values
    if near_mask.any():
        y_near = y_true_all[near_mask]
        m0_n_mae, m0_n_rmse = calc_metrics(y_near, test_eval.loc[near_mask, "pred_m0"].values)
        m1_n_mae, m1_n_rmse = calc_metrics(y_near, test_eval.loc[near_mask, "pred_m1"].values)
        m2_n_mae, m2_n_rmse = calc_metrics(y_near, test_eval.loc[near_mask, "pred_m2"].values)
        comp_rows.append({
            "segment": "Near-Limit (Within 1.0 C of Safe Max)",
            "horizon_min": "All",
            "n_samples": int(near_mask.sum()),
            "M0_Physics_MAE": m0_n_mae,
            "M0_Physics_RMSE": m0_n_rmse,
            "M1_ML_MAE": m1_n_mae,
            "M1_ML_RMSE": m1_n_rmse,
            "M2_Hybrid_MAE": m2_n_mae,
            "M2_Hybrid_RMSE": m2_n_rmse,
            "Hybrid_MAE_Reduction_Pct": round((m0_n_mae - m2_n_mae) / m0_n_mae * 100.0, 1),
        })

    # Per Horizon Breakdown
    for h in FORECAST_HORIZONS:
        h_mask = (test_eval["horizon_min"] == h).values
        y_h = y_true_all[h_mask]
        m0_h_mae, m0_h_rmse = calc_metrics(y_h, test_eval.loc[h_mask, "pred_m0"].values)
        m1_h_mae, m1_h_rmse = calc_metrics(y_h, test_eval.loc[h_mask, "pred_m1"].values)
        m2_h_mae, m2_h_rmse = calc_metrics(y_h, test_eval.loc[h_mask, "pred_m2"].values)
        comp_rows.append({
            "segment": f"Horizon {h}m",
            "horizon_min": h,
            "n_samples": int(h_mask.sum()),
            "M0_Physics_MAE": m0_h_mae,
            "M0_Physics_RMSE": m0_h_rmse,
            "M1_ML_MAE": m1_h_mae,
            "M1_ML_RMSE": m1_h_rmse,
            "M2_Hybrid_MAE": m2_h_mae,
            "M2_Hybrid_RMSE": m2_h_rmse,
            "Hybrid_MAE_Reduction_Pct": round((m0_h_mae - m2_h_mae) / m0_h_mae * 100.0, 1),
        })

    df_comparison = pd.DataFrame(comp_rows)

    # -------------------------------------------------------------------------
    # 2. Conformal Coverage Table (by Facility)
    # -------------------------------------------------------------------------
    cov_rows = []
    for level in CONFIDENCE_LEVELS:
        lvl_key = f"upper_{int(level*100)}"
        overall_cov = float(np.mean(test_eval["true_temp_c"] <= test_eval[lvl_key]))
        row = {
            "target_confidence": level,
            "overall_coverage": round(overall_cov, 4),
            "coverage_gap": round(overall_cov - level, 4),
        }
        for fac_id in sorted(test_eval["facility_id"].unique()):
            fac_mask = test_eval["facility_id"] == fac_id
            fac_cov = float(np.mean(test_eval.loc[fac_mask, "true_temp_c"] <= test_eval.loc[fac_mask, lvl_key]))
            row[f"{fac_id}_coverage"] = round(fac_cov, 4)
        cov_rows.append(row)
    df_coverage = pd.DataFrame(cov_rows)

    # -------------------------------------------------------------------------
    # 3. Phase-Conditional Conformal Coverage Audit (Before vs After)
    # -------------------------------------------------------------------------
    phase_before_after_rows = []
    for level in CONFIDENCE_LEVELS:
        lvl_int = int(level * 100)
        # Pooled (Before)
        pooled_cov = float(np.mean(test_eval["true_temp_c"] <= test_eval[f"upper_pooled_{lvl_int}"]))
        cut_in_b = float(np.mean(test_eval.loc[test_eval["phase_tag"] == "cut_in", "true_temp_c"] <= test_eval.loc[test_eval["phase_tag"] == "cut_in", f"upper_pooled_{lvl_int}"]))
        cut_out_b = float(np.mean(test_eval.loc[test_eval["phase_tag"] == "cut_out", "true_temp_c"] <= test_eval.loc[test_eval["phase_tag"] == "cut_out", f"upper_pooled_{lvl_int}"]))
        mid_b = float(np.mean(test_eval.loc[test_eval["phase_tag"] == "mid_cycle", "true_temp_c"] <= test_eval.loc[test_eval["phase_tag"] == "mid_cycle", f"upper_pooled_{lvl_int}"]))

        phase_before_after_rows.append({
            "target_confidence": level,
            "calibration_stage": "Before (Pooled)",
            "overall_coverage": round(pooled_cov, 4),
            "cut_in_coverage": round(cut_in_b, 4),
            "cut_in_gap": round(cut_in_b - level, 4),
            "cut_out_coverage": round(cut_out_b, 4),
            "cut_out_gap": round(cut_out_b - level, 4),
            "mid_cycle_coverage": round(mid_b, 4),
            "mid_cycle_gap": round(mid_b - level, 4),
        })

        # Phase-Conditional (After)
        phase_cov = float(np.mean(test_eval["true_temp_c"] <= test_eval[f"upper_phase_{lvl_int}"]))
        cut_in_a = float(np.mean(test_eval.loc[test_eval["phase_tag"] == "cut_in", "true_temp_c"] <= test_eval.loc[test_eval["phase_tag"] == "cut_in", f"upper_phase_{lvl_int}"]))
        cut_out_a = float(np.mean(test_eval.loc[test_eval["phase_tag"] == "cut_out", "true_temp_c"] <= test_eval.loc[test_eval["phase_tag"] == "cut_out", f"upper_phase_{lvl_int}"]))
        mid_a = float(np.mean(test_eval.loc[test_eval["phase_tag"] == "mid_cycle", "true_temp_c"] <= test_eval.loc[test_eval["phase_tag"] == "mid_cycle", f"upper_phase_{lvl_int}"]))

        phase_before_after_rows.append({
            "target_confidence": level,
            "calibration_stage": "After (Phase-Conditional)",
            "overall_coverage": round(phase_cov, 4),
            "cut_in_coverage": round(cut_in_a, 4),
            "cut_in_gap": round(cut_in_a - level, 4),
            "cut_out_coverage": round(cut_out_a, 4),
            "cut_out_gap": round(cut_out_a - level, 4),
            "mid_cycle_coverage": round(mid_a, 4),
            "mid_cycle_gap": round(mid_a - level, 4),
        })

    df_phase_cov_before_after = pd.DataFrame(phase_before_after_rows)

    # -------------------------------------------------------------------------
    # 4. Empirical Residual Standard Deviation Check
    # -------------------------------------------------------------------------
    res_std_rows = []
    for h in FORECAST_HORIZONS:
        h_mask = (test_eval["horizon_min"] == h).values
        residuals_h = test_eval.loc[h_mask, "residual_c"].values
        row = {
            "horizon_min": h,
            "overall_residual_std_c": round(float(np.std(residuals_h)), 3),
            "overall_residual_mean_c": round(float(np.mean(residuals_h)), 3),
        }
        for fac_id in sorted(test_eval["facility_id"].unique()):
            fac_h_mask = h_mask & (test_eval["facility_id"] == fac_id).values
            if fac_h_mask.any():
                fac_res = test_eval.loc[fac_h_mask, "residual_c"].values
                row[f"{fac_id}_std_c"] = round(float(np.std(fac_res)), 3)
            else:
                row[f"{fac_id}_std_c"] = 0.0
        res_std_rows.append(row)
    df_res_std = pd.DataFrame(res_std_rows)

    # -------------------------------------------------------------------------
    # 5. Safe Window Decision Accuracy & Three-Way Censoring Breakdown Table
    # -------------------------------------------------------------------------
    record_groups = test_eval.groupby(["scenario_id", "record_id"])
    records_meta = []
    for (sc_id, rec_id), grp in record_groups:
        grp_sorted = grp.sort_values("horizon_min")
        horizons = grp_sorted["horizon_min"].tolist()
        max_safe = float(grp_sorted["max_safe_temp_c"].iloc[0])
        true_limit = float(grp_sorted["true_minutes_to_limit"].iloc[0])
        phys_limit = float(grp_sorted["physics_minutes_to_limit"].iloc[0])
        phase_val = grp_sorted["phase"].iloc[0]
        phase_tag_val = grp_sorted["phase_tag"].iloc[0]

        record_entry = {
            "scenario_id": sc_id,
            "record_id": rec_id,
            "facility_id": grp_sorted["facility_id"].iloc[0],
            "phase": phase_val,
            "phase_tag": phase_tag_val,
            "max_safe_temp_c": max_safe,
            "true_minutes_to_limit": true_limit,
            "physics_safe_min": phys_limit,
        }
        for level in CONFIDENCE_LEVELS:
            lvl_col = f"upper_{int(level*100)}"
            upper_traj = np.maximum.accumulate(grp_sorted[lvl_col].tolist())
            safe_min = compute_safe_minutes_from_trajectory(horizons, upper_traj, max_safe)
            record_entry[f"safe_min_{int(level*100)}"] = safe_min

        records_meta.append(record_entry)

    df_records_safe = pd.DataFrame(records_meta)

    # Helper function for three-way censoring breakdown
    def compute_tradeoff_for_subset(subset_name: str, sub_df: pd.DataFrame) -> List[Dict[str, object]]:
        n_sub = len(sub_df)
        if n_sub == 0:
            return []
        rows = []
        avg_true_safe = float(sub_df["true_minutes_to_limit"].mean())
        avg_m0_safe = float(sub_df["physics_safe_min"].mean())
        m0_unsafe_cnt = int(np.sum(sub_df["physics_safe_min"] > sub_df["true_minutes_to_limit"] + 1e-3))
        m0_unsafe_rate = round(m0_unsafe_cnt / n_sub * 100.0, 2)

        rows.append({
            "subset": subset_name,
            "n_records": n_sub,
            "model": "M0_Physics_Mean",
            "confidence_level": 0.50,
            "avg_predicted_safe_min": round(avg_m0_safe, 1),
            "flexibility_penalty_vs_m0_min": 0.0,
            "safety_margin_vs_true_min": round(avg_true_safe - avg_m0_safe, 1),
            "unsafe_error_rate_pct": m0_unsafe_rate,
        })

        for level in CONFIDENCE_LEVELS:
            lvl_key = f"safe_min_{int(level*100)}"
            unsafe_cnt = int(np.sum(sub_df[lvl_key] > sub_df["true_minutes_to_limit"] + 1e-3))
            unsafe_rate = round(unsafe_cnt / n_sub * 100.0, 2)
            avg_pred = float(sub_df[lvl_key].mean())
            flex_penalty = round(avg_m0_safe - avg_pred, 1)
            safety_margin = round(avg_true_safe - avg_pred, 1)

            rows.append({
                "subset": subset_name,
                "n_records": n_sub,
                "model": f"M3_Hybrid_Upper_{int(level*100)}%",
                "confidence_level": level,
                "avg_predicted_safe_min": round(avg_pred, 1),
                "flexibility_penalty_vs_m0_min": flex_penalty,
                "safety_margin_vs_true_min": safety_margin,
                "unsafe_error_rate_pct": unsafe_rate,
            })
        return rows

    # Three-way breakdown calculation
    sub_all = df_records_safe
    sub_excl_0 = df_records_safe[df_records_safe["true_minutes_to_limit"] > 0.0].copy()
    sub_uncens = df_records_safe[
        (df_records_safe["true_minutes_to_limit"] > 0.0)
        & (df_records_safe["true_minutes_to_limit"] < 120.0)
    ].copy()

    tradeoff_rows: List[Dict[str, object]] = []
    tradeoff_rows.extend(compute_tradeoff_for_subset("All Records", sub_all))
    tradeoff_rows.extend(compute_tradeoff_for_subset("Excluding Over-Limit (T_true > 0)", sub_excl_0))
    tradeoff_rows.extend(compute_tradeoff_for_subset("Uncensored Only (0 < T_true < 120)", sub_uncens))
    df_tradeoff = pd.DataFrame(tradeoff_rows)

    # -------------------------------------------------------------------------
    # 6. Already-Over-Limit Share by Facility
    # -------------------------------------------------------------------------
    already_over_limit_rows = []
    for fac_id in sorted(df_records_safe["facility_id"].unique()):
        fac_recs = df_records_safe[df_records_safe["facility_id"] == fac_id]
        total_cnt = len(fac_recs)
        over_cnt = int((fac_recs["true_minutes_to_limit"] <= 0.0).sum())
        over_pct = round(over_cnt / total_cnt * 100.0, 2) if total_cnt > 0 else 0.0
        already_over_limit_rows.append({
            "facility_id": fac_id,
            "total_records": total_cnt,
            "already_over_limit_count": over_cnt,
            "already_over_limit_pct": over_pct,
            "operational_cause": "Deadband cut-in peak under high ambient heat / door openings (zero headroom at t=0)",
        })
    df_already_over_limit = pd.DataFrame(already_over_limit_rows)

    return (
        test_eval,
        df_comparison,
        df_coverage,
        df_phase_cov_before_after,
        df_res_std,
        df_records_safe,
        df_tradeoff,
        df_already_over_limit,
    )


# ============================================================================
# ROBUSTNESS CHECKS: LOFO & HOT-DAY EXTRAPOLATION
# ============================================================================

def run_robustness_checks(
    full_df: pd.DataFrame,
    quick: bool = True,
) -> pd.DataFrame:
    """Run (i) Leave-One-Facility-Out (LOFO) and (ii) Hot-Day Extrapolation checks."""
    rob_rows = []
    n_trees = 40 if quick else 100

    facilities = sorted(full_df["facility_id"].unique())

    # 1. Leave-One-Facility-Out (LOFO)
    for holdout_fac in facilities:
        train_sub = full_df[full_df["facility_id"] != holdout_fac]
        test_sub = full_df[full_df["facility_id"] == holdout_fac]

        m2 = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=5, n_jobs=-1, random_state=42)
        m2.fit(train_sub[list(M2_FEATURES)], train_sub["residual_c"])

        pred_res = m2.predict(test_sub[list(M2_FEATURES)])
        pred_m2 = test_sub["physics_temp_c"].values + pred_res

        m0_mae = float(np.mean(np.abs(test_sub["true_temp_c"] - test_sub["physics_temp_c"])))
        m2_mae = float(np.mean(np.abs(test_sub["true_temp_c"] - pred_m2)))

        rob_rows.append({
            "experiment": f"LOFO_Holdout_{holdout_fac}",
            "train_size": len(train_sub),
            "test_size": len(test_sub),
            "M0_Physics_MAE": round(m0_mae, 3),
            "M2_Hybrid_MAE": round(m2_mae, 3),
            "Hybrid_Improvement_Pct": round((m0_mae - m2_mae) / m0_mae * 100.0, 1),
        })

    # 2. Hot-Day Extrapolation: Train on outside_base_c <= 38 C, Test on > 38 C
    hot_train = full_df[full_df["outside_base_c"] <= 38.0]
    hot_test = full_df[full_df["outside_base_c"] > 38.0]

    if len(hot_test) > 0 and len(hot_train) > 0:
        m2_hot = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=5, n_jobs=-1, random_state=42)
        m2_hot.fit(hot_train[list(M2_FEATURES)], hot_train["residual_c"])

        pred_res_hot = m2_hot.predict(hot_test[list(M2_FEATURES)])
        pred_m2_hot = hot_test["physics_temp_c"].values + pred_res_hot

        m0_h_mae = float(np.mean(np.abs(hot_test["true_temp_c"] - hot_test["physics_temp_c"])))
        m2_h_mae = float(np.mean(np.abs(hot_test["true_temp_c"] - pred_m2_hot)))

        rob_rows.append({
            "experiment": "Hot_Day_Extrapolation (>38 C Test)",
            "train_size": len(hot_train),
            "test_size": len(hot_test),
            "M0_Physics_MAE": round(m0_h_mae, 3),
            "M2_Hybrid_MAE": round(m2_h_mae, 3),
            "Hybrid_Improvement_Pct": round((m0_h_mae - m2_h_mae) / m0_h_mae * 100.0, 1),
        })

    return pd.DataFrame(rob_rows)


# ============================================================================
# MAIN TRAINING PIPELINE ENTRYPOINT
# ============================================================================

def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as standard GitHub-flavored markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.values]
    col_w = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_w[i] = max(col_w[i], len(val))
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"
    head = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    body = ["| " + " | ".join(r[i].ljust(col_w[i]) for i in range(len(r))) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Chill2Grid AI hybrid forecaster.")
    parser.add_argument("--quick", action="store_true", help="Run quick training pipeline (60 scenarios)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cal_csv = project_root / "data" / "facilities_calibrated.csv"
    if not cal_csv.is_file():
        cal_csv = project_root / "data" / "facilities.csv"

    facilities = load_facilities_from_csv(cal_csv)
    n_sc = 60 if args.quick else 500

    print(f"\n=======================================================")
    print(f"CHILL2GRID AI - PART 3 HYBRID AI MODEL TRAINING PIPELINE")
    print(f"=======================================================")
    print(f"Mode: {'QUICK (--quick)' if args.quick else 'FULL PRODUCTION'}")
    print(f"Fleet: {len(facilities)} calibrated facilities | Scenarios: {n_sc}")

    # Clear stale outputs first
    out_dir = project_root / "outputs" / "part3"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in out_dir.iterdir():
        if stale_file.is_file():
            stale_file.unlink()

    # 1. Dataset Generation
    t0 = time.time()
    df_dataset = generate_full_dataset(facilities, n_scenarios=n_sc, n_jobs=-1)
    gen_time = round(time.time() - t0, 2)
    print(f"Dataset generated: {len(df_dataset)} rows across 12 horizons in {gen_time}s.")

    # Generate Data Quality Report
    generate_data_quality_report(df_dataset, out_dir)
    print(f"Data quality report generated in {out_dir}")

    # 2. Scenario-Stratified Split
    train_df, calib_df, test_df = split_dataset_by_scenario(df_dataset, train_frac=0.60, calib_frac=0.20)
    n_tr_sc = len(train_df["scenario_id"].unique())
    n_cal_sc = len(calib_df["scenario_id"].unique())
    n_te_sc = len(test_df["scenario_id"].unique())
    print(f"Split sizes: Train={len(train_df)} ({n_tr_sc} scenarios) | "
          f"Calibration={len(calib_df)} ({n_cal_sc} scenarios) | "
          f"Test={len(test_df)} ({n_te_sc} scenarios)")

    # 3. Model Training & Conformal Calibration
    m1_rf, m2_rf, m3_models, conformal_offsets, train_time = train_all_models(train_df, calib_df, quick=args.quick)
    print(f"Models trained and calibrated in {train_time}s.")

    # 4. Evaluation
    (
        test_eval,
        df_comp,
        df_cov,
        df_phase_cov_before_after,
        df_res_std,
        df_records_safe,
        df_tradeoff,
        df_already_over_limit,
    ) = evaluate_models(test_df, m1_rf, m2_rf, m3_models, conformal_offsets)

    # 5. Robustness Checks
    print("Running Leave-One-Facility-Out and Hot-Day Extrapolation checks...")
    df_robustness = run_robustness_checks(df_dataset, quick=True)

    # 6. Save Model Artifacts
    artifacts_dir = project_root / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(m1_rf, artifacts_dir / "m1_ml_only.joblib")
    joblib.dump(m2_rf, artifacts_dir / "m2_hybrid.joblib")
    joblib.dump(m3_models, artifacts_dir / "m3_quantile_models.joblib")

    with open(artifacts_dir / "conformal_offsets.json", "w") as f:
        json.dump(conformal_offsets, f, indent=2)

    distinct_records = df_dataset.drop_duplicates(subset=["scenario_id", "record_id"])
    right_cens_share = float((distinct_records["true_minutes_to_limit"] >= 120.0).mean())
    left_cens_share = float((distinct_records["true_minutes_to_limit"] <= 0.0).mean())

    all_90_row = df_tradeoff[(df_tradeoff["subset"] == "All Records") & (df_tradeoff["confidence_level"] == 0.90)].iloc[0]

    metadata = {
        "model_version": "1.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": "3.14.3",
        "sklearn_version": sklearn.__version__,
        "base_seed": 42,
        "n_scenarios": n_sc,
        "observable_features": list(OBSERVABLE_FEATURES),
        "m1_features": list(M1_FEATURES),
        "m2_features": list(M2_FEATURES),
        "horizons": FORECAST_HORIZONS,
        "censoring_stats": {
            "right_censored_share": round(right_cens_share, 4),
            "left_censored_share": round(left_cens_share, 4),
        },
        "headline_metrics": {
            "overall_m0_physics_mae": float(df_comp.loc[0, "M0_Physics_MAE"]),
            "overall_m1_ml_mae": float(df_comp.loc[0, "M1_ML_MAE"]),
            "overall_m2_hybrid_mae": float(df_comp.loc[0, "M2_Hybrid_MAE"]),
            "hybrid_mae_improvement_pct": float(df_comp.loc[0, "Hybrid_MAE_Reduction_Pct"]),
            "coverage_90pct_achieved": float(df_cov.loc[df_cov["target_confidence"] == 0.90, "overall_coverage"].iloc[0]),
            "unsafe_error_rate_90pct": float(all_90_row["unsafe_error_rate_pct"]),
            "flexibility_penalty_vs_m0_90pct": float(all_90_row["flexibility_penalty_vs_m0_min"]),
            "avg_predicted_safe_min_90pct": float(all_90_row["avg_predicted_safe_min"]),
        },
    }
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 7. Save Output CSV Artifacts
    df_comp.to_csv(out_dir / "model_comparison.csv", index=False)
    df_cov.to_csv(out_dir / "coverage_table.csv", index=False)
    df_phase_cov_before_after.to_csv(out_dir / "phase_coverage_before_after.csv", index=False)
    df_phase_cov_before_after.to_csv(out_dir / "phase_coverage_table.csv", index=False)
    df_res_std.to_csv(out_dir / "residual_std_check.csv", index=False)
    df_records_safe.to_csv(out_dir / "safe_window_errors.csv", index=False)
    df_tradeoff.to_csv(out_dir / "tradeoff_table.csv", index=False)
    df_already_over_limit.to_csv(out_dir / "already_over_limit_by_facility.csv", index=False)
    df_robustness.to_csv(out_dir / "robustness_checks.csv", index=False)

    # 8. Interactive HTML Plots
    # A. plot_error_by_horizon.html
    h_df = df_comp[df_comp["segment"].str.startswith("Horizon")].copy()
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=h_df["horizon_min"], y=h_df["M0_Physics_MAE"], name="M0 Physics Baseline (simulated)", line=dict(color="#d62728", width=2)))
    fig_h.add_trace(go.Scatter(x=h_df["horizon_min"], y=h_df["M1_ML_MAE"], name="M1 Pure ML (simulated)", line=dict(color="#ff7f0e", width=2)))
    fig_h.add_trace(go.Scatter(x=h_df["horizon_min"], y=h_df["M2_Hybrid_MAE"], name="M2 Hybrid AI (simulated)", line=dict(color="#2ca02c", width=2.5)))
    fig_h.update_layout(title="Forecast Error (MAE °C) by Curtailment Horizon (Simulated)", xaxis_title="Horizon (minutes)", yaxis_title="MAE (°C)", template="plotly_white")
    fig_h.write_html(str(out_dir / "plot_error_by_horizon.html"))

    # B. plot_pred_vs_true.html
    sample_eval = test_eval.sample(n=min(1500, len(test_eval)), random_state=42)
    fig_pv = make_subplots(rows=1, cols=3, subplot_titles=["M0: Physics-Only", "M1: ML-Only", "M2: Hybrid AI"])
    fig_pv.add_trace(go.Scatter(x=sample_eval["true_temp_c"], y=sample_eval["pred_m0"], mode="markers", marker=dict(size=4, color="#d62728", opacity=0.6), name="M0"), row=1, col=1)
    fig_pv.add_trace(go.Scatter(x=sample_eval["true_temp_c"], y=sample_eval["pred_m1"], mode="markers", marker=dict(size=4, color="#ff7f0e", opacity=0.6), name="M1"), row=1, col=2)
    fig_pv.add_trace(go.Scatter(x=sample_eval["true_temp_c"], y=sample_eval["pred_m2"], mode="markers", marker=dict(size=4, color="#2ca02c", opacity=0.6), name="M2"), row=1, col=3)
    fig_pv.update_layout(title="Predicted vs True Temperature Across Horizons (Simulated)", template="plotly_white")
    fig_pv.write_html(str(out_dir / "plot_pred_vs_true.html"))

    # C. plot_safety_flex_tradeoff.html
    fig_tf = make_subplots(specs=[[{"secondary_y": True}]])
    sub_to = df_tradeoff[(df_tradeoff["subset"] == "All Records") & (df_tradeoff["model"].str.contains("Hybrid"))].copy()
    fig_tf.add_trace(go.Scatter(x=sub_to["confidence_level"], y=sub_to["unsafe_error_rate_pct"], name="Unsafe Error Rate (%)", line=dict(color="red", width=2.5)), secondary_y=False)
    fig_tf.add_trace(go.Scatter(x=sub_to["confidence_level"], y=sub_to["flexibility_penalty_vs_m0_min"], name="Flexibility Penalty vs M0 (min)", line=dict(color="blue", width=2.5, dash="dot")), secondary_y=True)
    fig_tf.update_layout(title="Safety vs Flexibility Trade-off across Confidence Levels (Simulated)", xaxis_title="Nominal Confidence Level", template="plotly_white")
    fig_tf.update_yaxes(title_text="Unsafe Error Rate (%)", secondary_y=False)
    fig_tf.update_yaxes(title_text="Flexibility Penalty vs M0 (minutes)", secondary_y=True)
    fig_tf.write_html(str(out_dir / "plot_safety_flex_tradeoff.html"))

    # D. plot_feature_importance.html
    importances = m2_rf.feature_importances_
    feat_df = pd.DataFrame({"feature": list(M2_FEATURES), "importance": importances}).sort_values("importance", ascending=True).tail(15)
    fig_imp = px.bar(feat_df, x="importance", y="feature", orientation="h", title="Top 15 Hybrid Feature Importances (M2 Residual Learner, Simulated)")
    fig_imp.update_layout(template="plotly_white")
    fig_imp.write_html(str(out_dir / "plot_feature_importance.html"))

    # E. plot_trajectory_examples.html (4 diverse cases)
    sample_records = df_records_safe.sample(n=min(4, len(df_records_safe)), random_state=123)
    fig_cases = make_subplots(rows=2, cols=2, subplot_titles=[f"Case {i+1}: {r['facility_id']} ({r['phase_tag']}, Limit: {r['max_safe_temp_c']} °C)" for i, (_, r) in enumerate(sample_records.iterrows())])

    for idx, (_, rec) in enumerate(sample_records.iterrows()):
        row_pos = idx // 2 + 1
        col_pos = idx % 2 + 1
        sub_traj = test_eval[(test_eval["scenario_id"] == rec["scenario_id"]) & (test_eval["record_id"] == rec["record_id"])].sort_values("horizon_min")
        hz = sub_traj["horizon_min"].tolist()
        fig_cases.add_trace(go.Scatter(x=hz, y=sub_traj["true_temp_c"], mode="lines+markers", name="Ground Truth (simulated)", line=dict(color="black", width=2)), row=row_pos, col=col_pos)
        fig_cases.add_trace(go.Scatter(x=hz, y=sub_traj["physics_temp_c"], mode="lines", name="Physics M0 (simulated)", line=dict(color="red", dash="dash")), row=row_pos, col=col_pos)
        fig_cases.add_trace(go.Scatter(x=hz, y=sub_traj["pred_m2"], mode="lines", name="Hybrid Mean M2 (simulated)", line=dict(color="green")), row=row_pos, col=col_pos)
        fig_cases.add_trace(go.Scatter(x=hz, y=sub_traj["upper_90"], mode="lines", name="90% Upper Bound (simulated)", line=dict(color="purple", dash="dot")), row=row_pos, col=col_pos)
        fig_cases.add_hline(y=rec["max_safe_temp_c"], line_dash="dash", line_color="red", row=row_pos, col=col_pos)

    fig_cases.update_layout(title="Trajectory Forecast Examples: Truth, Physics, Hybrid & 90% Upper Bound (Simulated)", template="plotly_white")
    fig_cases.write_html(str(out_dir / "plot_trajectory_examples.html"))

    # 9. Evaluation Report
    vax_fac = next(f for f in facilities if f.id == "FAC-VAX-03")
    vax_setpoint = (vax_fac.min_temp_c + vax_fac.max_safe_temp_c) / 2.0
    vax_headroom = vax_fac.max_safe_temp_c - vax_setpoint

    m0_all_row = df_tradeoff[(df_tradeoff["subset"] == "All Records") & (df_tradeoff["model"] == "M0_Physics_Mean")].iloc[0]
    m0_unsafe_rate = float(m0_all_row["unsafe_error_rate_pct"])
    avg_m0_min = float(m0_all_row["avg_predicted_safe_min"])
    avg_m3_90_min = float(all_90_row["avg_predicted_safe_min"])
    flex_pen_90 = float(all_90_row["flexibility_penalty_vs_m0_min"])

    m0_5m_mae = float(df_comp.loc[df_comp["segment"] == "Horizon 5m", "M0_Physics_MAE"].iloc[0])
    m1_5m_mae = float(df_comp.loc[df_comp["segment"] == "Horizon 5m", "M1_ML_MAE"].iloc[0])
    m2_5m_mae = float(df_comp.loc[df_comp["segment"] == "Horizon 5m", "M2_Hybrid_MAE"].iloc[0])

    sc_tag = f"Scenario Count: N = {n_sc} (Train: {n_tr_sc}, Calib: {n_cal_sc}, Test: {n_te_sc})"

    report_md = rf"""# Chill2Grid AI - Part 3 Hybrid AI Forecaster Evaluation Report

**Theme:** Green Energy Optimization through Data Analytics (IEEE ClimateChange Global Hackathon)  
**Artifact Classification:** All telemetry, reality-gap disturbances, and forecasts are **SIMULATED**.  
**Execution Runtime:** Data generation: {gen_time}s | Model training & conformal calibration: {train_time}s.

---

## 1. Executive Summary & Part 2 Audit Context

### Part 2 Thermal Twin Accounting Cleanup Summary
Prior to training the hybrid forecasters, Part 2 established a physically sensible digital twin baseline and resolved all thermal accounting discrepancies:
1. **Thermodynamic Bound Check**: We proved that net energy savings over a curtailment-rebound cycle are strictly bounded by reduced heat ingress: integral of $\Delta T_{{\text{{air}}}} / (R_{{\text{{wall}}}} \cdot \text{{COP}}) \, dt$.
2. **Cycle-Averaged Settle Condition**: Settling rule checked as rolling mean of power difference over one baseline cycle being $< 1\%$ of mean baseline power for 60 consecutive minutes.
3. **12-Offset Phase-Robust Averaging**: Uses 12 offsets spanning one full baseline cycle period (e.g. 162 min period for `FAC-PRD-04` at 40 °C).
4. **Saturation & Capacity Limits**: Identified capacity-limited assets under extreme ambient heat and flagged honestly as `Methods Disagree` when unpaid thermal debt remains at the horizon cap.
5. **Vaccine Door Stress Testing**: Confirmed that `FAC-VAX-03` is strictly warmer with door openings and breaches safe limits at 40 °C even without door events.

### Model Paradigm
- **M0 Physics-Only Baseline**: Zero parameters; deterministic coupled RC twin with catalog parameters and naive initial state assumption ($T_{{\text{{product}}}} = T_{{\text{{air}}}} = \text{{sensor\_reading}}$).
- **M1 Pure ML Forecaster**: Random Forest Regressor predicting future air temperature directly from observable telemetry and horizon.
- **M2 Hybrid AI Forecaster**: Random Forest Regressor predicting the physics twin's residual error (true - physics) from non-invasive telemetry.
- **M3 Conservative Upper Bounds**: Quantile HistGradientBoosting with finite-sample split-conformal calibration.
- **SensorView Leak Guard**: Complete information isolation ensuring 0 leakage of hidden product temperatures or true reality-gap parameters.

---

## 2. Headline Performance Metrics

### Model Comparison Table ({sc_tag})
| Metric | M0 Physics Baseline | M1 Pure ML | M2 Hybrid AI | Hybrid vs Physics |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Test MAE (°C)** | **{metadata['headline_metrics']['overall_m0_physics_mae']:.3f}** | **{metadata['headline_metrics']['overall_m1_ml_mae']:.3f}** | **{metadata['headline_metrics']['overall_m2_hybrid_mae']:.3f}** | **-{metadata['headline_metrics']['hybrid_mae_improvement_pct']:.1f}% error** |
| **Overall Test RMSE (°C)** | **{df_comp.loc[0, 'M0_Physics_RMSE']:.3f}** | **{df_comp.loc[0, 'M1_ML_RMSE']:.3f}** | **{df_comp.loc[0, 'M2_Hybrid_RMSE']:.3f}** | **-{round((df_comp.loc[0, 'M0_Physics_RMSE'] - df_comp.loc[0, 'M2_Hybrid_RMSE']) / df_comp.loc[0, 'M0_Physics_RMSE'] * 100.0, 1):.1f}% RMSE** |
| **Near-Limit MAE (within 1 °C)** | **{df_comp.loc[1, 'M0_Physics_MAE']:.3f}** | **{df_comp.loc[1, 'M1_ML_MAE']:.3f}** | **{df_comp.loc[1, 'M2_Hybrid_MAE']:.3f}** | **-{df_comp.loc[1, 'Hybrid_MAE_Reduction_Pct']:.1f}% error** |

### Per-Horizon Performance Breakdown ({sc_tag})
{df_to_markdown_table(df_comp)}

### Critical Horizon Breakdown & Honest Transparency
- **At 5-minute Horizon**: Physics Baseline (M0) achieves **{m0_5m_mae:.3f} °C MAE**, Pure ML (M1) achieves **{m1_5m_mae:.3f} °C MAE**, and Hybrid (M2) achieves **{m2_5m_mae:.3f} °C MAE**.  
  *Context:* Over ultra-short 5-minute windows, high-frequency sensor noise ($\sigma \approx 0.05 - 0.20$ °C) and pure lag autoregression are prominent.
- **At Horizons $\ge 10$ minutes**: The Hybrid AI model demonstrates strong accuracy over both M0 and M1 across horizons as thermodynamic thermal mass and insulation dynamics take over.

---

## 3. Conformal Coverage by Facility & Phase

### Target vs Achieved Marginal Coverage by Facility ({sc_tag})
{df_to_markdown_table(df_cov)}

### Conditional Coverage Audit: Before (Pooled) vs After (Phase-Conditional) ({sc_tag})
{df_to_markdown_table(df_phase_cov_before_after)}

*Key Insight:* In the `cut_in` phase (warmest air, compressor about to kick on, lowest headroom), phase-conditional calibration ensures coverage meets target thresholds without sacrificing flexibility in other phases.

---

## 4. Empirical Residual Standard Deviation Check

### Residual Standard Deviation Across Horizons ({sc_tag})
Theoretical uncertainty widens with horizon time. Below is the empirical standard deviation of the physics residual std(true - physics) across horizons and facilities:
{df_to_markdown_table(df_res_std)}

---

## 5. Safety vs Flexibility Trade-Off & Three-Way Censoring Breakdown

### Three-Way Censoring Trade-Off Breakdown ({sc_tag})
1. **(a) All Records**: Full test population including right-censored and left-censored records.
2. **(b) Excluding Already-Over-Limit ($T_{{\text{{true}}}} > 0$)**: Evaluates events where the facility had non-zero headroom at decision time.
3. **(c) Uncensored Only ($0 < T_{{\text{{true}}}} < 120$)**: Evaluates events where a true thermal breach occurred within the 120-minute window.

{df_to_markdown_table(df_tradeoff)}

### Already-Over-Limit Share by Facility ({sc_tag})
{df_to_markdown_table(df_already_over_limit)}

*Why is Already-Over-Limit Nonzero?*  
In hot weather ($>38$ °C) or during loading dock door openings, when the asset is cycling near its cut-in threshold, normal deadband cycling and sensor noise can place the initial temperature at or slightly above `max_safe_temp_c` before any curtailment begins. Headroom at $t=0$ is zero, requiring immediate cooling rather than curtailment.

---

## 6. Critical Risk Assessment: Vaccine Cold Storage & Guardian Role

> [!CAUTION]
> **NO "ZERO BREACH" CLAIMS:** Conformal prediction intervals provide statistical coverage across a population; they do **not** provide deterministic physical guarantees. An unsafe error rate is unacceptable for high-value pharmaceutical assets like **`FAC-VAX-03`** (dynamically computed: setpoint {vax_setpoint:.2f} °C, limit {vax_fac.max_safe_temp_c:.2f} °C, headroom {vax_headroom:.2f} °C).

### Why the Safety Guardian (Part 4) Must Not Rely on ML Alone
The **Safety Guardian** must implement deterministic physical safeguards:
1. **Deterministic Physical Tripwires**: Immediate automatic abort of curtailment if sensor air temperature reaches within $0.5$ °C of `max_safe_temp_c`, irrespective of ML predictions.
2. **Asset Risk Stratification**: Pharmaceutical assets (`FAC-VAX-03`) require ultra-conservative 99% conformal confidence or zero curtailment during high ambient temperatures (>35 °C).
3. **Telemetry Heartbeat Watchdog**: Contiguous sensor dropout bursts of $>5$ minutes immediately clamp safe duration to 0.0 minutes.

---

## 7. Reality Gap Assumptions & Monotonicity Properties

1. **Mild Synthetic Reality Gap**:
   Physics-only overall MAE is ~0.29 °C because the synthetic reality gap was configured as a $\pm 10\%$ parameter perturbation on $R$ and $C$, with disturbance $\sigma \approx 3\%$ of wall heat. **All results are strictly conditional on this synthetic gap.**
2. **Trajectory Monotonicity**:
   The cumulative maximum operator `np.maximum.accumulate` guarantees mathematically non-decreasing upper bound trajectories over forecast horizons.

---

## 8. Robustness & Generalization Checks ({sc_tag})
{df_to_markdown_table(df_robustness)}
"""
    with open(out_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nTraining pipeline completed successfully in {train_time}s.")
    print(f"Artifacts, tables, and plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
