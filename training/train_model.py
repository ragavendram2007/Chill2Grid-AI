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
    Dict[float, Dict[int, float]],
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
    conformal_offsets: Dict[float, Dict[int, float]] = {}

    X_m2_calib = calib_df[list(M2_FEATURES)]
    y_res_calib = calib_df["residual_c"].values

    for level in CONFIDENCE_LEVELS:
        # Quantile gradient boosting
        q_model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=level,
            max_iter=50 if quick else 100,
            random_state=42,
        )
        q_model.fit(X_m2_tr, y_m2_tr)
        m3_models[level] = q_model

        # Split-conformal calibration per horizon
        raw_calib_pred = q_model.predict(X_m2_calib)
        conformal_offsets[level] = {}

        for h in FORECAST_HORIZONS:
            h_mask = (calib_df["horizon_min"] == h).values
            scores_h = y_res_calib[h_mask] - raw_calib_pred[h_mask]
            n_calib = len(scores_h)
            if n_calib > 0:
                # Finite-sample conformal correction quantile
                k_quantile = min(1.0, np.ceil((n_calib + 1) * level) / n_calib)
                offset = float(np.quantile(scores_h, k_quantile))
            else:
                offset = 0.0
            conformal_offsets[level][h] = round(offset, 4)

    training_duration_sec = round(time.time() - start_time, 2)
    return m1_rf, m2_rf, m3_models, conformal_offsets, training_duration_sec


# ============================================================================
# EVALUATION & METRICS PIPELINE
# ============================================================================

def evaluate_models(
    test_df: pd.DataFrame,
    m1_model: RandomForestRegressor,
    m2_model: RandomForestRegressor,
    m3_models: Dict[float, HistGradientBoostingRegressor],
    conformal_offsets: Dict[float, Dict[int, float]],
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Perform rigorous evaluation on the held-out test split."""
    test_eval = test_df.copy()

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

    # M3 Upper Bounds
    for level in CONFIDENCE_LEVELS:
        q_mod = m3_models[level]
        raw_upper = q_mod.predict(X_m2)
        # Apply per-horizon conformal offset
        offsets = test_eval["horizon_min"].map(conformal_offsets[level]).values
        calibrated_upper = test_eval["physics_temp_c"].values + raw_upper + offsets
        # Enforce upper bound >= mean hybrid prediction
        test_eval[f"upper_{int(level*100)}"] = np.maximum(calibrated_upper, test_eval["pred_m2"].values)

    # -------------------------------------------------------------------------
    # 1. Model Comparison Table (M0, M1, M2)
    # -------------------------------------------------------------------------
    def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        return round(mae, 3), round(rmse, 3)

    comp_rows = []
    # Overall
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
    # 2. Conformal Coverage Table
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
        # Facility breakdown
        for fac_id in sorted(test_eval["facility_id"].unique()):
            fac_mask = test_eval["facility_id"] == fac_id
            fac_cov = float(np.mean(test_eval.loc[fac_mask, "true_temp_c"] <= test_eval.loc[fac_mask, lvl_key]))
            row[f"{fac_id}_coverage"] = round(fac_cov, 4)
        cov_rows.append(row)
    df_coverage = pd.DataFrame(cov_rows)

    # -------------------------------------------------------------------------
    # 3. Safe Window Decision Accuracy & Tradeoff Table
    # -------------------------------------------------------------------------
    # Group test_eval by (scenario_id, record_id) to reconstruct time trajectories
    tradeoff_rows = []
    record_groups = test_eval.groupby(["scenario_id", "record_id"])

    # Extract true limit and physics baseline for each record
    records_meta = []
    for (sc_id, rec_id), grp in record_groups:
        grp_sorted = grp.sort_values("horizon_min")
        horizons = grp_sorted["horizon_min"].tolist()
        max_safe = float(grp_sorted["max_safe_temp_c"].iloc[0])
        true_limit = float(grp_sorted["true_minutes_to_limit"].iloc[0])
        phys_limit = float(grp_sorted["physics_minutes_to_limit"].iloc[0])

        record_entry = {
            "scenario_id": sc_id,
            "record_id": rec_id,
            "facility_id": grp_sorted["facility_id"].iloc[0],
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

    # Physics Baseline M0 Safe Window Evaluation
    m0_unsafe_cnt = int(np.sum(df_records_safe["physics_safe_min"] > df_records_safe["true_minutes_to_limit"] + 1e-3))
    m0_unsafe_rate = round(m0_unsafe_cnt / len(df_records_safe) * 100.0, 2)
    m0_flex_given_up = round(float(np.mean(np.maximum(0.0, df_records_safe["true_minutes_to_limit"] - df_records_safe["physics_safe_min"]))), 2)

    tradeoff_rows.append({
        "model": "M0_Physics_Mean",
        "confidence_level": 0.50,
        "unsafe_error_rate_pct": m0_unsafe_rate,
        "flexibility_given_up_min": m0_flex_given_up,
        "avg_predicted_safe_min": round(float(df_records_safe["physics_safe_min"].mean()), 1),
    })

    # Hybrid Conservative Bounds
    for level in CONFIDENCE_LEVELS:
        lvl_key = f"safe_min_{int(level*100)}"
        unsafe_cnt = int(np.sum(df_records_safe[lvl_key] > df_records_safe["true_minutes_to_limit"] + 1e-3))
        unsafe_rate = round(unsafe_cnt / len(df_records_safe) * 100.0, 2)
        flex_given_up = round(float(np.mean(np.maximum(0.0, df_records_safe["true_minutes_to_limit"] - df_records_safe[lvl_key]))), 2)
        tradeoff_rows.append({
            "model": f"M3_Hybrid_Upper_{int(level*100)}%",
            "confidence_level": level,
            "unsafe_error_rate_pct": unsafe_rate,
            "flexibility_given_up_min": flex_given_up,
            "avg_predicted_safe_min": round(float(df_records_safe[lvl_key].mean()), 1),
        })

    df_tradeoff = pd.DataFrame(tradeoff_rows)
    return test_eval, df_comparison, df_coverage, df_records_safe, df_tradeoff


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

        # Fit M2 Hybrid on train_sub
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

    # 1. Dataset Generation
    t0 = time.time()
    df_dataset = generate_full_dataset(facilities, n_scenarios=n_sc, n_jobs=-1)
    gen_time = round(time.time() - t0, 2)
    print(f"Dataset generated: {len(df_dataset)} rows across 12 horizons in {gen_time}s.")

    # 2. Scenario-Stratified Split
    train_df, calib_df, test_df = split_dataset_by_scenario(df_dataset, train_frac=0.60, calib_frac=0.20)
    print(f"Split sizes: Train={len(train_df)} ({len(train_df['scenario_id'].unique())} scenarios) | "
          f"Calibration={len(calib_df)} ({len(calib_df['scenario_id'].unique())} scenarios) | "
          f"Test={len(test_df)} ({len(test_df['scenario_id'].unique())} scenarios)")

    # 3. Model Training & Conformal Calibration
    m1_rf, m2_rf, m3_models, conformal_offsets, train_time = train_all_models(train_df, calib_df, quick=args.quick)
    print(f"Models trained and calibrated in {train_time}s.")

    # 4. Evaluation
    test_eval, df_comp, df_cov, df_records_safe, df_tradeoff = evaluate_models(
        test_df, m1_rf, m2_rf, m3_models, conformal_offsets
    )

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

    metadata = {
        "model_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": "3.14.3",
        "sklearn_version": sklearn.__version__,
        "base_seed": 42,
        "n_scenarios": n_sc,
        "observable_features": list(OBSERVABLE_FEATURES),
        "m1_features": list(M1_FEATURES),
        "m2_features": list(M2_FEATURES),
        "horizons": FORECAST_HORIZONS,
        "headline_metrics": {
            "overall_m0_physics_mae": float(df_comp.loc[0, "M0_Physics_MAE"]),
            "overall_m1_ml_mae": float(df_comp.loc[0, "M1_ML_MAE"]),
            "overall_m2_hybrid_mae": float(df_comp.loc[0, "M2_Hybrid_MAE"]),
            "hybrid_mae_improvement_pct": float(df_comp.loc[0, "Hybrid_MAE_Reduction_Pct"]),
            "coverage_90pct_achieved": float(df_cov.loc[df_cov["target_confidence"] == 0.90, "overall_coverage"].iloc[0]),
            "unsafe_error_rate_90pct": float(df_tradeoff.loc[df_tradeoff["confidence_level"] == 0.90, "unsafe_error_rate_pct"].iloc[0]),
            "flexibility_given_up_min_90pct": float(df_tradeoff.loc[df_tradeoff["confidence_level"] == 0.90, "flexibility_given_up_min"].iloc[0]),
        },
    }
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 7. Save Output CSV Artifacts
    out_dir = project_root / "outputs" / "part3"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_comp.to_csv(out_dir / "model_comparison.csv", index=False)
    df_cov.to_csv(out_dir / "coverage_table.csv", index=False)
    df_records_safe.to_csv(out_dir / "safe_window_errors.csv", index=False)
    df_tradeoff.to_csv(out_dir / "tradeoff_table.csv", index=False)
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
    sub_to = df_tradeoff[df_tradeoff["model"].str.contains("Hybrid")].copy()
    fig_tf.add_trace(go.Scatter(x=sub_to["confidence_level"], y=sub_to["unsafe_error_rate_pct"], name="Unsafe Error Rate (%)", line=dict(color="red", width=2.5)), secondary_y=False)
    fig_tf.add_trace(go.Scatter(x=sub_to["confidence_level"], y=sub_to["flexibility_given_up_min"], name="Flexibility Given Up (min)", line=dict(color="blue", width=2.5, dash="dot")), secondary_y=True)
    fig_tf.update_layout(title="Safety vs Flexibility Trade-off across Confidence Levels (Simulated)", xaxis_title="Nominal Confidence Level", template="plotly_white")
    fig_tf.update_yaxes(title_text="Unsafe Error Rate (%)", secondary_y=False)
    fig_tf.update_yaxes(title_text="Flexibility Given Up (minutes)", secondary_y=True)
    fig_tf.write_html(str(out_dir / "plot_safety_flex_tradeoff.html"))

    # D. plot_feature_importance.html
    importances = m2_rf.feature_importances_
    feat_df = pd.DataFrame({"feature": list(M2_FEATURES), "importance": importances}).sort_values("importance", ascending=True).tail(15)
    fig_imp = px.bar(feat_df, x="importance", y="feature", orientation="h", title="Top 15 Hybrid Feature Importances (M2 Residual Learner, Simulated)")
    fig_imp.update_layout(template="plotly_white")
    fig_imp.write_html(str(out_dir / "plot_feature_importance.html"))

    # E. plot_trajectory_examples.html (4 diverse cases)
    sample_records = df_records_safe.sample(n=min(4, len(df_records_safe)), random_state=123)
    fig_cases = make_subplots(rows=2, cols=2, subplot_titles=[f"Case {i+1}: {r['facility_id']} (Safe Limit: {r['max_safe_temp_c']} °C)" for i, (_, r) in enumerate(sample_records.iterrows())])

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
    m0_unsafe_rate = float(df_tradeoff.loc[df_tradeoff["model"] == "M0_Physics_Mean", "unsafe_error_rate_pct"].iloc[0])
    report_md = rf"""# Chill2Grid AI - Part 3 Hybrid AI Forecaster Evaluation Report

**Theme:** Green Energy Optimization through Data Analytics (IEEE ClimateChange Global Hackathon)  
**Artifact Classification:** All telemetry, reality-gap disturbances, and forecasts are **SIMULATED**.

---

## 1. Executive Summary & Headline Metrics
- **Model Paradigm:** Hybrid Physics + Machine Learning Residual Forecaster.
  - A first-order coupled RC thermal twin computes a deterministic temperature trajectory.
  - A Random Forest regressor predicts the physics model's residual error from non-invasive telemetry.
  - An ensemble of Quantile Gradient Boosting regressors with split-conformal calibration generates provably conservative upper bounds.
- **Overall Performance Comparison (Test Split):**
  - **M0 Physics-Only Baseline:** MAE = **{metadata['headline_metrics']['overall_m0_physics_mae']:.3f} °C** (RMSE = {df_comp.loc[0, 'M0_Physics_RMSE']:.3f} °C)
  - **M1 Pure ML Forecaster:** MAE = **{metadata['headline_metrics']['overall_m1_ml_mae']:.3f} °C** (RMSE = {df_comp.loc[0, 'M1_ML_RMSE']:.3f} °C)
  - **M2 Hybrid AI Model:** MAE = **{metadata['headline_metrics']['overall_m2_hybrid_mae']:.3f} °C** (RMSE = {df_comp.loc[0, 'M2_Hybrid_RMSE']:.3f} °C)
  - **Hybrid Improvement over Physics:** **{metadata['headline_metrics']['hybrid_mae_improvement_pct']:.1f}% reduction in MAE**.
- **Conservative 90% Bound & Safe Flexibility Window:**
  - **Achieved Test Coverage (90% Nominal):** **{metadata['headline_metrics']['coverage_90pct_achieved'] * 100:.2f}%**.
  - **Unsafe Decision Error Rate (Physics Mean M0):** **{m0_unsafe_rate:.1f}%** of records overestimate safe curtailment time.
  - **Unsafe Decision Error Rate (90% Upper Bound M3):** **{metadata['headline_metrics']['unsafe_error_rate_90pct']:.1f}%** (drastically reduces spoilage risk).
  - **Flexibility Given Up for Safety:** Average penalty of **{metadata['headline_metrics']['flexibility_given_up_min_90pct']:.1f} minutes** to protect cargo.

---

## 2. Model Comparison Table
{df_to_markdown_table(df_comp)}

---

## 3. Conformal Coverage by Facility
{df_to_markdown_table(df_cov)}

---

## 4. Safety vs Flexibility Trade-Off
{df_to_markdown_table(df_tradeoff)}

---

## 5. Robustness & Generalization Checks
{df_to_markdown_table(df_robustness)}

---

## 6. Assumptions & Real-World Translation
1. **Reality-Gap Disturbance Assumption:**
   Structural facility parameters ($R$ and $C$) deviate by $\pm 10\%$ from blueprints, and internal heat disturbances follow an AR(1) process with $\sigma \approx 3\%$ of wall heat plus loading activity spikes.
2. **Telemetry Constraint:**
   The AI observes *only* noisy air temperature sensors, compressor state history, ambient weather, and facility specifications. Internal product temperatures are completely hidden.
3. **Real-World Deployment Requirements:**
   In actual cold-storage facilities, historical sub-metering, loading dock door magnet sensors, and ambient weather station feeds would replace synthetic perturbations.
"""
    with open(out_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\nTraining pipeline completed successfully in {train_time}s.")
    print(f"Artifacts and plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
