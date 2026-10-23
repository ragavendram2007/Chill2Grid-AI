"""test_training.py - Unit and regression tests for scenario generation, splits, and training."""

import json
from pathlib import Path
import numpy as np
import pytest

from core.contracts import load_facilities_from_csv
from simulation.reality_gap import make_disturbance, make_world_facility
from simulation.thermal_model import simulate_temperature
from training.scenario_generator import (
    FORECAST_HORIZONS,
    FORBIDDEN_FEATURES,
    M1_FEATURES,
    M2_FEATURES,
    OBSERVABLE_FEATURES,
    SensorView,
    generate_data_quality_report,
    generate_full_dataset,
    split_dataset_by_scenario,
)


@pytest.fixture
def calibrated_facilities():
    project_root = Path(__file__).resolve().parent.parent
    cal_csv = project_root / "data" / "facilities_calibrated.csv"
    if not cal_csv.is_file():
        cal_csv = project_root / "data" / "facilities.csv"
    return load_facilities_from_csv(cal_csv)


def test_leakage_prevention() -> None:
    """Requirement c: Strict verification that forbidden features never leak into training sets."""
    for feat in FORBIDDEN_FEATURES:
        assert feat not in OBSERVABLE_FEATURES, f"Forbidden feature '{feat}' found in OBSERVABLE_FEATURES!"
        assert feat not in M1_FEATURES, f"Forbidden feature '{feat}' found in M1_FEATURES!"
        assert feat not in M2_FEATURES, f"Forbidden feature '{feat}' found in M2_FEATURES!"


def test_reality_gap_properties(calibrated_facilities) -> None:
    """Verify reality gap perturbations and disturbance time-series properties."""
    fac = calibrated_facilities[0]
    rng = np.random.default_rng(42)

    # World facility perturbation in [0.90, 1.10]
    world_fac = make_world_facility(fac, rng)
    r_ratio = world_fac.insulation_r_deg_c_per_kw / fac.insulation_r_deg_c_per_kw
    c_ratio = world_fac.thermal_mass_kwh_per_c / fac.thermal_mass_kwh_per_c
    assert 0.89 <= r_ratio <= 1.11
    assert 0.89 <= c_ratio <= 1.11

    # Disturbance AR(1) series std is ~3% of mean wall heat
    dist = make_disturbance(rng, n_minutes=300, mean_wall_heat_kw=100.0)
    assert len(dist) == 300
    std_kw = float(np.std(dist))
    assert 1.5 <= std_kw <= 4.5  # Bounded around 3.0 kW


def test_new_simulator_options_backward_compatibility(calibrated_facilities) -> None:
    """Requirement d: Verify new simulator options work while keeping original defaults unchanged."""
    fac = calibrated_facilities[0]
    from core.contracts import FacilityState

    setp = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
    state = FacilityState(
        facility_id=fac.id,
        temp_c=setp,
        outside_temp_c=35.0,
        humidity_pct=50.0,
        product_load_fraction=0.85,
        door_openings_per_hour=0.0,
        compressor_on=False,
        sensor_ok=True,
    )

    # 1. Default run
    df_def = simulate_temperature(fac, state, duration_min=30)
    assert df_def["air_temp_c"].iloc[0] == round(setp, 3)

    # 2. Initial temperature overrides
    df_over = simulate_temperature(
        fac,
        state,
        duration_min=30,
        initial_air_temp_c=-15.0,
        initial_product_temp_c=-14.0,
    )
    assert df_over["air_temp_c"].iloc[0] == -15.0
    assert df_over["product_temp_c"].iloc[0] == -14.0

    # 3. Expected door heat
    df_exp_door = simulate_temperature(fac, state, duration_min=30, expected_door_heat=True)
    assert not df_exp_door["door_event"].any()


def test_dataset_reproducibility_and_stratified_splits(calibrated_facilities) -> None:
    """Requirements a & b: Seed repeatability and scenario-stratified disjoint splits."""
    # Generate small dataset
    df1 = generate_full_dataset(calibrated_facilities[:2], n_scenarios=4, records_per_scenario=2, base_seed=123, n_jobs=1)
    df2 = generate_full_dataset(calibrated_facilities[:2], n_scenarios=4, records_per_scenario=2, base_seed=123, n_jobs=1)

    # Exact bitwise equality
    assert df1["true_temp_c"].equals(df2["true_temp_c"])
    assert df1["residual_c"].equals(df2["residual_c"])

    # Disjoint stratified splits
    tr, cal, te = split_dataset_by_scenario(df1, train_frac=0.50, calib_frac=0.25, seed=42)
    s_tr = set(tr["scenario_id"])
    s_cal = set(cal["scenario_id"])
    s_te = set(te["scenario_id"])

    assert s_tr.isdisjoint(s_cal)
    assert s_tr.isdisjoint(s_te)
    assert s_cal.isdisjoint(s_te)


def test_sensor_view_leak_guard() -> None:
    """Requirement c: SensorView strictly isolates hidden states and raises on forbidden access."""
    view = SensorView(
        sensor_air_temp_c=-18.0,
        lag_0=-18.0,
        lag_1=-18.1,
        lag_2=-18.2,
        lag_5=-18.3,
        lag_10=-18.4,
        lag_15=-18.5,
        lag_30=-18.6,
        lag_60=-18.8,
        slope_5min=0.06,
        slope_15min=0.033,
        min_30min=-18.6,
        max_30min=-18.0,
        compressor_on=1.0,
        minutes_in_compressor_state=12,
        minutes_since_curtailment=600.0,
        outside_temp_c=35.0,
        humidity_pct=60.0,
        product_load_fraction=0.85,
        door_rate=2.0,
        rated_power_kw=120.0,
        setpoint_c=-19.0,
        max_safe_temp_c=-15.0,
        min_temp_c=-23.0,
        thermal_mass=250.0,
        insulation_r=0.08,
        headroom_c=3.0,
        phase="cut_in",
        has_dropout_burst=False,
        dropout_burst_len=0,
    )
    # Observable features dict should match OBSERVABLE_FEATURES
    feat_dict = view.to_feature_dict()
    assert set(feat_dict.keys()) == set(OBSERVABLE_FEATURES)

    # Attempting to access forbidden hidden attributes raises AttributeError
    with pytest.raises(AttributeError, match="LeakGuardBlocked"):
        _ = view.product_temp_c

    with pytest.raises(AttributeError, match="LeakGuardBlocked"):
        _ = view.true_temp_c

    with pytest.raises(AttributeError, match="LeakGuardBlocked"):
        _ = view.disturbance


def test_phase_stratification_and_censoring(calibrated_facilities) -> None:
    """Verify phase-stratified sampling and label censoring flags."""
    df = generate_full_dataset(calibrated_facilities[:2], n_scenarios=4, records_per_scenario=6, base_seed=42, n_jobs=1)

    assert "phase" in df.columns
    assert "is_right_censored" in df.columns
    assert "is_left_censored" in df.columns
    assert "has_dropout_burst" in df.columns

    phases = set(df["phase"].unique())
    assert phases.issubset({"cut_in", "cut_out", "mid_cycle"})
    assert len(phases) >= 2


def test_data_quality_report_generation(tmp_path: Path, calibrated_facilities) -> None:
    """Verify generation of data_quality_report.csv and data_quality_report.md."""
    df = generate_full_dataset(calibrated_facilities[:2], n_scenarios=2, records_per_scenario=3, base_seed=42, n_jobs=1)
    q_df, md_content = generate_data_quality_report(df, tmp_path)

    assert (tmp_path / "data_quality_report.csv").is_file()
    assert (tmp_path / "data_quality_report.md").is_file()
    assert "Phase-Stratified Sampling Audit" in md_content
    assert "Label Censoring Analysis" in md_content
    assert "SensorView Leak-Guard Audit" in md_content


def test_hybrid_mae_superiority_to_physics() -> None:
    """Requirement f: On the held-out test split, Hybrid MAE <= Physics-only MAE."""
    project_root = Path(__file__).resolve().parent.parent
    meta_path = project_root / "models" / "artifacts" / "metadata.json"
    if not meta_path.is_file():
        pytest.skip("Models not yet trained; skipping hybrid MAE test.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    m0_mae = meta["headline_metrics"]["overall_m0_physics_mae"]
    m2_mae = meta["headline_metrics"]["overall_m2_hybrid_mae"]
    assert m2_mae <= m0_mae, f"Hybrid MAE ({m2_mae:.3f}) exceeded Physics MAE ({m0_mae:.3f})!"
