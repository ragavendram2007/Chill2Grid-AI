"""test_training.py - Unit and regression tests for scenario generation, splits, and training."""

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
    state = FacilityState(facility_id=fac.id, temp_c=setp, outside_temp_c=35.0, humidity_pct=50.0, product_load_fraction=0.85, door_openings_per_hour=0.0, compressor_on=False, sensor_ok=True)

    # 1. Default run
    df_def = simulate_temperature(fac, state, duration_min=30)
    assert df_def["air_temp_c"].iloc[0] == round(setp, 3)

    # 2. Initial temperature overrides
    df_over = simulate_temperature(
        fac, state, duration_min=30,
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
