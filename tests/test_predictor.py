"""test_predictor.py - Inference API, fail-safe logic, latency, and calibration tests."""

from pathlib import Path
import time
import numpy as np
import pytest

from core.contracts import FacilityState, load_facilities_from_csv
from models.temperature_predictor import (
    estimate_safe_minutes,
    load_predictor,
    predict_temperature,
    predict_trajectory,
    predict_trajectory_batch,
)


@pytest.fixture
def calibrated_facilities():
    project_root = Path(__file__).resolve().parent.parent
    cal_csv = project_root / "data" / "facilities_calibrated.csv"
    if not cal_csv.is_file():
        cal_csv = project_root / "data" / "facilities.csv"
    return load_facilities_from_csv(cal_csv)


def test_missing_model_files_clear_error(tmp_path: Path) -> None:
    """Requirement j: Missing model files raise a clear FileNotFoundError with instructions."""
    with pytest.raises(FileNotFoundError, match="python -m training.train_model"):
        load_predictor(artifacts_dir=tmp_path)


def test_metadata_feature_mismatch_clear_error(tmp_path: Path) -> None:
    """Requirement k: Metadata feature mismatch raises a clear ValueError."""
    # Create mock artifacts directory with invalid metadata
    (tmp_path / "m1_ml_only.joblib").write_bytes(b"mock")
    (tmp_path / "m2_hybrid.joblib").write_bytes(b"mock")
    (tmp_path / "m3_quantile_models.joblib").write_bytes(b"mock")
    (tmp_path / "conformal_offsets.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metadata.json").write_text('{"m2_features": ["invalid_feat"]}', encoding="utf-8")

    with pytest.raises(ValueError, match="Metadata feature mismatch"):
        load_predictor(artifacts_dir=tmp_path)


def test_inference_and_conformal_bounds(calibrated_facilities) -> None:
    """Requirements e & g: Non-decreasing physics, Upper >= Mean, and monotone bounds."""
    fac = calibrated_facilities[0]
    setp = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
    state = FacilityState(facility_id=fac.id, temp_c=setp, outside_temp_c=35.0, humidity_pct=60.0, product_load_fraction=0.85, door_openings_per_hour=2.0, compressor_on=True, sensor_ok=True)
    history = [setp] * 65

    forecast = predict_trajectory(fac, state, history)
    assert forecast.valid is True

    # Requirement e: Physics predictor is non-decreasing over horizon with cooling forced off
    for i in range(len(forecast.physics_c) - 1):
        assert forecast.physics_c[i + 1] >= forecast.physics_c[i] - 1e-4

    # Requirement g: Upper >= Mean and Upper is monotone over horizons
    for lvl, up_traj in forecast.upper_c.items():
        for i in range(len(up_traj)):
            assert up_traj[i] >= forecast.mean_c[i] - 1e-4
        for i in range(len(up_traj) - 1):
            assert up_traj[i + 1] >= up_traj[i] - 1e-4


def test_failsafe_sensor_diagnostics(calibrated_facilities) -> None:
    """Requirement h: More than 5 consecutive missing readings trigger fail-safe valid=False or safe_minutes=0."""
    fac = calibrated_facilities[0]
    setp = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
    state = FacilityState(facility_id=fac.id, temp_c=setp, outside_temp_c=35.0, humidity_pct=60.0, product_load_fraction=0.85, door_openings_per_hour=2.0, compressor_on=True, sensor_ok=True)

    # Corrupt history: 7 consecutive missing readings
    bad_history = [setp] * 50 + [None] * 7 + [setp] * 8
    forecast = predict_trajectory(fac, state, bad_history)
    assert forecast.valid is False
    assert "Sensor dropout" in forecast.reason

    safe_min, reason = estimate_safe_minutes(fac, state, bad_history)
    assert safe_min == 0.0
    assert "dropout" in reason.lower()

    # Temperature already at limit
    limit_state = FacilityState(facility_id=fac.id, temp_c=fac.max_safe_temp_c, outside_temp_c=35.0, humidity_pct=60.0, product_load_fraction=0.85, door_openings_per_hour=2.0, compressor_on=True, sensor_ok=True)
    safe_min_lim, reason_lim = estimate_safe_minutes(fac, limit_state, [fac.max_safe_temp_c] * 65)
    assert safe_min_lim == 0.0


def test_temperature_sensitivity_to_weather(calibrated_facilities) -> None:
    """Requirement i: Hotter outside temperature yields higher predicted temperature on average."""
    fac = calibrated_facilities[0]
    setp = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
    rng = np.random.default_rng(42)

    mild_preds = []
    hot_preds = []

    for _ in range(50):
        t_rand = float(setp + rng.uniform(-1.0, 1.0))
        hist = [t_rand] * 65

        st_mild = FacilityState(facility_id=fac.id, temp_c=t_rand, outside_temp_c=24.0, humidity_pct=50.0, product_load_fraction=0.8, door_openings_per_hour=2.0, compressor_on=True, sensor_ok=True)
        st_hot = FacilityState(facility_id=fac.id, temp_c=t_rand, outside_temp_c=42.0, humidity_pct=50.0, product_load_fraction=0.8, door_openings_per_hour=2.0, compressor_on=True, sensor_ok=True)

        m_mild, _ = predict_temperature(fac, st_mild, minutes=30, sensor_history_c=hist)
        m_hot, _ = predict_temperature(fac, st_hot, minutes=30, sensor_history_c=hist)

        mild_preds.append(m_mild)
        hot_preds.append(m_hot)

    assert np.mean(hot_preds) > np.mean(mild_preds)


def test_batch_inference_latency_target(calibrated_facilities) -> None:
    """Requirement l: Batch trajectory prediction for all 6 facilities completes in < 200 ms."""
    load_predictor()  # Warm cache

    states = []
    histories = []
    for fac in calibrated_facilities:
        setp = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
        states.append(FacilityState(facility_id=fac.id, temp_c=setp, outside_temp_c=35.0, humidity_pct=60.0, product_load_fraction=0.85, door_openings_per_hour=2.0, compressor_on=True, sensor_ok=True))
        histories.append([setp] * 65)

    # Measure latency over 5 iterations
    latencies = []
    for _ in range(5):
        t0 = time.time()
        res = predict_trajectory_batch(calibrated_facilities, states, histories)
        latencies.append(time.time() - t0)
        assert len(res) == len(calibrated_facilities)

    avg_latency_ms = float(np.mean(latencies) * 1000.0)
    assert avg_latency_ms < 200.0, f"Batch latency {avg_latency_ms:.1f} ms exceeds 200 ms target!"
