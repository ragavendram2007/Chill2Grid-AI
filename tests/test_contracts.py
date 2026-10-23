"""test_contracts.py - Unit tests for core contracts, dataclasses, constants, and CSV ingestion."""

from pathlib import Path
import pytest

from core.contracts import (
    CO2_GRID_EMISSION_FACTOR_KG_PER_KWH,
    SIMULATED_REWARD_RATE_INR_PER_KWH,
    TIME_STEP_MIN,
    CurtailmentAction,
    EventResult,
    Facility,
    FacilityState,
    FlexibilityEstimate,
    SafetyDecision,
    format_facilities_table,
    load_facilities_from_csv,
)


def test_constants_defined() -> None:
    """Verify that required constants are defined with physical units and non-zero values."""
    assert TIME_STEP_MIN == 1.0
    assert CO2_GRID_EMISSION_FACTOR_KG_PER_KWH > 0.0
    assert pytest.approx(CO2_GRID_EMISSION_FACTOR_KG_PER_KWH, 0.001) == 0.716
    assert SIMULATED_REWARD_RATE_INR_PER_KWH == 8.00


def test_facility_valid() -> None:
    """Verify standard valid Facility initialization and convenience properties."""
    fac = Facility(
        id="FAC-TEST-01",
        name="Test Cold Storage",
        product_type="Frozen Food",
        thermal_mass_kwh_per_c=30.0,
        insulation_r_deg_c_per_kw=0.4,
        rated_power_kw=150.0,
        min_temp_c=-22.0,
        max_safe_temp_c=-18.0,
        min_on_min=10,
        min_off_min=5,
        cop_nominal=2.5,
    )
    assert fac.id == "FAC-TEST-01"
    assert fac.name == "Test Cold Storage"
    assert fac.thermal_mass == 30.0
    assert fac.insulation_r == 0.4
    assert fac.rated_power_kw == 150.0
    assert fac.min_temp_c == -22.0
    assert fac.max_safe_temp_c == -18.0


@pytest.mark.parametrize(
    "kwargs, expected_substring",
    [
        (
            {"id": "", "min_temp_c": -20.0, "max_safe_temp_c": -15.0},
            "id' must be a non-empty string",
        ),
        (
            {"id": "F1", "min_temp_c": -10.0, "max_safe_temp_c": -15.0},
            "must be strictly less than max_safe_temp_c",
        ),
        (
            {"id": "F1", "min_temp_c": -15.0, "max_safe_temp_c": -15.0},
            "must be strictly less than max_safe_temp_c",
        ),
        (
            {"id": "F1", "rated_power_kw": 0.0},
            "rated_power_kw (0.0 kW) must be positive",
        ),
        (
            {"id": "F1", "thermal_mass_kwh_per_c": -5.0},
            "thermal_mass_kwh_per_c (-5.0) must be positive",
        ),
        (
            {"id": "F1", "insulation_r_deg_c_per_kw": 0.0},
            "insulation_r_deg_c_per_kw (0.0) must be positive",
        ),
        (
            {"id": "F1", "min_on_min": -1},
            "min_on_min and min_off_min must be non-negative",
        ),
        (
            {"id": "F1", "cop_nominal": -2.0},
            "cop_nominal (-2.0) must be positive",
        ),
    ],
)
def test_facility_invalid_inputs(kwargs: dict, expected_substring: str) -> None:
    """Verify that physical constraints raise ValueError on invalid facility inputs."""
    defaults = {
        "id": "VALID-ID",
        "product_type": "Produce",
        "thermal_mass_kwh_per_c": 20.0,
        "insulation_r_deg_c_per_kw": 0.3,
        "rated_power_kw": 100.0,
        "min_temp_c": 1.0,
        "max_safe_temp_c": 5.0,
        "min_on_min": 5,
        "min_off_min": 5,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError) as exc_info:
        Facility(**defaults)
    assert expected_substring in str(exc_info.value)


def test_facility_state_validation() -> None:
    """Verify bounds checking on state telemetry."""
    # Valid state
    state = FacilityState(
        facility_id="FAC-01",
        temp_c=-20.5,
        outside_temp_c=32.0,
        humidity_pct=65.0,
        product_load_fraction=0.85,
        door_openings_per_hour=3.0,
        compressor_on=True,
        sensor_ok=True,
    )
    assert state.temp_c == -20.5
    assert state.sensor_ok is True

    # Invalid humidity
    with pytest.raises(ValueError, match="humidity_pct"):
        FacilityState(
            temp_c=0.0,
            outside_temp_c=25.0,
            humidity_pct=105.0,
            product_load_fraction=0.5,
            door_openings_per_hour=1.0,
            compressor_on=True,
            sensor_ok=True,
        )

    # Invalid product load
    with pytest.raises(ValueError, match="product_load_fraction"):
        FacilityState(
            temp_c=0.0,
            outside_temp_c=25.0,
            humidity_pct=50.0,
            product_load_fraction=-0.1,
            door_openings_per_hour=1.0,
            compressor_on=True,
            sensor_ok=True,
        )


def test_curtailment_action() -> None:
    """Verify CurtailmentAction attributes and constraints."""
    action = CurtailmentAction(facility_id="FAC-01", start_min=15, duration_min=45)
    assert action.facility_id == "FAC-01"
    assert action.start_min == 15
    assert action.duration_min == 45

    with pytest.raises(ValueError, match="duration_min"):
        CurtailmentAction(facility_id="FAC-01", start_min=0, duration_min=0)


def test_safety_decision() -> None:
    """Verify SafetyDecision creation and is_safe property."""
    decision_ok = SafetyDecision(
        facility_id="FAC-01",
        allowed=True,
        reason="Temperature remains 2.1 C below max threshold",
        worst_case_temp_c=-19.5,
        safe_duration_min=45.0,
    )
    assert decision_ok.is_safe is True
    assert decision_ok.worst_case_temp_c == -19.5

    decision_blocked = SafetyDecision(
        facility_id="FAC-02",
        allowed=False,
        reason="Sensor fault telemetry detected",
        worst_case_temp_c=0.0,
    )
    assert decision_blocked.is_safe is False


def test_flexibility_estimate() -> None:
    """Verify FlexibilityEstimate validation and alias properties."""
    est = FlexibilityEstimate(
        facility_id="FAC-01",
        baseline_power_kw=120.0,
        curtailable_power_kw=100.0,
        safe_duration_min=40.0,
        confidence_score=0.92,
        reason="High thermal mass permits 40m safe drift",
    )
    assert est.baseline_kw == 120.0
    assert est.curtailable_kw == 100.0
    assert est.safe_minutes == 40.0
    assert est.confidence == 0.92

    # Curtailable cannot exceed baseline (Rule C: Curtailable is limited by baseline power draw)
    with pytest.raises(ValueError, match="cannot exceed actual baseline_power_kw"):
        FlexibilityEstimate(
            facility_id="FAC-01",
            baseline_power_kw=80.0,
            curtailable_power_kw=100.0,
            safe_duration_min=30.0,
            confidence_score=0.85,
            reason="Invalid excessive curtailment",
        )


def test_event_result_accounting() -> None:
    """Verify EventResult metric consistency and honest accounting summary."""
    event = EventResult(
        event_name="Peak-Hour Grid Demand Cut",
        target_power_kw=500.0,
        delivered_power_kw=420.0,
        shortfall_power_kw=80.0,
        duration_min=45.0,
        event_energy_saved_kwh=315.0,
        recovery_energy_added_kwh=65.0,
        net_energy_saved_kwh=250.0,
        co2_reduction_estimate_kg=250.0 * 0.716,
        simulated_reward_inr=250.0 * 8.0,
        selected_facilities=["FAC-FZN-01", "FAC-PRD-04"],
        rejected_facilities={"FAC-VAX-03": "Tight safety margin veto"},
        peak_rebound_kw=480.0,
        rebound_exceeds_baseline=False,
    )
    assert event.shortfall_power_kw == 80.0
    assert pytest.approx(event.co2_reduction_estimate_kg, 0.1) == 179.0
    assert event.simulated_reward_inr == 2000.0

    summary = event.summary_dict()
    assert "event_energy_saved_kwh (simulated)" in summary
    assert "net_energy_saved_kwh (simulated)" in summary
    assert "co2_reduction_kg (estimate)" in summary
    assert "simulated_reward_inr (simulated)" in summary
    assert summary["shortfall_power_kw"] == 80.0


def test_load_facilities_from_default_csv() -> None:
    """Verify loading of data/facilities.csv with 6 facilities."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "facilities.csv"
    facilities = load_facilities_from_csv(csv_path)

    assert len(facilities) == 6
    ids = [f.id for f in facilities]
    assert "FAC-FZN-01" in ids
    assert "FAC-ICE-02" in ids
    assert "FAC-VAX-03" in ids
    assert "FAC-PRD-04" in ids
    assert "FAC-SEA-05" in ids
    assert "FAC-MLK-06" in ids

    # Check products diversity
    products = {f.product_type for f in facilities}
    assert len(products) == 6

    # Verify that all loaded facilities satisfy physical rules
    for fac in facilities:
        assert fac.min_temp_c < fac.max_safe_temp_c
        assert fac.rated_power_kw > 0
        assert fac.thermal_mass_kwh_per_c > 0
        assert fac.insulation_r_deg_c_per_kw > 0
        assert fac.cop_nominal > 0


def test_load_facilities_missing_file(tmp_path: Path) -> None:
    """Verify FileNotFoundError is raised when file does not exist."""
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_facilities_from_csv(missing)


def test_load_facilities_malformed_csv(tmp_path: Path) -> None:
    """Verify ValueError is raised if columns are missing or values corrupted."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("facility_id,product_type\nFAC-01,Ice Cream\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_facilities_from_csv(bad_csv)


def test_format_facilities_table() -> None:
    """Verify ASCII table formatting includes headers and audit disclaimers."""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "facilities.csv"
    facilities = load_facilities_from_csv(csv_path)
    output = format_facilities_table(facilities)
    assert "FAC-FZN-01" in output
    assert "FAC-VAX-03" in output
    assert "simulated" in output.lower()
    assert "estimate" in output.lower()
