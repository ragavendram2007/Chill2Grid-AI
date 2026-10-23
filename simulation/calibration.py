"""calibration.py - Chill2Grid AI Facility Physics Calibration Engine.

Calibrates cold storage facility physical parameters (thermal resistance R_wall and
thermal capacitance C_total) from human-interpretable operational targets:
  1. Steady-state target compressor duty cycle (e.g. 55% to 85% at 35°C peak ambient)
  2. Target time-to-limit (e.g. 30 to 120 minutes of safe unpowered drift before reaching max safe temp)

Saves the calibrated fleet to data/facilities_calibrated.csv without modifying the raw catalog.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import brentq

from core.contracts import Facility, FacilityState
from simulation.thermal_model import cop, steady_state_duty_cycle, time_to_limit_min


# ============================================================================
# FACILITY CALIBRATION TARGETS
# ============================================================================

# Target design envelope for each facility based on thermal class:
# - Ice cream & vaccines: tighter envelopes (35-45 mins)
# - Chilled dairy & seafood: moderate envelopes (40-60 mins)
# - Bulk frozen foods & fresh produce: larger thermal inertia (75-105 mins)
CALIBRATION_TARGETS: Dict[str, Dict[str, float]] = {
    "FAC-FZN-01": {"target_duty_cycle": 0.65, "target_minutes_to_limit": 75.0},
    "FAC-ICE-02": {"target_duty_cycle": 0.72, "target_minutes_to_limit": 45.0},
    "FAC-VAX-03": {"target_duty_cycle": 0.60, "target_minutes_to_limit": 35.0},
    "FAC-PRD-04": {"target_duty_cycle": 0.70, "target_minutes_to_limit": 105.0},
    "FAC-SEA-05": {"target_duty_cycle": 0.75, "target_minutes_to_limit": 60.0},
    "FAC-MLK-06": {"target_duty_cycle": 0.68, "target_minutes_to_limit": 40.0},
}


# ============================================================================
# CALIBRATION SOLVER
# ============================================================================

def calibrate_facility(
    facility: Facility,
    target_duty_cycle: float,
    target_minutes_to_limit: float,
    reference_outside_c: float = 35.0,
    reference_load: float = 0.85,
    reference_door_rate: float = 2.0,
    reference_humidity_pct: float = 60.0,
) -> Tuple[Facility, Dict[str, float]]:
    """Calibrate thermal resistance (R_wall) and thermal capacitance (C_total) for a facility.

    Args:
        facility: Original uncalibrated Facility.
        target_duty_cycle: Desired steady-state compressor run fraction (e.g. 0.65).
        target_minutes_to_limit: Desired minutes from setpoint to max_safe_temp_c with cooling off.
        reference_outside_c: Standard outdoor design temperature (default 35.0 °C).
        reference_load: Cargo loading fraction (default 0.85).
        reference_door_rate: Loading dock door openings per hour (default 2.0).
        reference_humidity_pct: Ambient humidity percentage (default 60.0%).

    Returns:
        Tuple of (calibrated Facility, calibration report dict).
    """
    if not (0.20 <= target_duty_cycle <= 0.95):
        raise ValueError(f"target_duty_cycle ({target_duty_cycle}) outside plausible bounds [0.20, 0.95].")
    if target_minutes_to_limit <= 0.0:
        raise ValueError("target_minutes_to_limit must be strictly positive.")

    setpoint_c = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0

    # -------------------------------------------------------------------------
    # STEP 1: Calibrate R_wall from steady-state energy balance
    # -------------------------------------------------------------------------
    cop_eff = cop(reference_outside_c)
    max_cooling_thermal_kw = facility.rated_power_kw * cop_eff
    q_cool_target_kw = target_duty_cycle * max_cooling_thermal_kw

    # Average door heat
    temp_diff = max(0.0, reference_outside_c - setpoint_c)
    humid_scale = 1.0 + 0.005 * (reference_humidity_pct - 50.0)
    energy_door_event_kj = 15.0 * temp_diff * humid_scale
    door_heat_avg_kw = (reference_door_rate / 3600.0) * energy_door_event_kj

    q_wall_kw = q_cool_target_kw - door_heat_avg_kw
    if q_wall_kw <= 0.0:
        raise ValueError(
            f"Target duty cycle {target_duty_cycle} is too small to overcome door heat."
        )

    r_wall_calibrated = (reference_outside_c - setpoint_c) / q_wall_kw

    # -------------------------------------------------------------------------
    # STEP 2: Calibrate total thermal mass C_total with scipy.optimize.brentq
    # -------------------------------------------------------------------------
    # Create evaluation state at setpoint with zero door randomness
    test_state = FacilityState(
        facility_id=facility.id,
        temp_c=setpoint_c,
        outside_temp_c=reference_outside_c,
        humidity_pct=reference_humidity_pct,
        product_load_fraction=reference_load,
        door_openings_per_hour=0.0,
        compressor_on=False,
        sensor_ok=True,
    )

    def drift_error_fn(c_candidate: float) -> float:
        cand_facility = Facility(
            id=facility.id,
            name=facility.name,
            product_type=facility.product_type,
            thermal_mass_kwh_per_c=c_candidate,
            insulation_r_deg_c_per_kw=r_wall_calibrated,
            rated_power_kw=facility.rated_power_kw,
            min_temp_c=facility.min_temp_c,
            max_safe_temp_c=facility.max_safe_temp_c,
            min_on_min=facility.min_on_min,
            min_off_min=facility.min_off_min,
            cop_nominal=facility.cop_nominal,
        )
        achieved_min = time_to_limit_min(
            facility=cand_facility,
            state=test_state,
            max_minutes=int(target_minutes_to_limit * 3.0 + 60),
        )
        return float(achieved_min - target_minutes_to_limit)

    # Solve root: bracket bounds [1.0 kWh/°C, 3000.0 kWh/°C]
    c_calibrated, root_res = brentq(drift_error_fn, 1.0, 3000.0, full_output=True, xtol=1e-2)

    # Build final calibrated facility
    calibrated_fac = Facility(
        id=facility.id,
        name=facility.name,
        product_type=facility.product_type,
        thermal_mass_kwh_per_c=round(float(c_calibrated), 2),
        insulation_r_deg_c_per_kw=round(float(r_wall_calibrated), 4),
        rated_power_kw=facility.rated_power_kw,
        min_temp_c=facility.min_temp_c,
        max_safe_temp_c=facility.max_safe_temp_c,
        min_on_min=facility.min_on_min,
        min_off_min=facility.min_off_min,
        cop_nominal=facility.cop_nominal,
    )

    # Compute achieved metrics
    ref_state_duty = FacilityState(
        facility_id=facility.id,
        temp_c=setpoint_c,
        outside_temp_c=reference_outside_c,
        humidity_pct=reference_humidity_pct,
        product_load_fraction=reference_load,
        door_openings_per_hour=reference_door_rate,
        compressor_on=True,
        sensor_ok=True,
    )
    achieved_duty = steady_state_duty_cycle(calibrated_fac, ref_state_duty)
    achieved_time = time_to_limit_min(calibrated_fac, test_state, max_minutes=360)

    report = {
        "facility_id": facility.id,
        "product_type": facility.product_type,
        "target_duty_cycle": round(target_duty_cycle, 3),
        "achieved_duty_cycle": round(achieved_duty, 3),
        "duty_cycle_error_pct": round(abs(achieved_duty - target_duty_cycle) / target_duty_cycle * 100.0, 2),
        "target_minutes_to_limit": round(target_minutes_to_limit, 1),
        "achieved_minutes_to_limit": round(achieved_time, 1),
        "time_error_pct": round(abs(achieved_time - target_minutes_to_limit) / target_minutes_to_limit * 100.0, 2),
        "calibrated_r_deg_c_per_kw": round(float(r_wall_calibrated), 4),
        "calibrated_thermal_mass_kwh_per_c": round(float(c_calibrated), 2),
    }

    return calibrated_fac, report


# ============================================================================
# FLEET CALIBRATION PIPELINE
# ============================================================================

def calibrate_all_facilities(
    facilities: List[Facility],
    targets: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[List[Facility], List[Dict[str, float]]]:
    """Run calibration across the entire fleet and return calibrated objects and reports."""
    target_dict = targets or CALIBRATION_TARGETS
    calibrated_list: List[Facility] = []
    reports: List[Dict[str, float]] = []

    for fac in facilities:
        if fac.id not in target_dict:
            raise KeyError(f"Facility {fac.id} missing calibration target definition.")
        t_conf = target_dict[fac.id]
        cal_fac, rep = calibrate_facility(
            facility=fac,
            target_duty_cycle=t_conf["target_duty_cycle"],
            target_minutes_to_limit=t_conf["target_minutes_to_limit"],
        )
        calibrated_list.append(cal_fac)
        reports.append(rep)

    return calibrated_list, reports


def save_calibrated_facilities_csv(
    calibrated_facilities: List[Facility],
    output_path: str | Path,
) -> None:
    """Save calibrated facility objects to CSV without overwriting the original catalog."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "facility_id",
        "name",
        "product_type",
        "thermal_mass_kwh_per_c",
        "insulation_r_deg_c_per_kw",
        "rated_power_kw",
        "min_temp_c",
        "max_safe_temp_c",
        "min_on_min",
        "min_off_min",
        "cop_nominal",
    ]

    with open(out_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fac in calibrated_facilities:
            writer.writerow({
                "facility_id": fac.id,
                "name": fac.name,
                "product_type": fac.product_type,
                "thermal_mass_kwh_per_c": fac.thermal_mass_kwh_per_c,
                "insulation_r_deg_c_per_kw": fac.insulation_r_deg_c_per_kw,
                "rated_power_kw": fac.rated_power_kw,
                "min_temp_c": fac.min_temp_c,
                "max_safe_temp_c": fac.max_safe_temp_c,
                "min_on_min": fac.min_on_min,
                "min_off_min": fac.min_off_min,
                "cop_nominal": fac.cop_nominal,
            })
