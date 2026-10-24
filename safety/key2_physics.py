"""key2_physics.py - Key 2: Recalibrated Physics Twin with Pessimism Scalar Lambda.

Replaces stacked design margins with a unified pessimism scalar lambda in [0, 1]:
  - lambda = 0: Exact nominal physics twin (zero design margin)
  - lambda = 1: Full design margins (0.9x R, 0.9x C, 2x door heat, 2-sigma disturbance, noise bias)
  - OOD fallback: Enforces lambda = 1.0 with +20% extra guard margin.

Also implements dynamic twin-computed restart overshoot estimation.
All telemetry and outputs are labeled simulated.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.contracts import CurtailmentAction, Facility, FacilityState
from safety.rules import RiskTierConfig
from safety.types import FleetSnapshot
from simulation.thermal_model import (
    THERMOSTAT_DEADBAND_C,
    cop,
    simulate_temperature,
)


def compute_restart_overshoot_min(
    facility: Facility,
    outside_temp_c: float,
    tier: RiskTierConfig,
    lambda_pessimism: float = 1.0,
    restart_lag_min: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute physical restart overshoot rise (°C) and equivalent reserve minutes.

    Simulates the planned end of an event:
    1. Compressor is commanded ON with equipment restart lag.
    2. During restart lag, cooling is delayed; air temperature continues heating.
    3. Compressor engages and temperature peaks before turning downward.
    4. Overshoot = T_peak - T_start.
    5. Reserve minutes = overshoot / (heating rate in °C/min).

    Returns:
        (reserve_minutes, overshoot_rise_deg_c)
    """
    lag_m = float(
        restart_lag_min
        if restart_lag_min is not None
        else getattr(tier, "restart_lag_min", tier.restart_reserve_min)
    )
    if lag_m <= 0.0:
        return (0.0, 0.0)

    # Setpoint and warm-phase start temperature
    setpoint = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0
    limit_thresh = facility.max_safe_temp_c - tier.guard_margin_c
    start_temp = min(limit_thresh - 0.2, setpoint + THERMOSTAT_DEADBAND_C)

    # Parameter interpolation via lambda
    lam = float(np.clip(lambda_pessimism, 0.0, 1.0))
    r_wall = facility.insulation_r_deg_c_per_kw * (1.0 - 0.10 * lam)
    c_air = (facility.thermal_mass_kwh_per_c * 3600.0 * 0.10) * (1.0 - 0.10 * lam)

    # Approximate heating rate at start_temp during curtailment
    q_wall = max(0.0, (outside_temp_c - start_temp) / max(0.01, r_wall))
    q_door = (2.0 / 3600.0) * (15.0 * max(0.0, outside_temp_c - setpoint)) * (1.0 + lam)
    heat_rate_c_per_min = ((q_wall + q_door) / max(1.0, c_air)) * 60.0  # °C / min
    heat_rate_c_per_min = max(0.02, heat_rate_c_per_min)

    # Effective cooling rate when compressor starts
    cop_val = cop(outside_temp_c)
    cooling_kw = facility.rated_power_kw * cop_val
    cool_rate_c_per_min = ((cooling_kw - q_wall - q_door) / max(1.0, c_air)) * 60.0

    # Turnaround time after compressor starts: typically 1-3 minutes
    turnaround_m = 1.5 if cool_rate_c_per_min > 0 else 5.0
    overshoot_deg_c = heat_rate_c_per_min * (lag_m + 0.5 * turnaround_m)

    # Convert overshoot to equivalent reserve minutes
    reserve_min = float(round(overshoot_deg_c / heat_rate_c_per_min, 1))
    # Cap reserve by the tier restart reserve limit
    reserve_min = min(reserve_min, lag_m + 2.0)

    return (reserve_min, round(overshoot_deg_c, 3))


def simulate_key2_worst_case(
    facility: Facility,
    snapshot: FleetSnapshot,
    tier: RiskTierConfig,
    lambda_pessimism: float = 1.0,
    is_ood: bool = False,
    noise_sigma_c: float = 0.15,
) -> float:
    """Simulate worst-case deterministic physical trajectory to find Key 2 safe window.

    Unified Pessimism Scalar Lambda in [0, 1]:
      - lambda = 0: Exact nominal physics twin (zero design margin)
      - lambda = 1: Full design margins (0.9x R, 0.9x C, 2x door heat, 2-sigma disturbance)
      - is_ood = True: Forces lambda = 1.0 with an extra 20% guard margin.

    Returns:
        safe_duration_minutes (0.0 to 120.0 minutes)
    """
    # Enforce lambda = 1.0 if OOD or if lambda out of range
    lam = 1.0 if is_ood else float(np.clip(lambda_pessimism, 0.0, 1.0))

    # 1. Parameter Interpolation on Facility
    c_eff = facility.thermal_mass_kwh_per_c * (1.0 - 0.10 * lam)
    r_eff = facility.insulation_r_deg_c_per_kw * (1.0 - 0.10 * lam)

    key2_facility = Facility(
        id=facility.id,
        name=facility.name,
        product_type=facility.product_type,
        rated_power_kw=facility.rated_power_kw,
        min_temp_c=facility.min_temp_c,
        max_safe_temp_c=facility.max_safe_temp_c,
        thermal_mass_kwh_per_c=round(c_eff, 4),
        insulation_r_deg_c_per_kw=round(r_eff, 4),
        cop_nominal=facility.cop_nominal,
        min_on_min=facility.min_on_min,
        min_off_min=facility.min_off_min,
    )

    # 2. Start Air Temperature Interpolation
    setpoint = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0
    nom_start = snapshot.temp_c
    pess_start = max(snapshot.temp_c + 2.0 * noise_sigma_c, setpoint + THERMOSTAT_DEADBAND_C)
    start_air = (1.0 - lam) * nom_start + lam * pess_start

    # 3. Door Heat Multiplier Interpolation
    nom_door = snapshot.door_openings_per_hour
    pess_door = max(2.0, snapshot.door_openings_per_hour * 2.0)
    door_rate_eff = (1.0 - lam) * nom_door + lam * pess_door

    key2_state = FacilityState(
        facility_id=facility.id,
        temp_c=start_air,
        outside_temp_c=snapshot.outside_temp_c,
        humidity_pct=snapshot.humidity_pct,
        product_load_fraction=snapshot.product_load_fraction,
        door_openings_per_hour=door_rate_eff,
        compressor_on=False,
        sensor_ok=True,
        timestamp_min=0.0,
    )

    # 4. 2-Sigma Disturbance Heat Interpolation
    dist_sigma_kw = 0.05 * facility.rated_power_kw
    extra_heat_val = lam * 2.0 * dist_sigma_kw
    extra_heat_profile = [float(extra_heat_val)] * 120 if extra_heat_val > 1e-4 else None

    # 5. Forced Curtailment Simulation (Compressor forced OFF for 120 min)
    act = CurtailmentAction(facility.id, start_min=0, duration_min=120)
    sim_df = simulate_temperature(
        key2_facility,
        key2_state,
        duration_min=120,
        curtailments=[act],
        extra_heat_kw=extra_heat_profile,
        expected_door_heat=True,
        seed=42,
    )

    # 6. Guard Margin (with +20% extra margin if Out-Of-Distribution)
    effective_guard_margin = tier.guard_margin_c * (1.20 if is_ood else 1.0)
    limit_threshold = facility.max_safe_temp_c - effective_guard_margin

    # 7. Find First Breach Minute
    if sim_df["air_temp_c"].iloc[0] >= limit_threshold:
        return 0.0

    breach_mask = sim_df["air_temp_c"] >= limit_threshold
    if breach_mask.any():
        return float(sim_df.loc[breach_mask, "minute"].iloc[0])

    return 120.0
