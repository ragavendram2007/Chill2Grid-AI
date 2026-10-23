"""thermal_model.py - Chill2Grid AI Two-Node Physics Thermal Digital Twin.

Simulates cold-storage thermal dynamics with two coupled nodes:
1. Air node (sensor-measured, fast dynamics, time constant ~10-25 mins)
2. Product node (hidden cargo bulk, slow thermal inertia, time constant ~4-12 hours)

Equations:
  C_air  * d(T_air)/dt  = (T_out - T_air)/R_wall + (T_prod - T_air)/R_ap + Q_door(t) - Q_cool(t)
  C_prod * d(T_prod)/dt = (T_air - T_prod)/R_ap

Units:
  C in kJ / K (1 kWh = 3600 kJ)
  R in K / kW (°C / kW)
  Q in kW (kJ / s)
  t in seconds (integrated in 10-second sub-steps inside each minute)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from core.contracts import CurtailmentAction, Facility, FacilityState


# ============================================================================
# PHYSICAL CONSTANTS & MODEL TUNING PARAMETERS
# ============================================================================

# COP model constants: COP = max(1.0, cop_ref - slope * (T_out - 25))
# Efficiency falls as outdoor temperature rises because the compressor must pump
# heat across a steeper thermal lift / pressure difference.
COP_REF: float = 2.0
COP_SLOPE: float = 0.03  # 1/°C
COP_MIN: float = 1.0
COP_T_REF_C: float = 25.0

# Thermostat and control parameters
THERMOSTAT_DEADBAND_C: float = 0.5  # Compressor switch deadband ±0.5 °C around setpoint
AUXILIARY_POWER_FRACTION: float = 0.03  # Fans, controls, lights: 3% of rated kW, always ON

# Thermal node partitioning constants:
# In refrigerated logistics, air has low density (~1.2 kg/m³) and low volumetric heat capacity,
# whereas stored cargo (water, ice, produce, frozen meat) has huge heat capacity (~2-4 kJ/kg/K).
# We allocate 8% of nominal thermal mass to air, and 92% to product (scaled by loading fraction).
# Coupling resistance R_ap is set to 0.02 * R_wall, reflecting low convective boundary resistance
# under forced evaporator airflow. This gives an air node time constant of ~5-15 mins and product
# time constant of ~1-4 hours, with loading fraction strongly influencing safe unpowered drift.
AIR_THERMAL_MASS_FRACTION: float = 0.08
AIR_PRODUCT_COUPLING_RATIO: float = 0.02

# Simulation integration sub-stepping
INTEGRATION_SUB_STEP_SEC: float = 10.0  # 10s sub-steps ensure numerical stability
SECONDS_PER_MINUTE: float = 60.0
KWH_TO_KJ: float = 3600.0


# ============================================================================
# WEATHER-AWARE COMPRESSOR EFFICIENCY
# ============================================================================

def cop(outside_temp_c: float) -> float:
    """Calculate compressor Coefficient of Performance (COP) as a function of outdoor weather.

    Formula:
        COP(T_out) = max(1.0, 2.0 - 0.03 * (T_out - 25.0))
    Units:
        outside_temp_c in °C.
        COP is dimensionless (thermal cooling kW / electrical kW).
    """
    cop_val = COP_REF - COP_SLOPE * (outside_temp_c - COP_T_REF_C)
    return float(max(COP_MIN, cop_val))


# ============================================================================
# SENSOR TELEMETRY EMULATION
# ============================================================================

def read_sensor(
    row_or_value: Union[float, Dict[str, float], pd.Series],
    rng: np.random.Generator,
    noise_std_c: float = 0.1,
    failure_mode: Optional[str] = None,
    stuck_value: Optional[float] = None,
    drift_offset_c: float = 0.0,
) -> Optional[float]:
    """Emulate physical sensor reading with realistic Gaussian noise and fault modes.

    Args:
        row_or_value: Instantaneous air temperature float or row containing 'air_temp_c'.
        rng: Seeded numpy random Generator.
        noise_std_c: Gaussian noise standard deviation in °C (default 0.1 °C).
        failure_mode: None (normal), 'dropout' (None), 'stuck' (stuck value), or 'drift'.
        stuck_value: Last valid reading to freeze on when failure_mode == 'stuck'.
        drift_offset_c: Offset to add when failure_mode == 'drift'.

    Returns:
        Noisy temperature float, or None in case of sensor dropout.
    """
    if isinstance(row_or_value, (dict, pd.Series)):
        temp_val = float(row_or_value.get("air_temp_c", row_or_value.get("temp_c", 0.0)))
    else:
        temp_val = float(row_or_value)

    if failure_mode == "dropout":
        return None
    elif failure_mode == "stuck":
        return float(stuck_value if stuck_value is not None else temp_val)
    elif failure_mode == "drift":
        noise = rng.normal(0.0, noise_std_c)
        return float(temp_val + drift_offset_c + noise)
    else:
        noise = rng.normal(0.0, noise_std_c)
        return float(temp_val + noise)


# ============================================================================
# STEADY-STATE ESTIMATION & PHYSICS BASELINE
# ============================================================================

def steady_state_duty_cycle(facility: Facility, state: FacilityState) -> float:
    """Calculate the theoretical steady-state compressor duty cycle at given ambient conditions.

    Includes average infiltration heat from loading doors and weather-adjusted COP.
    """
    setpoint_c = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0
    r_wall = facility.insulation_r_deg_c_per_kw
    heat_wall_kw = max(0.0, (state.outside_temp_c - setpoint_c) / r_wall)

    # Average door heat calculation
    temp_diff = max(0.0, state.outside_temp_c - setpoint_c)
    humid_scale = 1.0 + 0.005 * (state.humidity_pct - 50.0)
    energy_per_door_event_kj = 15.0 * temp_diff * humid_scale
    door_heat_avg_kw = (state.door_openings_per_hour / 3600.0) * energy_per_door_event_kj

    cop_eff = cop(state.outside_temp_c)
    max_cooling_thermal_kw = facility.rated_power_kw * cop_eff

    duty = (heat_wall_kw + door_heat_avg_kw) / max_cooling_thermal_kw
    return float(np.clip(duty, 0.0, 1.0))


def time_to_limit_min(facility: Facility, state: FacilityState, max_minutes: int = 360) -> float:
    """Deterministic calculation of minutes until air temperature hits max_safe_temp_c.

    Evaluated with compressor forced OFF and no random door openings.
    Serves as the deterministic physics baseline for ML models in Part 3.
    """
    # Deterministic run: single-action curtailment covering the full simulation window
    curtailment = CurtailmentAction(
        facility_id=facility.id,
        start_min=0,
        duration_min=max_minutes,
    )
    # Zero door openings for deterministic calculation
    det_state = FacilityState(
        facility_id=state.facility_id,
        temp_c=state.temp_c,
        outside_temp_c=state.outside_temp_c,
        humidity_pct=state.humidity_pct,
        product_load_fraction=state.product_load_fraction,
        door_openings_per_hour=0.0,
        compressor_on=False,
        sensor_ok=True,
    )

    df = simulate_temperature(
        facility=facility,
        state=det_state,
        duration_min=max_minutes,
        curtailments=[curtailment],
        seed=42,
    )

    limit_mask = df["air_temp_c"] >= facility.max_safe_temp_c
    if limit_mask.any():
        idx = int(df.loc[limit_mask, "minute"].iloc[0])
        if idx == 0:
            return 0.0
        # Linear interpolation between previous minute and crossing minute
        t_prev = df.loc[idx - 1, "air_temp_c"]
        t_curr = df.loc[idx, "air_temp_c"]
        if t_curr > t_prev:
            frac = (facility.max_safe_temp_c - t_prev) / (t_curr - t_prev)
            return float(round((idx - 1) + frac, 2))
        return float(idx)

    return float(max_minutes)


# ============================================================================
# PRIMARY SIMULATOR: TWO-NODE ODE SOLVER
# ============================================================================

def simulate_temperature(
    facility: Facility,
    state: FacilityState,
    duration_min: int,
    curtailments: Optional[Union[CurtailmentAction, List[CurtailmentAction]]] = None,
    seed: int = 0,
    outside_profile_c: Optional[List[float]] = None,
    restart_delay_min: int = 0,
    initial_air_temp_c: Optional[float] = None,
    initial_product_temp_c: Optional[float] = None,
    initial_compressor_on: Optional[bool] = None,
    initial_minutes_in_state: Optional[int] = None,
    extra_heat_kw: Optional[List[float]] = None,
    expected_door_heat: bool = False,
) -> pd.DataFrame:
    """Run discrete-time physical simulation of warehouse temperatures and power draw.

    Args:
        facility: Physical Facility configuration.
        state: Initial operational telemetry and conditions.
        duration_min: Total simulation window in minutes.
        curtailments: CurtailmentAction or list of CurtailmentAction commands.
        seed: Random seed for door openings and stochastic telemetry.
        outside_profile_c: Optional per-minute outside temperatures. If None, uses state.outside_temp_c.
        restart_delay_min: Stagger delay in minutes before compressor can restart after curtailment.
        initial_air_temp_c: Optional starting air temperature override. Default is state.temp_c.
        initial_product_temp_c: Optional starting product temperature override. Default is state.temp_c.
        initial_compressor_on: Optional starting compressor status override. Default is state.compressor_on.
        initial_minutes_in_state: Optional elapsed anti-cycling timer. Default 100 mins.
        extra_heat_kw: Optional per-minute list of unmodelled disturbance heat additions (kW).
        expected_door_heat: If True, uses analytical average door heat instead of stochastic events.

    Returns:
        DataFrame with per-minute telemetry and warning flags.
    """
    if duration_min <= 0:
        raise ValueError("duration_min must be strictly positive.")

    # Normalize curtailments list
    actions: List[CurtailmentAction] = []
    if curtailments is not None:
        if isinstance(curtailments, CurtailmentAction):
            actions = [curtailments]
        else:
            actions = list(curtailments)

    # Initialize thermal capacitances (kJ / K)
    total_mass_kj = facility.thermal_mass_kwh_per_c * KWH_TO_KJ
    c_air = total_mass_kj * AIR_THERMAL_MASS_FRACTION
    c_prod = total_mass_kj * (1.0 - AIR_THERMAL_MASS_FRACTION) * state.product_load_fraction
    c_prod = max(c_prod, 100.0)  # Numerical safety floor

    # Thermal resistances (K / kW)
    r_wall = facility.insulation_r_deg_c_per_kw
    r_ap = r_wall * AIR_PRODUCT_COUPLING_RATIO

    # Thermostat setpoints
    setpoint_c = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0
    deadband = THERMOSTAT_DEADBAND_C

    # Initial states (with optional overrides)
    air_temp_c = float(state.temp_c if initial_air_temp_c is None else initial_air_temp_c)
    product_temp_c = float(state.temp_c if initial_product_temp_c is None else initial_product_temp_c)
    compressor_on = bool(state.compressor_on if initial_compressor_on is None else initial_compressor_on)
    time_in_state_min = 100 if initial_minutes_in_state is None else int(initial_minutes_in_state)

    # Stagger tracking
    last_curtailment_end_min = -999

    # Door opening events: either deterministic expected heat or seeded random draws
    door_events_flag = [False] * duration_min
    door_heat_kw = [0.0] * duration_min

    if expected_door_heat:
        # Analytical expected average door heat
        for m in range(duration_min):
            t_out_m = outside_profile_c[m] if outside_profile_c is not None else state.outside_temp_c
            delta_t = max(0.0, t_out_m - air_temp_c)
            humid_factor = 1.0 + 0.005 * (state.humidity_pct - 50.0)
            energy_per_event = 15.0 * delta_t * humid_factor
            door_heat_kw[m] = (state.door_openings_per_hour / 3600.0) * energy_per_event
    else:
        rng = np.random.default_rng(seed)
        p_door = state.door_openings_per_hour / 60.0
        for m in range(duration_min):
            t_out = outside_profile_c[m] if outside_profile_c is not None else state.outside_temp_c
            if state.door_openings_per_hour > 0.0 and rng.uniform(0.0, 1.0) < p_door:
                door_events_flag[m] = True
                delta_t = max(0.0, t_out - air_temp_c)
                humid_factor = 1.0 + 0.005 * (state.humidity_pct - 50.0)
                # Energy in kJ per door opening event spread across 2 minutes
                energy_kj = 15.0 * delta_t * humid_factor
                power_spread_kw = energy_kj / 120.0
                door_heat_kw[m] += power_spread_kw
                if m + 1 < duration_min:
                    door_heat_kw[m + 1] += power_spread_kw

    sub_steps = int(SECONDS_PER_MINUTE / INTEGRATION_SUB_STEP_SEC)
    records = []

    for m in range(duration_min):
        t_out = float(outside_profile_c[m] if outside_profile_c is not None else state.outside_temp_c)

        # Check for active curtailment command
        is_curtailed = any(
            act.start_min <= m < (act.start_min + act.duration_min)
            for act in actions
        )

        warn_min_on_violated = False
        warn_min_off_violated = False

        if is_curtailed:
            forced_off = True
            # Rule: OBEY forced-off even if min ON time is violated, and set warning flag
            if compressor_on:
                if time_in_state_min < facility.min_on_min:
                    warn_min_on_violated = True
                compressor_on = False
                time_in_state_min = 0
            # Track the latest minute curtailment was active
            last_curtailment_end_min = m + 1
        else:
            forced_off = False
            # Check staggered recovery delay
            in_restart_delay = (m < last_curtailment_end_min + restart_delay_min)

            # Thermostat logic
            if compressor_on:
                if air_temp_c <= setpoint_c - deadband and time_in_state_min >= facility.min_on_min:
                    compressor_on = False
                    time_in_state_min = 0
            else:
                if (
                    not in_restart_delay
                    and air_temp_c >= setpoint_c + deadband
                    and time_in_state_min >= facility.min_off_min
                ):
                    compressor_on = True
                    time_in_state_min = 0

        # High temperature threshold warning flag
        warn_limit_exceeded = air_temp_c > facility.max_safe_temp_c

        # Power accounting
        cop_current = cop(t_out)
        compressor_power_kw = facility.rated_power_kw if compressor_on else 0.0
        auxiliary_power_kw = facility.rated_power_kw * AUXILIARY_POWER_FRACTION
        total_power_kw = compressor_power_kw + auxiliary_power_kw
        energy_kwh = total_power_kw * (1.0 / 60.0)
        cooling_kw_thermal = compressor_power_kw * cop_current

        # Record minute state at start of integration interval
        records.append({
            "minute": m,
            "outside_temp_c": round(t_out, 3),
            "air_temp_c": round(air_temp_c, 3),
            "product_temp_c": round(product_temp_c, 3),
            "compressor_on": compressor_on,
            "forced_off": forced_off,
            "door_event": door_events_flag[m],
            "cooling_kw_thermal": round(cooling_kw_thermal, 3),
            "power_kw": round(total_power_kw, 3),
            "energy_kwh": round(energy_kwh, 5),
            "warn_limit_exceeded": warn_limit_exceeded,
            "warn_min_on_violated": warn_min_on_violated,
            "warn_min_off_violated": warn_min_off_violated,
        })

        # Integrate differential equations across sub-steps
        q_door_sub = door_heat_kw[m]
        q_extra_sub = float(extra_heat_kw[m]) if extra_heat_kw is not None else 0.0
        q_cool_sub = cooling_kw_thermal

        # Sub-step bounded by explicit Euler numerical stability limit
        max_stable_dt = max(0.2, 0.4 * c_air * r_ap)
        sub_dt = min(INTEGRATION_SUB_STEP_SEC, max_stable_dt)
        sub_steps = int(np.ceil(SECONDS_PER_MINUTE / sub_dt))
        dt = SECONDS_PER_MINUTE / sub_steps

        for _ in range(sub_steps):
            q_wall = (t_out - air_temp_c) / r_wall
            q_ap = (product_temp_c - air_temp_c) / r_ap

            d_air = (dt / c_air) * (q_wall + q_ap + q_door_sub + q_extra_sub - q_cool_sub)
            d_prod = (dt / c_prod) * (-q_ap)

            air_temp_c += d_air
            product_temp_c += d_prod

        time_in_state_min += 1

    return pd.DataFrame(records)


# ============================================================================
# CLOSED-LOOP EVENT ACCOUNTING: BASELINE VS EVENT PAIR
# ============================================================================

def simulate_pair(
    facility: Facility,
    state: FacilityState,
    duration_min: Optional[int] = None,
    curtailments: Optional[Union[CurtailmentAction, List[CurtailmentAction]]] = None,
    seed: int = 0,
    outside_profile_c: Optional[List[float]] = None,
    restart_delay_min: int = 0,
    phase_robust: bool = True,
    n_phase_offsets: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Execute identical twin simulation comparing undisturbed baseline against curtailment event.

    Both runs use the exact same random seed so weather and door opening events match perfectly.
    All energy and power metrics are calculated strictly by direct difference between the twin runs.
    Horizon runs until |stored-heat difference| < 1% of heat injected during event (capped at 12 hours).
    When phase_robust=True, runs across 8 start offsets spread across one baseline compressor cycle.
    """
    # Determine event time boundaries
    if curtailments is None:
        actions: List[CurtailmentAction] = []
    elif isinstance(curtailments, CurtailmentAction):
        actions = [curtailments]
    else:
        actions = list(curtailments)

    # 12-hour maximum horizon cap (720 minutes)
    horizon = max(duration_min or 0, 720)

    # Simulate undisturbed baseline run over full horizon
    baseline_df = simulate_temperature(
        facility=facility,
        state=state,
        duration_min=horizon,
        curtailments=None,
        seed=seed,
        outside_profile_c=outside_profile_c,
    )

    if not actions:
        # Zero curtailment: twin runs are identical
        event_df = baseline_df.copy()
        summary = {
            "energy_saved_during_event_kwh": 0.0,
            "recovery_extra_kwh": 0.0,
            "net_kwh_saved": 0.0,
            "net_kwh_saved_std": 0.0,
            "net_kwh_saved_heat_corrected": 0.0,
            "accounting_uncertainty_kwh": 0.0,
            "peak_power_baseline_kw": float(baseline_df["power_kw"].max()),
            "peak_power_event_kw": float(baseline_df["power_kw"].max()),
            "peak_recovery_kw": 0.0,
            "peak_rebound_kw": 0.0,
            "shifted_fraction": 0.0,
            "settle_time_min": 0.0,
            "max_air_temp_event_c": float(baseline_df["air_temp_c"].max()),
            "minutes_over_limit": 0,
        }
        return baseline_df, event_df, summary

    action_dur = max(a.duration_min for a in actions)
    orig_start = min(a.start_min for a in actions)

    # Identify baseline cycle length and start offsets
    cut_ins = []
    for m in range(60, min(300, horizon - 1)):
        if not baseline_df.loc[m - 1, "compressor_on"] and baseline_df.loc[m, "compressor_on"]:
            cut_ins.append(m)

    if len(cut_ins) >= 2:
        cycle_len = cut_ins[1] - cut_ins[0]
        base_start = cut_ins[0]
    else:
        cycle_len = max(20, facility.min_on_min + facility.min_off_min)
        base_start = orig_start

    if phase_robust and n_phase_offsets > 1:
        offsets = [int(round(k * cycle_len / float(n_phase_offsets))) for k in range(n_phase_offsets)]
    else:
        offsets = [0]
        base_start = orig_start

    saved_list: List[float] = []
    net_prim_list: List[float] = []
    net_cross_list: List[float] = []
    peak_rebound_list: List[float] = []
    settle_list: List[float] = []
    max_temp_list: List[float] = []
    over_limit_list: List[int] = []

    c_tot = facility.thermal_mass_kwh_per_c
    c_air = c_tot * AIR_THERMAL_MASS_FRACTION
    c_prod = c_tot * (1.0 - AIR_THERMAL_MASS_FRACTION) * state.product_load_fraction
    t_end_idx = horizon - 1

    first_event_df: Optional[pd.DataFrame] = None

    for off in offsets:
        ev_start = base_start + off
        ev_end = ev_start + action_dur
        action_k = CurtailmentAction(facility.id, start_min=ev_start, duration_min=action_dur)

        e_df = simulate_temperature(
            facility=facility,
            state=state,
            duration_min=horizon,
            curtailments=[action_k],
            seed=seed,
            outside_profile_c=outside_profile_c,
            restart_delay_min=restart_delay_min,
        )
        if first_event_df is None:
            first_event_df = e_df

        dE = (baseline_df["energy_kwh"] - e_df["energy_kwh"]).cumsum()

        # Stored heat difference in kWh thermal (event minus baseline)
        dH_air = c_air * (e_df["air_temp_c"] - baseline_df["air_temp_c"])
        dH_prod = c_prod * (e_df["product_temp_c"] - baseline_df["product_temp_c"])
        dH_tot = dH_air + dH_prod

        # Heat injected during event
        dH_inj = abs(float(dH_tot.iloc[ev_end - 1]))
        thresh = 0.02 * dH_inj

        # Find settle time: moving average of |dH| over one baseline cycle < 2% of injected heat
        ma_dH = dH_tot.abs().rolling(window=cycle_len, min_periods=1).mean()
        settle_m = horizon
        for m in range(ev_end, horizon):
            if ma_dH.iloc[m] < thresh:
                settle_m = m
                break
        settle_list.append(float(settle_m))

        # 1. Saved during event
        saved_ev = float(dE.iloc[ev_end - 1] - (dE.iloc[ev_start - 1] if ev_start > 0 else 0.0))
        saved_list.append(saved_ev)

        # 2. Net primary: average of dE over the last 2 hours (120 min) of settled horizon
        net_prim = float(dE.iloc[-120:].mean())
        net_prim_list.append(net_prim)

        # 3. Net cross-check: dE(t_end) - (dH_event - dH_base) / COP
        cop_val = cop(float(e_df["outside_temp_c"].iloc[t_end_idx]))
        net_cross = float(dE.iloc[t_end_idx] - (dH_tot.iloc[t_end_idx] / cop_val))
        net_cross_list.append(net_cross)

        # 4. Peak rebound kW: maximum 15-minute average of (event power - baseline power) after event ends
        diff_p = e_df["power_kw"] - baseline_df["power_kw"]
        rebound_series = diff_p.loc[e_df["minute"] >= ev_end]
        rolling_15 = rebound_series.rolling(window=15, min_periods=1).mean()
        peak_reb = float(max(0.0, rolling_15.max())) if len(rolling_15) > 0 else 0.0
        peak_rebound_list.append(peak_reb)

        # Temperatures
        max_temp_list.append(float(e_df["air_temp_c"].max()))
        over_limit_list.append(int((e_df["air_temp_c"] > facility.max_safe_temp_c).sum()))

    saved_during_event_kwh = float(np.mean(saved_list))
    net_kwh_saved = float(np.mean(net_prim_list))
    net_kwh_saved_std = float(np.std(net_prim_list))
    net_kwh_saved_heat_corrected = float(np.mean(net_cross_list))
    accounting_uncertainty_kwh = float(abs(net_kwh_saved - net_kwh_saved_heat_corrected))
    recovery_extra_kwh = float(saved_during_event_kwh - net_kwh_saved)
    shifted_fraction = (
        float(recovery_extra_kwh / saved_during_event_kwh)
        if saved_during_event_kwh > 0.0
        else 0.0
    )
    peak_rebound_kw = float(np.mean(peak_rebound_list))
    settle_time_min = float(np.mean(settle_list))
    max_air_temp_event_c = float(np.max(max_temp_list))
    minutes_over_limit = int(np.max(over_limit_list))

    event_df = first_event_df if first_event_df is not None else baseline_df

    summary = {
        "energy_saved_during_event_kwh": round(saved_during_event_kwh, 4),
        "recovery_extra_kwh": round(recovery_extra_kwh, 4),
        "net_kwh_saved": round(net_kwh_saved, 4),
        "net_kwh_saved_std": round(net_kwh_saved_std, 4),
        "net_kwh_saved_heat_corrected": round(net_kwh_saved_heat_corrected, 4),
        "accounting_uncertainty_kwh": round(accounting_uncertainty_kwh, 4),
        "peak_power_baseline_kw": round(float(baseline_df["power_kw"].max()), 2),
        "peak_power_event_kw": round(float(event_df["power_kw"].max()), 2),
        "peak_recovery_kw": round(peak_rebound_kw, 2),  # Kept as alias
        "peak_rebound_kw": round(peak_rebound_kw, 2),
        "shifted_fraction": round(shifted_fraction, 4),
        "settle_time_min": round(settle_time_min, 1),
        "max_air_temp_event_c": round(max_air_temp_event_c, 2),
        "minutes_over_limit": minutes_over_limit,
    }

    return baseline_df, event_df, summary
