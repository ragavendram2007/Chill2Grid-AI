"""reality_gap.py - Physics-to-Reality Gap Modeling for Chill2Grid AI.

Models the epistemic gap between the nominal digital twin (calibrated beliefs)
and the true physical world:
  1. Structural parameter mismatch: Real insulation resistance R and thermal capacitance C
     deviate by ±10% from the operator's nominal model (aging, frost buildup, sensor drift).
  2. Unmodelled internal thermal disturbances: Forklifts, loading dock activity, human occupancy,
     and lighting create an AR(1) process plus intermittent heat steps with std dev ~3% of mean wall heat.

NOTE ON INVENTED ASSUMPTIONS:
These perturbation distributions represent synthetic domain assumptions for hackathon evaluation
of hybrid physics+ML residual learning. They are fixed constants and must not be tuned to artificially
flatter ML benchmark metrics.
"""

from __future__ import annotations

from typing import List
import numpy as np

from core.contracts import Facility


def make_world_facility(facility: Facility, rng: np.random.Generator) -> Facility:
    """Create a perturbed 'true world' copy of a facility representing real-world physical variation.

    Thermal resistance R and thermal capacitance C are each perturbed independently
    by a uniform random multiplier in [0.90, 1.10]. The nominal digital twin only knows
    the unperturbed catalog values.

    Args:
        facility: Calibrated nominal Facility instance.
        rng: Seeded numpy random generator.

    Returns:
        New Facility instance with perturbed physical properties.
    """
    r_factor = float(rng.uniform(0.90, 1.10))
    c_factor = float(rng.uniform(0.90, 1.10))

    perturbed_r = round(facility.insulation_r_deg_c_per_kw * r_factor, 4)
    perturbed_c = round(facility.thermal_mass_kwh_per_c * c_factor, 2)

    return Facility(
        id=facility.id,
        name=facility.name,
        product_type=facility.product_type,
        thermal_mass_kwh_per_c=perturbed_c,
        insulation_r_deg_c_per_kw=perturbed_r,
        rated_power_kw=facility.rated_power_kw,
        min_temp_c=facility.min_temp_c,
        max_safe_temp_c=facility.max_safe_temp_c,
        min_on_min=facility.min_on_min,
        min_off_min=facility.min_off_min,
        cop_nominal=facility.cop_nominal,
    )


def make_disturbance(
    rng: np.random.Generator,
    n_minutes: int,
    mean_wall_heat_kw: float,
) -> List[float]:
    """Generate unmodelled heat disturbance time-series (kW) on the air node.

    Combines an AR(1) continuous stochastic background process with intermittent
    short heat pulses (representing loading dock forklift entries and worker activity).
    Target standard deviation is approximately 3% of mean wall heat gain.

    Args:
        rng: Seeded numpy random generator.
        n_minutes: Duration of simulation in minutes.
        mean_wall_heat_kw: Expected nominal heat ingress through warehouse envelope (kW).

    Returns:
        List of per-minute disturbance values in kW (mean ~ 0).
    """
    if n_minutes <= 0:
        return []

    target_std_kw = max(0.5, 0.03 * abs(mean_wall_heat_kw))
    phi = 0.85  # AR(1) persistence coefficient
    innov_std = target_std_kw * np.sqrt(1.0 - phi**2)

    # 1. Generate AR(1) process
    disturbance = np.zeros(n_minutes, dtype=float)
    current_val = float(rng.normal(0.0, target_std_kw))
    for m in range(n_minutes):
        current_val = phi * current_val + float(rng.normal(0.0, innov_std))
        disturbance[m] = current_val

    # 2. Add intermittent short heat steps (duration 5-15 mins, probability ~0.01 per minute)
    m = 0
    while m < n_minutes:
        if rng.uniform(0.0, 1.0) < 0.012:
            step_len = int(rng.integers(5, 16))
            step_mag = float(rng.uniform(1.2, 2.2) * target_std_kw)
            end_step = min(n_minutes, m + step_len)
            disturbance[m:end_step] += step_mag
            m = end_step
        else:
            m += 1

    # Standardize scale to ensure std is close to 3% target
    actual_std = float(np.std(disturbance))
    if actual_std > 1e-4:
        disturbance = disturbance * (target_std_kw / actual_std)

    return [round(float(v), 4) for v in disturbance]
