"""demo_part3.py - CLI Demonstration for Part 3 Hybrid AI Forecaster.

Evaluates each of the 6 calibrated facilities under normal operational conditions:
- Current sensor temperature
- Physics 30-minute drift forecast (M0)
- Hybrid AI 30-minute forecast (M2)
- 90% Conservative Upper Bound (M3 Conformal)
- Predicted Safe Curtailment Minutes (90% confidence)
- Ground Truth Safe Minutes from True Physics Digital Twin
"""

from pathlib import Path
import numpy as np

from core.contracts import CurtailmentAction, FacilityState, load_facilities_from_csv
from models.temperature_predictor import estimate_safe_minutes, load_predictor, predict_trajectory
from simulation.reality_gap import make_world_facility
from simulation.thermal_model import simulate_temperature


def run_part3_demo() -> None:
    """Execute live CLI forecast inspection across the entire fleet."""
    project_root = Path(__file__).resolve().parent.parent
    cal_csv = project_root / "data" / "facilities_calibrated.csv"
    if not cal_csv.is_file():
        cal_csv = project_root / "data" / "facilities.csv"

    facilities = load_facilities_from_csv(cal_csv)
    load_predictor()  # Warm cache

    headers = [
        "Facility ID",
        "Product",
        "Current T",
        "Safe Max",
        "Physics 30m",
        "Hybrid 30m",
        "90% Upper",
        "Pred Safe",
        "True Safe",
        "Margin",
    ]
    rows = []

    rng = np.random.default_rng(42)

    for fac in facilities:
        setpoint = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
        # Normal steady-state temperature slightly below midpoint
        current_temp = round(float(setpoint - 0.2), 2)

        state = FacilityState(
            facility_id=fac.id,
            temp_c=current_temp,
            outside_temp_c=35.0,
            humidity_pct=60.0,
            product_load_fraction=0.85,
            door_openings_per_hour=2.0,
            compressor_on=True,
            sensor_ok=True,
        )

        # Generate realistic 60-minute sensor history
        sensor_history = [
            round(float(current_temp + rng.normal(0.0, 0.08)), 2)
            for _ in range(65)
        ]

        # Forecast trajectory
        forecast = predict_trajectory(
            facility=fac,
            state=state,
            sensor_history_c=sensor_history,
            horizons=[30],
        )

        phys_30 = forecast.physics_c[0]
        hyb_30 = forecast.mean_c[0]
        upper_90 = forecast.upper_c[0.90][0]

        pred_safe_min, _ = estimate_safe_minutes(
            facility=fac,
            state=state,
            sensor_history_c=sensor_history,
            confidence=0.90,
        )

        # Ground truth safe drift minutes evaluated by branching the perturbed true world
        world_fac = make_world_facility(fac, rng=np.random.default_rng(123))
        act = CurtailmentAction(world_fac.id, start_min=0, duration_min=120)
        true_df = simulate_temperature(
            facility=world_fac,
            state=state,
            duration_min=120,
            curtailments=[act],
            seed=42,
            initial_air_temp_c=current_temp,
            initial_product_temp_c=current_temp,
            initial_compressor_on=False,
            initial_minutes_in_state=0,
        )

        breached = true_df["air_temp_c"] >= fac.max_safe_temp_c
        true_safe_min = float(true_df.loc[breached, "minute"].iloc[0]) if breached.any() else 120.0
        margin = round(true_safe_min - pred_safe_min, 1)

        rows.append([
            fac.id,
            fac.product_type[:15],
            f"{current_temp:+.1f} C",
            f"{fac.max_safe_temp_c:+.1f} C",
            f"{phys_30:+.2f} C",
            f"{hyb_30:+.2f} C",
            f"{upper_90:+.2f} C",
            f"{pred_safe_min:.1f} m",
            f"{true_safe_min:.1f} m",
            f"{margin:+.1f} m",
        ])

    col_widths = [len(h) for h in headers]
    for r in rows:
        for idx, val in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(val))

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print("\n" + "=" * len(separator))
    print("CHILL2GRID AI - PART 3 HYBRID AI FORECASTS & CONSERVATIVE SAFE WINDOWS (SIMULATED)")
    print("=" * len(separator))
    print(separator)
    print(header_str)
    print(separator)
    for r in rows:
        line_str = "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(r))) + " |"
        print(line_str)
    print(separator)
    print("* All forecasts evaluate unpowered drift at reference 35 C outdoor temperature.")
    print("* 90% Upper bound uses split-conformal finite-sample correction.")
    print("* Margin = True Safe Minutes minus Predicted Safe Minutes (>0 means cargo protected, conservative).")
    print("* All telemetry, models, and drift metrics are labeled as SIMULATED.\n")


if __name__ == "__main__":
    run_part3_demo()
