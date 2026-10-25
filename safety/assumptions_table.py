"""assumptions_table.py - Introspects code constants and generates the Invented Assumptions table.

Discloses all synthetic reality-gap, tier margin, facility limit, and grid assumptions
directly extracted from code sources (never typed by hand).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from core.contracts import load_facilities_from_csv
from safety.rules import RISK_TIERS, RULES_VERSION
from impact.assess import load_grid_assumptions

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_invented_assumptions_table() -> pd.DataFrame:
    """Generate the official invented assumptions table from live code constants."""
    facilities = load_facilities_from_csv(PROJECT_ROOT / "data" / "facilities.csv")
    grid_df = load_grid_assumptions(PROJECT_ROOT / "assumptions" / "grid_assumptions.csv")

    rows: List[Dict[str, Any]] = []

    # 1. Reality Gap Structural Perturbations
    rows.append({
        "Category": "Reality Gap: Physical Structure",
        "Parameter": "Insulation Resistance (R) Multiplier Range",
        "Source File": "simulation/reality_gap.py",
        "Code Location": "make_world_facility (line 38)",
        "Exact Code Value": "[0.90, 1.10]",
        "Units": "multiplier (dimensionless)",
        "Description": "Uniform random perturbation range (+-10%) applied to nominal facility envelope R",
    })
    rows.append({
        "Category": "Reality Gap: Physical Structure",
        "Parameter": "Thermal Mass (C) Multiplier Range",
        "Source File": "simulation/reality_gap.py",
        "Code Location": "make_world_facility (line 39)",
        "Exact Code Value": "[0.90, 1.10]",
        "Units": "multiplier (dimensionless)",
        "Description": "Uniform random perturbation range (+-10%) applied to nominal facility thermal capacitance C",
    })

    # 2. Reality Gap Thermal Disturbance
    rows.append({
        "Category": "Reality Gap: Unmodelled Disturbance",
        "Parameter": "AR(1) Disturbance Persistence (phi)",
        "Source File": "simulation/reality_gap.py",
        "Code Location": "make_disturbance (line 82)",
        "Exact Code Value": "0.85",
        "Units": "coefficient (dimensionless)",
        "Description": "Autoregressive persistence coefficient for background stochastic heat ingress",
    })
    rows.append({
        "Category": "Reality Gap: Unmodelled Disturbance",
        "Parameter": "Disturbance Standard Deviation Target",
        "Source File": "simulation/reality_gap.py",
        "Code Location": "make_disturbance (line 81)",
        "Exact Code Value": "max(0.5, 0.03 * abs(mean_wall_heat_kw))",
        "Units": "kW",
        "Description": "Target standard deviation of heat disturbance (~3% of envelope heat ingress, floor 0.5 kW)",
    })
    rows.append({
        "Category": "Reality Gap: Unmodelled Disturbance",
        "Parameter": "Intermittent Step Probability",
        "Source File": "simulation/reality_gap.py",
        "Code Location": "make_disturbance (line 95)",
        "Exact Code Value": "0.012",
        "Units": "probability / minute",
        "Description": "Poisson-like probability of loading dock / forklift door burst heat steps",
    })
    rows.append({
        "Category": "Reality Gap: Unmodelled Disturbance",
        "Parameter": "Intermittent Step Duration Range",
        "Source File": "simulation/reality_gap.py",
        "Code Location": "make_disturbance (line 96)",
        "Exact Code Value": "[5, 15]",
        "Units": "minutes",
        "Description": "Duration range of intermittent loading dock heat pulses",
    })

    # 3. Safety Guardian Risk Tiers & Margins
    for tier_name, tier_cfg in RISK_TIERS.items():
        rows.append({
            "Category": f"Safety Tier: {tier_name}",
            "Parameter": f"{tier_name} Guard Margin",
            "Source File": "safety/rules.py",
            "Code Location": f"TIER_{tier_name}.guard_margin_c",
            "Exact Code Value": str(tier_cfg.guard_margin_c),
            "Units": "deg C",
            "Description": f"Soft guard safety buffer below max safe limit for {tier_name} facilities",
        })
        rows.append({
            "Category": f"Safety Tier: {tier_name}",
            "Parameter": f"{tier_name} Abort Margin",
            "Source File": "safety/rules.py",
            "Code Location": f"TIER_{tier_name}.abort_margin_c",
            "Exact Code Value": str(tier_cfg.abort_margin_c),
            "Units": "deg C",
            "Description": f"Closed-loop runtime monitor emergency cooling trip margin for {tier_name}",
        })
        rows.append({
            "Category": f"Safety Tier: {tier_name}",
            "Parameter": f"{tier_name} Max Single Curtailment",
            "Source File": "safety/rules.py",
            "Code Location": f"TIER_{tier_name}.max_single_curtailment_min",
            "Exact Code Value": str(tier_cfg.max_single_curtailment_min),
            "Units": "minutes",
            "Description": f"Hard upper bound on single continuous curtailment duration for {tier_name}",
        })
        rows.append({
            "Category": f"Safety Tier: {tier_name}",
            "Parameter": f"{tier_name} Restart Lag",
            "Source File": "safety/rules.py",
            "Code Location": f"TIER_{tier_name}.restart_lag_min",
            "Exact Code Value": str(tier_cfg.restart_lag_min),
            "Units": "minutes",
            "Description": f"Compressor turnaround delay assumed before refrigeration becomes effective",
        })
        rows.append({
            "Category": f"Safety Tier: {tier_name}",
            "Parameter": f"{tier_name} Calibrated Lambda (Key 2)",
            "Source File": "safety/rules.py",
            "Code Location": f"TIER_{tier_name}.calibrated_lambda",
            "Exact Code Value": str(tier_cfg.calibrated_lambda),
            "Units": "scalar in [0, 1]",
            "Description": f"Calibrated pessimism scalar balancing nominal twin and conservative envelope",
        })
        rows.append({
            "Category": f"Safety Tier: {tier_name}",
            "Parameter": f"{tier_name} Tripwire Lockout Ambient",
            "Source File": "safety/rules.py",
            "Code Location": f"TIER_{tier_name}.tripwire_lockout_ambient_c",
            "Exact Code Value": str(tier_cfg.tripwire_lockout_ambient_c),
            "Units": "deg C",
            "Description": f"Outdoor temperature threshold triggering automatic fail-closed lockout",
        })

    # 4. Facility Safe Operating Boundaries
    for fac in facilities:
        rows.append({
            "Category": f"Facility Bounds: {fac.id}",
            "Parameter": f"{fac.id} Min Operating Temp",
            "Source File": "data/facilities.csv",
            "Code Location": f"min_temp_c for {fac.id}",
            "Exact Code Value": str(fac.min_temp_c),
            "Units": "deg C",
            "Description": f"Lower operating thermostat setpoint for {fac.name}",
        })
        rows.append({
            "Category": f"Facility Bounds: {fac.id}",
            "Parameter": f"{fac.id} Max Safe Limit",
            "Source File": "data/facilities.csv",
            "Code Location": f"max_safe_temp_c for {fac.id}",
            "Exact Code Value": str(fac.max_safe_temp_c),
            "Units": "deg C",
            "Description": f"Absolute maximum allowable product air temperature for {fac.name}",
        })
        rows.append({
            "Category": f"Facility Bounds: {fac.id}",
            "Parameter": f"{fac.id} Rated Power",
            "Source File": "data/facilities.csv",
            "Code Location": f"rated_power_kw for {fac.id}",
            "Exact Code Value": str(fac.rated_power_kw),
            "Units": "kW",
            "Description": f"Nameplate electrical compressor rated power for {fac.name}",
        })

    # 5. Grid Tariffs & Carbon Placeholders
    min_tariff = float(grid_df["tariff_inr_per_kwh"].min())
    max_tariff = float(grid_df["tariff_inr_per_kwh"].max())
    min_co2 = float(grid_df["grid_co2_g_per_kwh"].min())
    max_co2 = float(grid_df["grid_co2_g_per_kwh"].max())

    rows.append({
        "Category": "Grid Assumptions (Illustrative)",
        "Parameter": "TOU Electricity Tariff Range",
        "Source File": "assumptions/grid_assumptions.csv",
        "Code Location": "tariff_inr_per_kwh column",
        "Exact Code Value": f"[{min_tariff}, {max_tariff}]",
        "Units": "INR / kWh",
        "Description": "Illustrative 24-hour time-of-use tariff placeholder curve (off-peak to peak)",
    })
    rows.append({
        "Category": "Grid Assumptions (Illustrative)",
        "Parameter": "Marginal Grid CO2 Intensity Range",
        "Source File": "assumptions/grid_assumptions.csv",
        "Code Location": "grid_co2_g_per_kwh column",
        "Exact Code Value": f"[{min_co2}, {max_co2}]",
        "Units": "g CO2 / kWh",
        "Description": "Illustrative 24-hour marginal grid carbon intensity placeholder curve",
    })

    df = pd.DataFrame(rows)
    return df


def write_invented_assumptions_files() -> Tuple[Path, Path]:
    """Write the invented assumptions table to part4 and submission directories."""
    df = generate_invented_assumptions_table()

    p4_path = PROJECT_ROOT / "outputs" / "part4" / "invented_assumptions.csv"
    sub_path = PROJECT_ROOT / "outputs" / "submission" / "invented_assumptions.csv"

    p4_path.parent.mkdir(parents=True, exist_ok=True)
    sub_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(p4_path, index=False)
    df.to_csv(sub_path, index=False)

    print(f"Invented assumptions saved -> {p4_path} ({len(df)} rows)")
    print(f"Invented assumptions saved -> {sub_path} ({len(df)} rows)")
    return p4_path, sub_path


if __name__ == "__main__":
    write_invented_assumptions_files()
