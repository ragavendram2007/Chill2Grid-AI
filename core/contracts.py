"""contracts.py - Chill2Grid AI Core Contracts and Constants.

This module defines the single source of truth for dataclasses, units,
system constants, and schema validation across the entire Chill2Grid AI project.
All modules interact strictly through these dataclasses.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# SYSTEM CONSTANTS & CITATIONS (Section H)
# ============================================================================

# Default simulation time step in minutes (discrete time progression)
TIME_STEP_MIN: float = 1.0

# Central Electricity Authority (CEA), Ministry of Power, Government of India:
# "CO2 Baseline Database for the Indian Power Sector", User Guide Version 19.0 (Dec 2023).
# National Grid weighted average emission factor: 0.716 kg CO2 / kWh.
# NOTE: All emissions accounting results are analytical estimates for simulated scenarios.
CO2_GRID_EMISSION_FACTOR_KG_PER_KWH: float = 0.716
CO2_GRID_EMISSION_SOURCE: str = (
    "CEA India CO2 Baseline Database v19.0 (Dec 2023), 0.716 kg CO2/kWh (estimate)"
)

# Demand flexibility incentive rate in Indian Rupees per eligible curtailed kWh.
# Simulated tariff modeled after Indian DISCOM commercial peak-shaving pilots.
SIMULATED_REWARD_RATE_INR_PER_KWH: float = 8.00
SIMULATED_REWARD_SOURCE: str = (
    "Simulated tariff benchmarked at Rs 8.00/kWh for commercial demand flexibility"
)

# Safety percentile: 90th percentile worst-case temperature used for safety guarantees
CONSERVATIVE_PERCENTILE: float = 90.0

# Standard audit labels
LABEL_SIMULATED: str = "simulated"
LABEL_ESTIMATE: str = "estimate"


# ============================================================================
# CORE DATACLASSES
# ============================================================================

@dataclass
class Facility:
    """Represents a refrigerated cold-storage warehouse or temperature-controlled room.

    Units:
      - thermal_mass_kwh_per_c: Thermal inertia / heat capacity (kWh / °C).
      - insulation_r_deg_c_per_kw: Thermal resistance to ambient heat ingress (°C / kW).
      - rated_power_kw: Nominal electrical power draw of the refrigeration plant (kW).
      - min_temp_c: Target lowest operating temperature setpoint (°C).
      - max_safe_temp_c: Absolute upper threshold for product safety (°C).
      - min_on_min: Compressor anti-cycling minimum run duration (minutes).
      - min_off_min: Compressor anti-cycling minimum rest duration (minutes).
      - cop_nominal: Nominal coefficient of performance (thermal cooling kW / electrical kW).
    """

    id: str
    product_type: str
    thermal_mass_kwh_per_c: float
    insulation_r_deg_c_per_kw: float
    rated_power_kw: float
    min_temp_c: float
    max_safe_temp_c: float
    min_on_min: int
    min_off_min: int
    name: str = ""
    cop_nominal: float = 2.5

    def __post_init__(self) -> None:
        """Validate realistic physical parameters."""
        if not self.id or not str(self.id).strip():
            raise ValueError("Facility 'id' must be a non-empty string.")
        if not self.name:
            self.name = self.id
        if self.min_temp_c >= self.max_safe_temp_c:
            raise ValueError(
                f"Facility '{self.id}': min_temp_c ({self.min_temp_c}°C) must be "
                f"strictly less than max_safe_temp_c ({self.max_safe_temp_c}°C)."
            )
        if self.rated_power_kw <= 0.0:
            raise ValueError(
                f"Facility '{self.id}': rated_power_kw ({self.rated_power_kw} kW) must be positive."
            )
        if self.thermal_mass_kwh_per_c <= 0.0:
            raise ValueError(
                f"Facility '{self.id}': thermal_mass_kwh_per_c ({self.thermal_mass_kwh_per_c}) must be positive."
            )
        if self.insulation_r_deg_c_per_kw <= 0.0:
            raise ValueError(
                f"Facility '{self.id}': insulation_r_deg_c_per_kw ({self.insulation_r_deg_c_per_kw}) must be positive."
            )
        if self.min_on_min < 0 or self.min_off_min < 0:
            raise ValueError(
                f"Facility '{self.id}': min_on_min and min_off_min must be non-negative."
            )
        if self.cop_nominal <= 0.0:
            raise ValueError(
                f"Facility '{self.id}': cop_nominal ({self.cop_nominal}) must be positive."
            )

    @property
    def thermal_mass(self) -> float:
        """Convenience alias for thermal_mass_kwh_per_c."""
        return self.thermal_mass_kwh_per_c

    @property
    def insulation_r(self) -> float:
        """Convenience alias for insulation_r_deg_c_per_kw."""
        return self.insulation_r_deg_c_per_kw


@dataclass
class FacilityState:
    """Instantaneous operational and telemetry snapshot of a cold storage facility.

    Units:
      - temp_c: Current indoor product/air temperature (°C).
      - outside_temp_c: Ambient outdoor weather temperature (°C).
      - humidity_pct: Relative indoor/outdoor humidity (0% - 100%).
      - product_load_fraction: Current thermal fill capacity (0.0 = empty, 1.0 = full).
      - door_openings_per_hour: Frequency of loading dock door openings (events/hour).
      - compressor_on: Boolean flag indicating if refrigeration compressor is active.
      - sensor_ok: Boolean flag indicating health of telemetry sensors.
      - timestamp_min: Elapsed minutes from simulation start.
    """

    temp_c: float
    outside_temp_c: float
    humidity_pct: float
    product_load_fraction: float
    door_openings_per_hour: float
    compressor_on: bool
    sensor_ok: bool
    facility_id: str = ""
    timestamp_min: float = 0.0

    def __post_init__(self) -> None:
        """Validate telemetry ranges."""
        if not (0.0 <= self.humidity_pct <= 100.0):
            raise ValueError(
                f"humidity_pct ({self.humidity_pct}) must be within [0.0, 100.0]."
            )
        if not (0.0 <= self.product_load_fraction <= 1.0):
            raise ValueError(
                f"product_load_fraction ({self.product_load_fraction}) must be within [0.0, 1.0]."
            )
        if self.door_openings_per_hour < 0.0:
            raise ValueError(
                f"door_openings_per_hour ({self.door_openings_per_hour}) cannot be negative."
            )


@dataclass
class CurtailmentAction:
    """Command requesting or scheduling curtailment of a facility.

    Units:
      - start_min: Start time of curtailment event in simulation minutes.
      - duration_min: Requested duration of curtailment in minutes.
      - target_reduction_kw: Targeted power curtailment in kW (0.0 = full curtailment).
    """

    facility_id: str
    start_min: int
    duration_min: int
    target_reduction_kw: float = 0.0

    def __post_init__(self) -> None:
        """Validate action schedule."""
        if not self.facility_id:
            raise ValueError("CurtailmentAction requires a valid facility_id.")
        if self.start_min < 0:
            raise ValueError("start_min cannot be negative.")
        if self.duration_min <= 0:
            raise ValueError("duration_min must be strictly positive.")


@dataclass
class SafetyDecision:
    """Deterministic verdict issued by the Safety Guardian for a proposed action.

    Safety is never compromised: deterministic rules always override ML estimates.
    """

    facility_id: str
    allowed: bool
    reason: str
    worst_case_temp_c: float
    safe_duration_min: float = 0.0

    @property
    def is_safe(self) -> bool:
        """Convenience alias for allowed."""
        return self.allowed


@dataclass
class FlexibilityEstimate:
    """Estimated demand response flexibility metrics for a specific facility.

    Units:
      - baseline_power_kw: Actual baseline electrical draw (duty cycle * rated kW).
      - curtailable_power_kw: Power that can safely be shed (limited by baseline_power_kw).
      - safe_duration_min: Minutes the facility can remain shut off without breaching max safe temp.
      - confidence_score: Confidence estimate [0.0, 1.0] based on model prediction uncertainty.
    """

    facility_id: str
    baseline_power_kw: float
    curtailable_power_kw: float
    safe_duration_min: float
    confidence_score: float
    reason: str

    def __post_init__(self) -> None:
        """Validate flexibility values."""
        if self.baseline_power_kw < 0.0:
            raise ValueError("baseline_power_kw cannot be negative.")
        if self.curtailable_power_kw < 0.0:
            raise ValueError("curtailable_power_kw cannot be negative.")
        if self.curtailable_power_kw > self.baseline_power_kw + 1e-4:
            raise ValueError(
                f"curtailable_power_kw ({self.curtailable_power_kw} kW) cannot exceed "
                f"actual baseline_power_kw ({self.baseline_power_kw} kW)."
            )
        if self.safe_duration_min < 0.0:
            raise ValueError("safe_duration_min cannot be negative.")
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError("confidence_score must be between 0.0 and 1.0.")

    @property
    def baseline_kw(self) -> float:
        """Alias for baseline_power_kw."""
        return self.baseline_power_kw

    @property
    def curtailable_kw(self) -> float:
        """Alias for curtailable_power_kw."""
        return self.curtailable_power_kw

    @property
    def safe_minutes(self) -> float:
        """Alias for safe_duration_min."""
        return self.safe_duration_min

    @property
    def confidence(self) -> float:
        """Alias for confidence_score."""
        return self.confidence_score


@dataclass
class EventResult:
    """Comprehensive accounting of a VPP grid event, including recovery and carbon impact.

    Honest Accounting Rule:
      - event_energy_saved_kwh: Gross energy shed during the curtailment window.
      - recovery_energy_added_kwh: Extra energy consumed during recovery pull-down.
      - net_energy_saved_kwh: True net energy savings (event_energy_saved - recovery_energy_added).
      - co2_reduction_estimate_kg: Net CO2 offset using CEA India factor (labeled as estimate).
      - simulated_reward_inr: Financial benefit calculated at Rs 8.00/kWh (labeled as simulated).
      - shortfall_power_kw: grid request shortfall (if requested kW exceeds safe available kW).
    """

    event_name: str
    target_power_kw: float
    delivered_power_kw: float
    shortfall_power_kw: float
    duration_min: float
    event_energy_saved_kwh: float
    recovery_energy_added_kwh: float
    net_energy_saved_kwh: float
    co2_reduction_estimate_kg: float
    simulated_reward_inr: float
    selected_facilities: List[str] = field(default_factory=list)
    rejected_facilities: Dict[str, str] = field(default_factory=dict)
    grid_emission_factor_kg_per_kwh: float = CO2_GRID_EMISSION_FACTOR_KG_PER_KWH
    reward_rate_inr_per_kwh: float = SIMULATED_REWARD_RATE_INR_PER_KWH
    peak_rebound_kw: float = 0.0
    rebound_exceeds_baseline: bool = False
    notes: str = ""

    def summary_dict(self) -> Dict[str, object]:
        """Return a structured dictionary with honest simulation labels."""
        return {
            "event_name": self.event_name,
            "target_power_kw": round(self.target_power_kw, 2),
            "delivered_power_kw": round(self.delivered_power_kw, 2),
            "shortfall_power_kw": round(self.shortfall_power_kw, 2),
            "duration_min": round(self.duration_min, 1),
            "event_energy_saved_kwh (simulated)": round(self.event_energy_saved_kwh, 2),
            "recovery_energy_added_kwh (simulated)": round(self.recovery_energy_added_kwh, 2),
            "net_energy_saved_kwh (simulated)": round(self.net_energy_saved_kwh, 2),
            "co2_reduction_kg (estimate)": round(self.co2_reduction_estimate_kg, 2),
            "simulated_reward_inr (simulated)": round(self.simulated_reward_inr, 2),
            "participating_facilities_count": len(self.selected_facilities),
            "rejected_facilities_count": len(self.rejected_facilities),
            "rebound_exceeds_baseline": self.rebound_exceeds_baseline,
        }


# ============================================================================
# DATA INGESTION & VALIDATION
# ============================================================================

def load_facilities_from_csv(csv_path: str | Path) -> List[Facility]:
    """Load, parse, and validate facility definitions from a CSV file.

    Args:
        csv_path: Absolute or relative path to facilities CSV file.

    Returns:
        List of validated Facility instances.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If headers are missing or numeric fields are invalid.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Facilities CSV file not found at: {path.resolve()}")

    required_columns = {
        "facility_id",
        "product_type",
        "thermal_mass_kwh_per_c",
        "insulation_r_deg_c_per_kw",
        "rated_power_kw",
        "min_temp_c",
        "max_safe_temp_c",
        "min_on_min",
        "min_off_min",
    }

    facilities: List[Facility] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file at {path} is empty or lacks headers.")

        missing_cols = required_columns - set(reader.fieldnames)
        if missing_cols:
            raise ValueError(
                f"CSV at {path} is missing required columns: {sorted(missing_cols)}"
            )

        for row_idx, row in enumerate(reader, start=2):
            try:
                fac = Facility(
                    id=row["facility_id"].strip(),
                    name=row.get("name", "").strip() or row["facility_id"].strip(),
                    product_type=row["product_type"].strip(),
                    thermal_mass_kwh_per_c=float(row["thermal_mass_kwh_per_c"]),
                    insulation_r_deg_c_per_kw=float(row["insulation_r_deg_c_per_kw"]),
                    rated_power_kw=float(row["rated_power_kw"]),
                    min_temp_c=float(row["min_temp_c"]),
                    max_safe_temp_c=float(row["max_safe_temp_c"]),
                    min_on_min=int(float(row["min_on_min"])),
                    min_off_min=int(float(row["min_off_min"])),
                    cop_nominal=float(row.get("cop_nominal", 2.5)),
                )
                facilities.append(fac)
            except (ValueError, KeyError) as err:
                raise ValueError(
                    f"Error parsing row {row_idx} in {path.name}: {err}"
                ) from err

    if not facilities:
        raise ValueError(f"No facility records found in {path}.")

    return facilities


# ============================================================================
# CLI DEMO DISPLAY
# ============================================================================

def format_facilities_table(facilities: List[Facility]) -> str:
    """Format a list of Facility objects into an aligned ASCII text table."""
    headers = [
        "Facility ID",
        "Product Type",
        "Rated kW",
        "Temp Range (C)",
        "Safe Max (C)",
        "Mass (kWh/C)",
        "R (C/kW)",
        "COP",
    ]
    rows = []
    for fac in facilities:
        temp_range = f"{fac.min_temp_c:+.1f} to {fac.max_safe_temp_c:+.1f}"
        rows.append([
            fac.id,
            fac.product_type[:24],
            f"{fac.rated_power_kw:.1f}",
            temp_range,
            f"{fac.max_safe_temp_c:+.1f}",
            f"{fac.thermal_mass_kwh_per_c:.1f}",
            f"{fac.insulation_r_deg_c_per_kw:.2f}",
            f"{fac.cop_nominal:.1f}",
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_str = (
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    )

    lines = [
        "=" * len(separator),
        "CHILL2GRID AI - REGISTERED COLD STORAGE FLEET (SIMULATED)",
        "=" * len(separator),
        separator,
        header_str,
        separator,
    ]
    for r in rows:
        row_str = (
            "| "
            + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(r)))
            + " |"
        )
        lines.append(row_str)
    lines.append(separator)

    # Footnotes
    lines.append(f"* National Grid Emission Factor: {CO2_GRID_EMISSION_SOURCE}")
    lines.append(f"* Simulated DR Reward Rate: {SIMULATED_REWARD_SOURCE}")
    lines.append("* Note: All telemetry and models are labeled as simulated / estimate.")
    return "\n".join(lines)


if __name__ == "__main__":
    # Tiny demo command for Part 1
    default_csv = Path(__file__).resolve().parent.parent / "data" / "facilities.csv"
    loaded = load_facilities_from_csv(default_csv)
    print(format_facilities_table(loaded))
