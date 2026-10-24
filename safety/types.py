"""types.py - Dataclasses and Enums for Safety Guardian.

All types defined here are internal to Part 4, ensuring zero modifications to core/contracts.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DecisionEnum(str, Enum):
    """Safety Guardian high-level verdict decision."""

    APPROVE = "APPROVE"
    APPROVE_SHORTENED = "APPROVE_SHORTENED"
    DEFER = "DEFER"
    REFUSE = "REFUSE"


class BindingConstraintEnum(str, Enum):
    """Root physical or procedural cause that clamped or refused a request."""

    EQUIPMENT_MIN_ON = "EQUIPMENT_MIN_ON"
    EQUIPMENT_MIN_OFF = "EQUIPMENT_MIN_OFF"
    THERMAL_DEBT_COOLDOWN = "THERMAL_DEBT_COOLDOWN"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TELEMETRY_FAULT = "TELEMETRY_FAULT"
    STUCK_SENSOR = "STUCK_SENSOR"
    SENSOR_DRIFT = "SENSOR_DRIFT"
    NOT_RECOVERED = "NOT_RECOVERED"
    CAPACITY_LIMITED = "CAPACITY_LIMITED"
    RECOVERY_EXCEEDS_CAP = "RECOVERY_EXCEEDS_CAP"
    KEY1_CONFORMAL = "KEY1_CONFORMAL"
    KEY2_PHYSICS = "KEY2_PHYSICS"
    TIER_MAX_DURATION = "TIER_MAX_DURATION"
    OPERATOR_LOCK = "OPERATOR_LOCK"
    OOD_GUARD = "OOD_GUARD"
    INSUFFICIENT_WINDOW = "INSUFFICIENT_WINDOW"
    NONE = "NONE"


class MonitorActionEnum(str, Enum):
    """Runtime lease monitor actions."""

    CONTINUE = "CONTINUE"
    SHORTEN_NOW = "SHORTEN_NOW"
    ABORT_NOW = "ABORT_NOW"


@dataclass
class SafetyVerdict:
    """Complete, actionable verdict issued by the Safety Guardian for a proposed curtailment."""

    decision: str  # APPROVE / APPROVE_SHORTENED / DEFER / REFUSE
    facility_id: str
    approved_duration_min: float
    approved_total_off_min: float
    restart_slack_min: float
    wait_min: float
    binding_constraint: str
    reason_codes: List[str]
    evidence: Dict[str, Any]
    plain_English: str
    risk_tier: str
    model_version: str
    rules_version: str
    audit_id: str
    valid_until_min: float  # Lease expiration timestamp

    @property
    def is_safe(self) -> bool:
        """True if any non-zero curtailment duration was approved."""
        return self.decision in (DecisionEnum.APPROVE.value, DecisionEnum.APPROVE_SHORTENED.value) and self.approved_duration_min >= 10.0

    @property
    def is_approved(self) -> bool:
        """True if fully or partially approved."""
        return self.is_safe


@dataclass
class MonitorVerdict:
    """Action returned by the runtime monitor during an active curtailment lease."""

    action: str  # CONTINUE / SHORTEN_NOW / ABORT_NOW
    reason: str
    remaining_safe_min: float
    current_temp_c: float
    abort_margin_breached: bool = False
    emergency_override: bool = False
    adjusted_duration_min: Optional[float] = None


@dataclass
class FleetSnapshot:
    """Snapshot of facility state, telemetry health, and recent operational history."""

    facility_id: str
    temp_c: float
    outside_temp_c: float
    humidity_pct: float
    product_load_fraction: float
    door_openings_per_hour: float
    compressor_on: bool
    sensor_ok: bool
    timestamp_min: float = 0.0
    minutes_in_current_state: float = 60.0
    operator_lock: bool = False
    maintenance_hold: bool = False
    n_missing_last_30: int = 0
    max_gap_last_30: int = 0
    recent_temperatures: List[float] = field(default_factory=list)
    recent_curtailments: List[Dict[str, Any]] = field(default_factory=list)
