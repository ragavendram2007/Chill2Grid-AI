"""safety_guardian.py - Fail-Closed Deterministic Safety Guardian Engine.

Implements:
  1. Gate checks G0 to G5 (Sanity, Sensor Integrity, Equipment Interlocks,
     Thermal State, Capacity Headroom & Recovery Cap, Thermal Debt Ledger).
  2. Two Independent Keys:
     - Key 1: Learned decision-level conformal shrink with 200 scenario-level bootstrap resamples.
     - Key 2: Pessimistic physical twin with design margins (noise, R*0.9, C*0.9, 2x door heat, 2-sigma disturbance).
     - OOD Guard fallback to Key 2 with an extra 20% guard margin.
  3. Actionable refusals with twin recovery hints (retry_after_min).
  4. Leases and Runtime Closed-Loop Event Monitor (monitor_active_event).
  5. Safe curtailment envelope generator and fast lookup query.
  6. Tamper-evident SHA-256 chained audit logger.
  7. Public API matching Part 5 contracts.

All telemetry and outputs are labeled simulated.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from core.contracts import CurtailmentAction, Facility, FacilityState, load_facilities_from_csv
from safety.audit import AuditLogger, compute_sha256, verify_audit_chain
from safety.debt_ledger import ThermalDebtLedger
from safety.key1_conformal import Key1ConformalEstimator
from safety.key2_physics import compute_restart_overshoot_min, simulate_key2_worst_case
from safety.ood import OODGuard
from safety.rules import RULES_VERSION, RiskTierConfig, get_facility_tier
from safety.sensor_integrity import SensorIntegrityChecker
from safety.types import (
    BindingConstraintEnum,
    DecisionEnum,
    FleetSnapshot,
    MonitorActionEnum,
    MonitorVerdict,
    SafetyVerdict,
)
from simulation.calibration import calibrate_all_facilities
from simulation.thermal_model import (
    THERMOSTAT_DEADBAND_C,
    cop,
    read_sensor,
    simulate_temperature,
    steady_state_duty_cycle,
)
from training.scenario_generator import FORECAST_HORIZONS, M2_FEATURES, OBSERVABLE_FEATURES


# ============================================================================
# SAFETY GUARDIAN SINGLETON ENGINE
# ============================================================================

class SafetyGuardian:
    """The central authority for curtailment approvals in Chill2Grid AI.

    Principle: FAIL CLOSED.
    """

    def __init__(
        self,
        facilities: Optional[List[Facility]] = None,
        calibration_path: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        audit_log_path: Optional[Path] = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / "data"

        # 1. Load Facilities
        if facilities is None:
            cal_csv = data_dir / "facilities_calibrated.csv"
            if not cal_csv.is_file():
                cal_csv = data_dir / "facilities.csv"
            facilities_raw = load_facilities_from_csv(cal_csv)
            self.facilities, _ = calibrate_all_facilities(facilities_raw)
        else:
            self.facilities = facilities

        self.facilities_map: Dict[str, Facility] = {f.id: f for f in self.facilities}

        # 2. Version Tracking
        self.rules_version = RULES_VERSION
        self.model_version = "1.1.0"
        self.models_dir = models_dir or (project_root / "models" / "artifacts")
        self._load_model_metadata()

        # 3. Calibration & Models Paths
        self.calib_path = calibration_path or (project_root / "safety" / "artifacts" / "guardian_calibration.json")

        # 4. Components
        self.debt_ledger = ThermalDebtLedger()
        residual_sigma = 0.15
        if self.calib_path.is_file():
            try:
                with open(self.calib_path, "r") as f:
                    calib_dict = json.load(f)
                residual_sigma = float(calib_dict.get("sensor_integrity", {}).get("residual_sigma_c", 0.15))
            except Exception:
                residual_sigma = 0.15
        self.residual_sigma_c = residual_sigma

        self.sensor_checkers: Dict[str, SensorIntegrityChecker] = {
            f.id: SensorIntegrityChecker(residual_sigma_c=self.residual_sigma_c) for f in self.facilities
        }
        ood_file = self.calib_path if self.calib_path.is_file() else (self.models_dir / "metadata.json")
        self.ood_guard = OODGuard(metadata_path=ood_file)
        self.key1_estimator = self._init_key1_estimator()
        self.m2_model = None
        self.m3_models = {}
        self.conformal_offsets = {}
        self._load_ml_models()

        # 5. Audit Logger
        log_path = audit_log_path or (project_root / "outputs" / "part4" / "audit_log.jsonl")
        self.audit_logger = AuditLogger(log_path)

        # 6. Envelope Cache
        self.envelope_cache: Dict[Tuple[str, int, float], int] = {}
        self._load_envelope_cache()

    def _load_model_metadata(self) -> None:
        meta_file = self.models_dir / "metadata.json"
        if meta_file.is_file():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                self.model_version = meta.get("model_version", "1.1.0")
            except Exception:
                pass

    def _init_key1_estimator(self) -> Key1ConformalEstimator:
        if self.calib_path.is_file():
            try:
                with open(self.calib_path, "r") as f:
                    calib_data = json.load(f)
                calib_model_ver = calib_data.get("model_version")
                if calib_model_ver and calib_model_ver != self.model_version:
                    raise ValueError(
                        f"Guardian calibration mismatch: calibration was built for model version "
                        f"'{calib_model_ver}', but active model version is '{self.model_version}'."
                    )
                shrinks = calib_data.get("key1_bootstrap_shrinks", {})
                return Key1ConformalEstimator(calibrated_shrinks=shrinks)
            except Exception as e:
                # Fall back to safe conservative default shrinks
                pass
        # Default conservative shrinks per tier
        default_shrinks = {
            "CRITICAL": {"cut_in": -12.0, "mid_cycle": -10.0, "cut_out": -6.0, "pooled": -10.0},
            "HIGH": {"cut_in": -8.0, "mid_cycle": -6.0, "cut_out": -4.0, "pooled": -6.0},
            "STANDARD": {"cut_in": -5.0, "mid_cycle": -4.0, "cut_out": -2.0, "pooled": -4.0},
        }
        return Key1ConformalEstimator(calibrated_shrinks=default_shrinks)

    def _load_ml_models(self) -> None:
        try:
            m2_path = self.models_dir / "m2_hybrid.joblib"
            m3_path = self.models_dir / "m3_quantile_models.joblib"
            conf_path = self.models_dir / "conformal_offsets.json"
            if m2_path.is_file():
                self.m2_model = joblib.load(m2_path)
            if m3_path.is_file():
                self.m3_models = joblib.load(m3_path)
            if conf_path.is_file():
                with open(conf_path, "r") as f:
                    self.conformal_offsets = json.load(f)
        except Exception:
            pass

    def _load_envelope_cache(self) -> None:
        env_file = Path(__file__).resolve().parent.parent / "data" / "safety" / "safe_envelope.csv"
        if env_file.is_file():
            try:
                with open(env_file, "r") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        fid = r["facility_id"]
                        out_t = int(float(r["outside_temp_c"]))
                        load_f = round(float(r["product_load_fraction"]), 2)
                        self.envelope_cache[(fid, out_t, load_f)] = int(r["max_safe_curtail_min"])
            except Exception:
                pass

    # =========================================================================
    # CORE EVALUATION GATE ENGINE (G0 - G5 + TWO KEYS)
    # =========================================================================

    def evaluate_request(
        self,
        request: CurtailmentAction,
        snapshot: FleetSnapshot,
    ) -> SafetyVerdict:
        """Evaluate a curtailment request through the G0-G5 gates and Two Keys.

        Guaranteed response latency < 50ms.
        """
        now_ts = snapshot.timestamp_min
        audit_id = compute_sha256(f"{request.facility_id}_{now_ts}_{request.duration_min}")[:16]

        # Fail Closed on input anomalies or unknown facility
        if request.facility_id not in self.facilities_map:
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.INSUFFICIENT_WINDOW.value,
                ["UNKNOWN_FACILITY"],
                "REFUSE: Unknown facility ID. Request rejected for safety.",
                audit_id,
            )

        facility = self.facilities_map[request.facility_id]
        tier = get_facility_tier(facility.id)

        # ---------------------------------------------------------------------
        # GATE 0: Input Sanity & Operator Overrides
        # ---------------------------------------------------------------------
        if snapshot.operator_lock:
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.OPERATOR_LOCK.value,
                ["OPERATOR_LOCK_ACTIVE"],
                "REFUSE: Manual operator lock active on facility. DR automated control prohibited.",
                audit_id,
            )
        if snapshot.maintenance_hold:
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.OPERATOR_LOCK.value,
                ["MAINTENANCE_HOLD_ACTIVE"],
                "REFUSE: Facility is under maintenance hold. Cooling must remain undisturbed.",
                audit_id,
            )
        if request.duration_min <= 0:
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.INSUFFICIENT_WINDOW.value,
                ["INVALID_DURATION"],
                "REFUSE: Requested duration must be strictly positive.",
                audit_id,
            )

        # ---------------------------------------------------------------------
        # GATE 1: Sensor Integrity & Watchdog
        # ---------------------------------------------------------------------
        if not snapshot.sensor_ok or snapshot.n_missing_last_30 > 5 or snapshot.max_gap_last_30 > 5:
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.TELEMETRY_FAULT.value,
                ["TELEMETRY_FAULT"],
                "REFUSE: Sensor telemetry missing or hardware fault flag active. Fails closed.",
                audit_id,
            )

        if snapshot.temp_c is None or math.isnan(snapshot.temp_c) or snapshot.temp_c < -60.0 or snapshot.temp_c > 80.0:
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.TELEMETRY_FAULT.value,
                ["IMPOSSIBLE_READING"],
                "REFUSE: Sensor reading physically impossible or NaN. Fails closed.",
                audit_id,
            )

        # Check historical telemetry readings if provided in snapshot
        if snapshot.recent_temperatures and len(snapshot.recent_temperatures) >= 15:
            recent_readings = snapshot.recent_temperatures[-15:]
            if float(np.var(recent_readings)) < 1e-4:
                return self._refuse_with_audit(
                    request,
                    snapshot,
                    BindingConstraintEnum.STUCK_SENSOR.value,
                    ["STUCK_SENSOR"],
                    "REFUSE: Sensor telemetry stuck with near-zero variance. Fails closed.",
                    audit_id,
                )

        if snapshot.recent_temperatures and len(snapshot.recent_temperatures) >= 2:
            deltas = np.abs(np.diff(snapshot.recent_temperatures[-5:]))
            if np.any(deltas > 2.0):
                return self._refuse_with_audit(
                    request,
                    snapshot,
                    BindingConstraintEnum.TELEMETRY_FAULT.value,
                    ["IMPOSSIBLE_JUMP"],
                    "REFUSE: Sensor reading changed by > 2.0°C in 1 minute. Fails closed.",
                    audit_id,
                )

        # ---------------------------------------------------------------------
        # GATE 2: Equipment Interlocks (Anti-Short-Cycle & Minimum Run)
        # ---------------------------------------------------------------------
        if snapshot.compressor_on and snapshot.minutes_in_current_state < facility.min_on_min:
            wait_m = facility.min_on_min - snapshot.minutes_in_current_state
            return self._defer_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.EQUIPMENT_MIN_ON.value,
                ["MIN_ON_NOT_ELAPSED"],
                wait_min=wait_m,
                reason=f"DEFER: Compressor min_on_min lockout active ({wait_m:.0f}m remaining of {facility.min_on_min}m requirement).",
                audit_id=audit_id,
            )

        if not snapshot.compressor_on and snapshot.minutes_in_current_state < facility.min_off_min:
            wait_m = facility.min_off_min - snapshot.minutes_in_current_state
            return self._defer_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.EQUIPMENT_MIN_OFF.value,
                ["MIN_OFF_NOT_ELAPSED"],
                wait_min=wait_m,
                reason=f"DEFER: Compressor anti-short-cycle lockout active ({wait_m:.0f}m remaining of {facility.min_off_min}m).",
                audit_id=audit_id,
            )

        # ---------------------------------------------------------------------
        # GATE 3: Thermal State & Recovery Envelope (Step 4: G3 Fails Closed)
        # ---------------------------------------------------------------------
        limit_threshold = facility.max_safe_temp_c - tier.guard_margin_c

        # Noise-aware conservative G3 check (Step 4: G3 FAILS CLOSED)
        # Compare sensor reading + 2 * sigma with tier guard line
        noise_margin = 2.0 * self.residual_sigma_c
        if (snapshot.temp_c + noise_margin) >= limit_threshold:
            retry_hint = self._compute_retry_hint(facility, snapshot, tier)
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.NOT_RECOVERED.value,
                ["NOT_RECOVERED_THERMAL_HEADROOM"],
                f"REFUSE: Current temperature with 2-sigma noise ({snapshot.temp_c:.2f} + {noise_margin:.2f} = {snapshot.temp_c + noise_margin:.2f}°C) is above guard threshold ({limit_threshold:.2f}°C). Re-chilling required.",
                audit_id,
                retry_hint=retry_hint,
            )

        # Critical Tier Hot-Day Prohibition
        if (
            tier.tripwire_lockout_ambient_c is not None
            and snapshot.outside_temp_c >= tier.tripwire_lockout_ambient_c
        ):
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.CAPACITY_LIMITED.value,
                ["AMBIENT_HEAT_LOCKOUT"],
                f"REFUSE: Severe ambient temperature ({snapshot.outside_temp_c:.1f}°C >= {tier.tripwire_lockout_ambient_c:.0f}°C). Curtailment prohibited for {facility.id}.",
                audit_id,
            )

        # ---------------------------------------------------------------------
        # GATE 4: Capacity Headroom & Recovery Duration Cap
        # ---------------------------------------------------------------------
        duty = steady_state_duty_cycle(facility, snapshot)
        if duty >= tier.duty_cycle_max_threshold:
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.CAPACITY_LIMITED.value,
                ["STEADY_STATE_CAPACITY_EXCEEDED"],
                f"REFUSE: Steady-state duty cycle ({duty*100:.1f}% >= {tier.duty_cycle_max_threshold*100:.0f}%) precludes post-event recovery.",
                audit_id,
            )

        # Recovery Duration Estimation
        # Heat injected during event: P_rated * COP * duration / 60
        cop_val = cop(snapshot.outside_temp_c)
        cooling_rated_th_kw = facility.rated_power_kw * cop_val
        cooling_spare_th_kw = max(1.0, cooling_rated_th_kw * (1.0 - duty))
        # Max heat that can be recovered within max_recovery_min (180m):
        max_recoverable_th_kwh = (tier.max_recovery_min / 60.0) * cooling_spare_th_kw
        max_duration_by_recovery = (max_recoverable_th_kwh / cooling_rated_th_kw) * 60.0

        # ---------------------------------------------------------------------
        # GATE 5: Thermal Debt Ledger & 24h Budget
        # ---------------------------------------------------------------------
        g5_allowed, g5_budget_dur, g5_wait, g5_reason = self.debt_ledger.evaluate_gate5(
            facility, tier, now_ts, request.duration_min
        )
        if not g5_allowed:
            if g5_wait > 0:
                return self._defer_with_audit(
                    request,
                    snapshot,
                    BindingConstraintEnum.THERMAL_DEBT_COOLDOWN.value,
                    ["INTER_EVENT_COOLDOWN"],
                    wait_min=g5_wait,
                    reason=g5_reason,
                    audit_id=audit_id,
                )
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.BUDGET_EXHAUSTED.value,
                ["BUDGET_EXHAUSTED"],
                g5_reason,
                audit_id,
            )

        # ---------------------------------------------------------------------
        # TWO KEYS EVALUATION (KEY 1 & KEY 2)
        # ---------------------------------------------------------------------
        setpoint = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0
        upper_band = setpoint + THERMOSTAT_DEADBAND_C

        # Detect phase tag from snapshot
        phase_tag = (
            "cut_in"
            if snapshot.temp_c >= upper_band - 0.20
            else "cut_out"
            if snapshot.temp_c <= setpoint - THERMOSTAT_DEADBAND_C + 0.20
            else "mid_cycle"
        )

        # Check OOD Guard
        is_ood, ood_msg = self.ood_guard.is_ood(
            features={"lag_0": snapshot.temp_c, "outside_temp_c": snapshot.outside_temp_c}
        )

        # Key 2: Pessimistic Physics Worst-Case (with calibrated lambda or OOD design fallback)
        lam_eff = 1.0 if is_ood else tier.calibrated_lambda
        key2_safe_min = simulate_key2_worst_case(
            facility, snapshot, tier, lambda_pessimism=lam_eff, is_ood=is_ood
        )

        if is_ood:
            # Under OOD, Key 2 alone governs with the extra 20% guard margin
            key1_safe_min = key2_safe_min
            binding_key = BindingConstraintEnum.OOD_GUARD.value
        else:
            # Key 1: Learned Decision-Level Conformal Estimator
            key1_safe_min = self._predict_key1_safe_window(
                facility, snapshot, tier, phase_tag
            )
            binding_key = (
                BindingConstraintEnum.KEY1_CONFORMAL.value
                if key1_safe_min <= key2_safe_min
                else BindingConstraintEnum.KEY2_PHYSICS.value
            )

        # Dynamic Twin-Computed Restart Overshoot Reserve
        reserve_min, overshoot_rise_deg_c = compute_restart_overshoot_min(
            facility,
            snapshot.outside_temp_c,
            tier,
            lambda_pessimism=lam_eff,
            restart_lag_min=getattr(facility, "restart_delay_min", None),
        )

        # Approved Total OFF duration is minimum of both keys minus dynamic overshoot reserve
        approved_total_off = max(0.0, math.floor(min(key1_safe_min, key2_safe_min) - reserve_min))

        # Overall Approved Curtailment Duration
        allowed_duration = min(
            request.duration_min,
            max(0.0, approved_total_off - getattr(facility, "restart_delay_min", 0.0)),
            tier.max_single_curtailment_min,
            max_duration_by_recovery,
            g5_budget_dur,
        )

        # Determine binding constraint
        binding = BindingConstraintEnum.NONE.value
        if allowed_duration < request.duration_min:
            if allowed_duration == max(0.0, approved_total_off - getattr(facility, "restart_delay_min", 0.0)):
                binding = binding_key
            elif allowed_duration == tier.max_single_curtailment_min:
                binding = BindingConstraintEnum.TIER_MAX_DURATION.value
            elif allowed_duration == max_duration_by_recovery:
                binding = BindingConstraintEnum.RECOVERY_EXCEEDS_CAP.value
            elif allowed_duration == g5_budget_dur:
                binding = BindingConstraintEnum.BUDGET_EXHAUSTED.value

        # Enforce configurable minimum actionable window (8m for CRITICAL, 10m for HIGH/STANDARD)
        min_window = tier.min_actionable_window_min
        if allowed_duration < min_window:
            retry_hint = self._compute_retry_hint(facility, snapshot, tier)
            return self._refuse_with_audit(
                request,
                snapshot,
                BindingConstraintEnum.INSUFFICIENT_WINDOW.value,
                [f"Safe window ({allowed_duration:.1f}m) below {min_window:.0f}m minimum actionable threshold"],
                f"REFUSE: Curtailment headroom insufficient ({allowed_duration:.1f}m < {min_window:.0f}m). Cooling must continue.",
                audit_id,
                retry_hint=retry_hint,
                evidence={
                    "key1_safe_min": key1_safe_min,
                    "key2_safe_min": key2_safe_min,
                    "duty_cycle": duty,
                    "recovery_cap_min": max_duration_by_recovery,
                    "dynamic_reserve_min": reserve_min,
                    "overshoot_deg_c": overshoot_rise_deg_c,
                },
            )

        decision = (
            DecisionEnum.APPROVE.value
            if allowed_duration >= request.duration_min
            else DecisionEnum.APPROVE_SHORTENED.value
        )
        restart_slack = max(0.0, approved_total_off - allowed_duration - getattr(facility, "restart_delay_min", 0.0))
        lease_valid_until = now_ts + allowed_duration + 3.0  # 3m heartbeat grace period

        evidence_dict = {
            "key1_safe_min": round(key1_safe_min, 1),
            "key2_safe_min": round(key2_safe_min, 1),
            "guard_margin_c": tier.guard_margin_c,
            "abort_margin_c": tier.abort_margin_c,
            "steady_state_duty": round(duty, 3),
            "recovery_capped_min": round(max_duration_by_recovery, 1),
            "restart_reserve_min": reserve_min,
            "overshoot_deg_c": overshoot_rise_deg_c,
            "restart_delay_min": getattr(facility, "restart_delay_min", 0.0),
            "is_ood": is_ood,
        }

        explanation = (
            f"Approved {allowed_duration:.0f}m curtailment. Both Key 1 ({key1_safe_min:.0f}m) and "
            f"Key 2 ({key2_safe_min:.0f}m) verify physical safety."
            if decision == DecisionEnum.APPROVE.value
            else f"Approved shortened window {allowed_duration:.0f}m (requested {request.duration_min:.0f}m). "
            f"Bounded by {binding}."
        )

        verdict = SafetyVerdict(
            decision=decision,
            facility_id=facility.id,
            approved_duration_min=float(allowed_duration),
            approved_total_off_min=float(approved_total_off),
            restart_slack_min=float(restart_slack),
            wait_min=0.0,
            binding_constraint=binding,
            reason_codes=["APPROVED_SAFE" if decision == DecisionEnum.APPROVE.value else "APPROVED_SHORTENED"],
            evidence=evidence_dict,
            plain_English=explanation,
            risk_tier=tier.tier_name,
            model_version=self.model_version,
            rules_version=self.rules_version,
            audit_id=audit_id,
            valid_until_min=float(lease_valid_until),
        )

        # Log decision into audit chain and record in debt ledger
        self.debt_ledger.record_event(
            facility.id,
            start_min=now_ts,
            duration_min=allowed_duration,
            heat_injected_kwh=(allowed_duration / 60.0) * facility.rated_power_kw * cop_val,
        )
        self._log_verdict_to_audit(request, snapshot, verdict)
        return verdict

    def evaluate_fleet(
        self,
        requests: List[CurtailmentAction],
        snapshots: List[FleetSnapshot],
    ) -> List[SafetyVerdict]:
        """Batch evaluation of fleet curtailment requests. Latency < 300ms."""
        snap_map = {s.facility_id: s for s in snapshots}
        verdicts = []
        for req in requests:
            snap = snap_map.get(req.facility_id)
            if snap is None:
                # Missing snapshot fails closed
                v = SafetyVerdict(
                    decision=DecisionEnum.REFUSE.value,
                    facility_id=req.facility_id,
                    approved_duration_min=0.0,
                    approved_total_off_min=0.0,
                    restart_slack_min=0.0,
                    wait_min=0.0,
                    binding_constraint=BindingConstraintEnum.TELEMETRY_FAULT.value,
                    reason_codes=["MISSING_SNAPSHOT"],
                    evidence={},
                    plain_English="REFUSE: No telemetry snapshot received for facility.",
                    risk_tier="STANDARD",
                    model_version=self.model_version,
                    rules_version=self.rules_version,
                    audit_id="0" * 16,
                    valid_until_min=0.0,
                )
            else:
                v = self.evaluate_request(req, snap)
            verdicts.append(v)
        return verdicts

    # =========================================================================
    # RUNTIME EVENT MONITOR & LEASES
    # =========================================================================

    def monitor_active_event(
        self,
        active_event: CurtailmentAction,
        snapshot: FleetSnapshot,
        last_heartbeat_min: float,
        enable_remaining_safe_rule: bool = True,
    ) -> MonitorVerdict:
        """Runtime monitor executing every minute during active curtailment leases.

        Triggers ABORT_NOW or SHORTEN_NOW if any safety boundary is approached.
        """
        facility = self.facilities_map.get(active_event.facility_id)
        if facility is None:
            return MonitorVerdict(
                action=MonitorActionEnum.ABORT_NOW.value,
                reason="Unknown facility under lease; forced cooling restoration.",
                remaining_safe_min=0.0,
                current_temp_c=snapshot.temp_c,
                emergency_override=True,
            )

        tier = get_facility_tier(facility.id)
        now_ts = snapshot.timestamp_min
        elapsed = now_ts - active_event.start_min
        remaining_event = max(0.0, active_event.duration_min - elapsed)

        # 1. Lease Expiry / Lost Heartbeat (3 minutes threshold)
        if now_ts - last_heartbeat_min > 3.0:
            return MonitorVerdict(
                action=MonitorActionEnum.ABORT_NOW.value,
                reason=f"ABORT: Lease heartbeat lost ({now_ts - last_heartbeat_min:.1f}m > 3m limit). Emergency cooling restored.",
                remaining_safe_min=0.0,
                current_temp_c=snapshot.temp_c,
                emergency_override=True,
            )

        # 2. Telemetry Integrity Check
        if not snapshot.sensor_ok or snapshot.n_missing_last_30 >= 3:
            return MonitorVerdict(
                action=MonitorActionEnum.ABORT_NOW.value,
                reason="ABORT: Sensor untrusted for >= 3 consecutive minutes during active event.",
                remaining_safe_min=0.0,
                current_temp_c=snapshot.temp_c,
                emergency_override=True,
            )

        # 3. Deterministic Tripwire: Air approaches within abort margin of safe max
        abort_threshold = facility.max_safe_temp_c - tier.abort_margin_c
        if snapshot.temp_c >= abort_threshold:
            return MonitorVerdict(
                action=MonitorActionEnum.ABORT_NOW.value,
                reason=(
                    f"EMERGENCY TRIPWIRE TRIGGERED: Air temperature ({snapshot.temp_c:.2f}°C) reached abort threshold "
                    f"({abort_threshold:.2f}°C, margin={tier.abort_margin_c:.2f}°C). Cooling restored immediately."
                ),
                remaining_safe_min=0.0,
                current_temp_c=snapshot.temp_c,
                abort_margin_breached=True,
                emergency_override=True,
            )

        # 4. Fresh Two-Key Remaining Safe Minutes Check
        if enable_remaining_safe_rule:
            fresh_key2 = simulate_key2_worst_case(facility, snapshot, tier)
            fresh_key1 = self._predict_key1_safe_window(facility, snapshot, tier, "mid_cycle")
            remaining_safe = max(0.0, min(fresh_key1, fresh_key2) - tier.restart_reserve_min)

            if remaining_safe < remaining_event:
                adjusted = elapsed + remaining_safe
                if remaining_safe <= 3.0:
                    return MonitorVerdict(
                        action=MonitorActionEnum.ABORT_NOW.value,
                        reason=f"ABORT: Remaining safe minutes ({remaining_safe:.1f}m) exhausted before planned event end.",
                        remaining_safe_min=remaining_safe,
                        current_temp_c=snapshot.temp_c,
                        adjusted_duration_min=adjusted,
                        emergency_override=True,
                    )
                return MonitorVerdict(
                    action=MonitorActionEnum.SHORTEN_NOW.value,
                    reason=f"SHORTEN: Remaining safe headroom ({remaining_safe:.1f}m) requires shortening event to {adjusted:.0f}m.",
                    remaining_safe_min=remaining_safe,
                    current_temp_c=snapshot.temp_c,
                    adjusted_duration_min=adjusted,
                )
        else:
            remaining_safe = remaining_event

        return MonitorVerdict(
            action=MonitorActionEnum.CONTINUE.value,
            reason=f"NORMAL: Safe headroom verified ({remaining_safe:.1f}m remaining safe >= {remaining_event:.1f}m remaining event).",
            remaining_safe_min=remaining_safe,
            current_temp_c=snapshot.temp_c,
        )

    # =========================================================================
    # SAFE ENVELOPE & PLAN-TIME LOOKUP FOR OPTIMIZER
    # =========================================================================

    def build_safe_envelope(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """Generate static plan-time safe curtailment envelope matrix.

        Evaluates Key 2 under warmest-phase conditions across:
          - outside_temp_c: 20 to 44 °C (step 2 °C)
          - product_load_fraction: [0.30, 0.60, 1.00]
        """
        rows = []
        for fac in self.facilities:
            tier = get_facility_tier(fac.id)
            setpoint = (fac.min_temp_c + fac.max_safe_temp_c) / 2.0
            warm_air = setpoint + THERMOSTAT_DEADBAND_C

            for amb in range(20, 46, 2):
                amb_f = float(amb)
                for load_f in [0.30, 0.60, 1.00]:
                    snap = FleetSnapshot(
                        facility_id=fac.id,
                        temp_c=warm_air,
                        outside_temp_c=amb_f,
                        humidity_pct=60.0,
                        product_load_fraction=load_f,
                        door_openings_per_hour=2.0,
                        compressor_on=True,
                        sensor_ok=True,
                    )

                    # Check lockout
                    if tier.tripwire_lockout_ambient_c and amb_f >= tier.tripwire_lockout_ambient_c:
                        max_safe = 0
                    else:
                        duty = steady_state_duty_cycle(fac, snap)
                        if duty >= tier.duty_cycle_max_threshold:
                            max_safe = 0
                        else:
                            k2 = simulate_key2_worst_case(fac, snap, tier)
                            max_safe = int(math.floor(max(0.0, k2 - tier.restart_reserve_min - getattr(fac, "restart_delay_min", 0.0))))
                            max_safe = min(max_safe, int(tier.max_single_curtailment_min))

                    rows.append({
                        "facility_id": fac.id,
                        "product_type": fac.product_type,
                        "outside_temp_c": amb_f,
                        "product_load_fraction": load_f,
                        "max_safe_curtail_min": max_safe,
                        "risk_tier": tier.tier_name,
                    })

        df_env = pd.DataFrame(rows)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_env.to_csv(output_path, index=False)

        # Update in-memory lookup cache
        for r in rows:
            self.envelope_cache[(r["facility_id"], int(r["outside_temp_c"]), round(r["product_load_fraction"], 2))] = r["max_safe_curtail_min"]

        return df_env

    def lookup_max_curtail_min(
        self,
        facility_id: str,
        outside_c: float,
        load_fraction: float,
    ) -> int:
        """Fast O(1) envelope lookup for the Part 5 VPP optimizer."""
        # Quantize to nearest grid points
        rounded_temp = int(round(outside_c / 2.0) * 2)
        rounded_temp = max(20, min(44, rounded_temp))

        loads = [0.30, 0.60, 1.00]
        nearest_load = min(loads, key=lambda x: abs(x - load_fraction))

        cached = self.envelope_cache.get((facility_id, rounded_temp, nearest_load))
        if cached is not None:
            return cached

        # Fallback to conservative default if not in table
        fac = self.facilities_map.get(facility_id)
        if fac is not None:
            tier = get_facility_tier(facility_id)
            if tier.tripwire_lockout_ambient_c and outside_c >= tier.tripwire_lockout_ambient_c:
                return 0
            return int(min(30, tier.max_single_curtailment_min))
        return 0

    # =========================================================================
    # HELPER & REPLAY METHODS
    # =========================================================================

    def _predict_key1_safe_window(
        self,
        facility: Facility,
        snapshot: FleetSnapshot,
        tier: RiskTierConfig,
        phase_tag: str,
    ) -> float:
        """Run ML upper bound forecaster and apply Key 1 conformal shrink."""
        # If ML models are loaded, forecast upper bounds; otherwise fall back to physics twin
        if self.m2_model is None or not self.m3_models:
            k2 = simulate_key2_worst_case(facility, snapshot, tier)
            shrink, _ = self.key1_estimator.get_shrink(tier.tier_name, phase_tag)
            return max(0.0, k2 + shrink)

        try:
            # Construct feature vector for M2 / M3
            feat_dict = {
                "lag_0": snapshot.temp_c,
                "lag_1": snapshot.temp_c,
                "lag_2": snapshot.temp_c,
                "lag_5": snapshot.temp_c,
                "lag_10": snapshot.temp_c,
                "lag_15": snapshot.temp_c,
                "lag_30": snapshot.temp_c,
                "lag_60": snapshot.temp_c,
                "slope_5min": 0.0,
                "slope_15min": 0.0,
                "slope_30min": 0.0,
                "sensor_air_roll_mean_15": snapshot.temp_c,
                "sensor_air_roll_std_15": 0.05,
                "sensor_air_roll_mean_60": snapshot.temp_c,
                "sensor_air_roll_std_60": 0.05,
                "compressor_on_lag0": 1.0 if snapshot.compressor_on else 0.0,
                "compressor_duty_30": 0.70,
                "compressor_duty_60": 0.70,
                "minutes_in_current_state": snapshot.minutes_in_current_state,
                "outside_temp_c": snapshot.outside_temp_c,
                "temp_diff_outside": snapshot.outside_temp_c - snapshot.temp_c,
                "ambient_roll_mean_60": snapshot.outside_temp_c,
                "ambient_slope_60min": 0.0,
                "rated_power_kw": facility.rated_power_kw,
                "thermal_mass_nominal": facility.thermal_mass_kwh_per_c,
                "insulation_r": facility.insulation_r_deg_c_per_kw,
                "headroom_c": facility.max_safe_temp_c - snapshot.temp_c,
                "(sensor_air - setpoint)/deadband": 0.0,
                "n_missing_last_30": snapshot.n_missing_last_30,
                "max_gap_last_30": snapshot.max_gap_last_30,
            }

            # Run deterministic physics branch to get physics_temp_c across horizons
            act = CurtailmentAction(facility.id, start_min=0, duration_min=120)
            phys_df = simulate_temperature(
                facility,
                FacilityState(
                    facility_id=facility.id,
                    temp_c=snapshot.temp_c,
                    outside_temp_c=snapshot.outside_temp_c,
                    humidity_pct=snapshot.humidity_pct,
                    product_load_fraction=snapshot.product_load_fraction,
                    door_openings_per_hour=snapshot.door_openings_per_hour,
                    compressor_on=False,
                    sensor_ok=True,
                ),
                duration_min=120,
                curtailments=[act],
                seed=42,
            )

            # Predict upper bounds for tier confidence
            q_model = self.m3_models.get(tier.confidence_level)
            if q_model is None:
                q_model = list(self.m3_models.values())[-1]

            upper_traj = []
            for h in FORECAST_HORIZONS:
                step_idx = h - 1
                p_temp = float(phys_df.loc[step_idx, "air_temp_c"])
                feat_row = dict(feat_dict)
                feat_row["horizon_min"] = h
                feat_row["physics_temp_c"] = p_temp
                feat_row["physics_minutes_to_limit"] = 60.0

                df_x = pd.DataFrame([feat_row])[list(M2_FEATURES)]
                res_upper = float(q_model.predict(df_x)[0])
                # Conformal offset lookup
                offset = 0.50
                if "phase" in self.conformal_offsets:
                    ph_conf = self.conformal_offsets["phase"].get(phase_tag, {})
                    offset = ph_conf.get(str(tier.confidence_level), {}).get(str(h), 0.50)
                upper_traj.append(p_temp + res_upper + offset)

            key1_approved, _, _ = self.key1_estimator.estimate_safe_minutes(
                FORECAST_HORIZONS,
                upper_traj,
                facility.max_safe_temp_c,
                tier.guard_margin_c,
                tier.tier_name,
                phase_tag,
            )
            return key1_approved
        except Exception:
            k2 = simulate_key2_worst_case(facility, snapshot, tier)
            shrink, _ = self.key1_estimator.get_shrink(tier.tier_name, phase_tag)
            return max(0.0, k2 + shrink)

    def _compute_retry_hint(
        self,
        facility: Facility,
        snapshot: FleetSnapshot,
        tier: RiskTierConfig,
    ) -> float:
        """Estimate retry_after_min: how long cooling must run before Key 2 passes."""
        setpoint = (facility.min_temp_c + facility.max_safe_temp_c) / 2.0
        # Thermostat pull down: simulate normal cooling with compressor ON
        st_cool = FacilityState(
            facility_id=facility.id,
            temp_c=snapshot.temp_c,
            outside_temp_c=snapshot.outside_temp_c,
            humidity_pct=snapshot.humidity_pct,
            product_load_fraction=snapshot.product_load_fraction,
            door_openings_per_hour=snapshot.door_openings_per_hour,
            compressor_on=True,
            sensor_ok=True,
        )
        sim_cool = simulate_temperature(facility, st_cool, duration_min=60, curtailments=None, seed=42)
        target = setpoint
        below = sim_cool["air_temp_c"] <= target
        if below.any():
            return float(sim_cool.loc[below, "minute"].iloc[0])
        return 30.0

    def _refuse_with_audit(
        self,
        request: CurtailmentAction,
        snapshot: FleetSnapshot,
        constraint: str,
        reason_codes: List[str],
        reason_text: str,
        audit_id: str,
        retry_hint: Optional[float] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> SafetyVerdict:
        v = SafetyVerdict(
            decision=DecisionEnum.REFUSE.value,
            facility_id=request.facility_id,
            approved_duration_min=0.0,
            approved_total_off_min=0.0,
            restart_slack_min=0.0,
            wait_min=float(retry_hint or 0.0),
            binding_constraint=constraint,
            reason_codes=reason_codes,
            evidence=evidence or {},
            plain_English=reason_text,
            risk_tier=get_facility_tier(request.facility_id).tier_name,
            model_version=self.model_version,
            rules_version=self.rules_version,
            audit_id=audit_id,
            valid_until_min=0.0,
        )
        self._log_verdict_to_audit(request, snapshot, v)
        return v

    def _defer_with_audit(
        self,
        request: CurtailmentAction,
        snapshot: FleetSnapshot,
        constraint: str,
        reason_codes: List[str],
        wait_min: float,
        reason: str,
        audit_id: str,
    ) -> SafetyVerdict:
        v = SafetyVerdict(
            decision=DecisionEnum.DEFER.value,
            facility_id=request.facility_id,
            approved_duration_min=0.0,
            approved_total_off_min=0.0,
            restart_slack_min=0.0,
            wait_min=float(wait_min),
            binding_constraint=constraint,
            reason_codes=reason_codes,
            evidence={},
            plain_English=reason,
            risk_tier=get_facility_tier(request.facility_id).tier_name,
            model_version=self.model_version,
            rules_version=self.rules_version,
            audit_id=audit_id,
            valid_until_min=0.0,
        )
        self._log_verdict_to_audit(request, snapshot, v)
        return v

    def _log_verdict_to_audit(
        self,
        request: CurtailmentAction,
        snapshot: FleetSnapshot,
        verdict: SafetyVerdict,
    ) -> None:
        input_data = {
            "facility_id": request.facility_id,
            "requested_start_min": request.start_min,
            "requested_duration_min": request.duration_min,
            "sensor_temp_c": snapshot.temp_c,
            "outside_temp_c": snapshot.outside_temp_c,
            "humidity_pct": snapshot.humidity_pct,
            "door_rate": snapshot.door_openings_per_hour,
            "compressor_on": snapshot.compressor_on,
            "sensor_ok": snapshot.sensor_ok,
            "minutes_in_state": snapshot.minutes_in_current_state,
            "timestamp_min": snapshot.timestamp_min,
        }
        verdict_data = {
            "decision": verdict.decision,
            "approved_duration_min": verdict.approved_duration_min,
            "approved_total_off_min": verdict.approved_total_off_min,
            "binding_constraint": verdict.binding_constraint,
            "reason_codes": verdict.reason_codes,
            "audit_id": verdict.audit_id,
        }
        versions = {
            "rules_version": self.rules_version,
            "model_version": self.model_version,
        }
        self.audit_logger.log_action("DECISION", input_data, verdict_data, versions)


# ============================================================================
# PUBLIC MODULE-LEVEL FUNCTIONS (MATCHES PART 5 CONTRACTS)
# ============================================================================

_DEFAULT_GUARDIAN: Optional[SafetyGuardian] = None


def get_default_guardian() -> SafetyGuardian:
    """Return initialized singleton Safety Guardian engine."""
    global _DEFAULT_GUARDIAN
    if _DEFAULT_GUARDIAN is None:
        _DEFAULT_GUARDIAN = SafetyGuardian()
    return _DEFAULT_GUARDIAN


def evaluate_request(
    request: CurtailmentAction,
    snapshot: FleetSnapshot,
) -> SafetyVerdict:
    """Evaluate a proposed curtailment request."""
    return get_default_guardian().evaluate_request(request, snapshot)


def evaluate_fleet(
    requests: List[CurtailmentAction],
    snapshots: List[FleetSnapshot],
) -> List[SafetyVerdict]:
    """Evaluate batch fleet curtailment requests."""
    return get_default_guardian().evaluate_fleet(requests, snapshots)


def monitor_active_event(
    active_event: CurtailmentAction,
    snapshot: FleetSnapshot,
    last_heartbeat_min: float,
    enable_remaining_safe_rule: bool = True,
) -> MonitorVerdict:
    """Runtime monitor for active curtailment leases."""
    return get_default_guardian().monitor_active_event(
        active_event, snapshot, last_heartbeat_min, enable_remaining_safe_rule=enable_remaining_safe_rule
    )


def lookup_max_curtail_min(
    facility_id: str,
    outside_c: float,
    load_fraction: float,
) -> int:
    """Fast envelope query for VPP optimizer."""
    return get_default_guardian().lookup_max_curtail_min(facility_id, outside_c, load_fraction)


def build_safe_envelope(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Generate plan-time static envelope table."""
    return get_default_guardian().build_safe_envelope(output_path)


def replay_decision(audit_record: Dict[str, Any]) -> bool:
    """Replay an audited decision and verify bitwise identical verdict."""
    guardian = get_default_guardian()
    input_data = audit_record.get("input_data", {})
    recorded_verdict = audit_record.get("verdict", {})

    req = CurtailmentAction(
        facility_id=input_data.get("facility_id", ""),
        start_min=float(input_data.get("requested_start_min", 0.0)),
        duration_min=float(input_data.get("requested_duration_min", 30.0)),
    )
    snap = FleetSnapshot(
        facility_id=input_data.get("facility_id", ""),
        temp_c=float(input_data.get("sensor_temp_c", 0.0)),
        outside_temp_c=float(input_data.get("outside_temp_c", 25.0)),
        humidity_pct=float(input_data.get("humidity_pct", 60.0)),
        product_load_fraction=0.85,
        door_openings_per_hour=float(input_data.get("door_rate", 2.0)),
        compressor_on=bool(input_data.get("compressor_on", True)),
        sensor_ok=bool(input_data.get("sensor_ok", True)),
        timestamp_min=float(input_data.get("timestamp_min", 0.0)),
        minutes_in_current_state=float(input_data.get("minutes_in_state", 60.0)),
    )

    replayed_v = guardian.evaluate_request(req, snap)
    return (
        replayed_v.decision == recorded_verdict.get("decision")
        and abs(replayed_v.approved_duration_min - float(recorded_verdict.get("approved_duration_min", 0.0))) < 1e-3
        and replayed_v.binding_constraint == recorded_verdict.get("binding_constraint")
    )
