"""candidates.py - Phase-Robust Candidate Generation with Guardian in the Loop.

Rolls facility forward under K = 8 sampled futures (varying initial phases and door opening seeds).
Evaluates Guardian evaluate_request across each future.
Enforces that a candidate is plan-feasible only if the Guardian approves in >= 7 of 8 futures.
Sets usable duration = min(approved_duration across approving futures).
Uses conservative P10 kW cut from response library.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.contracts import CurtailmentAction, Facility, FacilityState
from optimizer.build_library import query_library_delta
from optimizer.types import Candidate, OptimizerConfig
from safety.safety_guardian import SafetyGuardian, lookup_max_curtail_min
from safety.types import DecisionEnum, FleetSnapshot, SafetyVerdict
from simulation.thermal_model import simulate_temperature


def precompute_future_snapshots(
    facility: Facility,
    base_snapshot: FleetSnapshot,
    offsets: List[int],
    outside_temp_c: float,
    k_futures: int = 8,
) -> Dict[int, List[FleetSnapshot]]:
    """Precompute the K future snapshot states for each offset once per facility.

    Avoids re-running simulations for every duration/delay combination.
    """
    future_snaps_by_offset: Dict[int, List[FleetSnapshot]] = {}

    for offset in offsets:
        snaps_k: List[FleetSnapshot] = []
        event_start = base_snapshot.timestamp_min + float(offset)

        for k in range(k_futures):
            if offset > 0:
                roll_state = FacilityState(
                    facility_id=facility.id,
                    temp_c=base_snapshot.temp_c,
                    outside_temp_c=outside_temp_c,
                    humidity_pct=base_snapshot.humidity_pct,
                    product_load_fraction=base_snapshot.product_load_fraction,
                    door_openings_per_hour=base_snapshot.door_openings_per_hour,
                    compressor_on=bool((k % 2) == 1) if k > 0 else base_snapshot.compressor_on,
                    sensor_ok=True,
                    timestamp_min=base_snapshot.timestamp_min,
                )
                outside_prof = [outside_temp_c] * max(720, offset + 10)
                roll_df = simulate_temperature(
                    facility=facility,
                    state=roll_state,
                    duration_min=offset,
                    seed=k * 100 + 42,
                    outside_profile_c=outside_prof,
                )
                final_row = roll_df.iloc[-1]
                t_true = float(final_row["air_temp_c"])
                c_on = bool(final_row["compressor_on"])
                c_vals = roll_df["compressor_on"].values
                mins_state = 1.0
                for idx_m in range(len(c_vals) - 2, -1, -1):
                    if c_vals[idx_m] == c_on:
                        mins_state += 1.0
                    else:
                        break
            else:
                t_true = base_snapshot.temp_c
                c_on = base_snapshot.compressor_on
                mins_state = base_snapshot.minutes_in_current_state

            # Noisy sensor view: N(0, sigma^2) with sigma = 0.306 C
            np.random.seed(k * 7 + 13)
            sensor_noise = float(np.random.normal(0.0, 0.306))
            t_sensor = t_true + sensor_noise

            snap = FleetSnapshot(
                facility_id=facility.id,
                temp_c=t_sensor,
                outside_temp_c=outside_temp_c,
                humidity_pct=base_snapshot.humidity_pct,
                product_load_fraction=base_snapshot.product_load_fraction,
                door_openings_per_hour=base_snapshot.door_openings_per_hour,
                compressor_on=c_on,
                sensor_ok=True,
                timestamp_min=event_start,
                minutes_in_current_state=mins_state,
            )
            snaps_k.append(snap)

        future_snaps_by_offset[offset] = snaps_k

    return future_snaps_by_offset


def evaluate_candidate_across_futures(
    facility: Facility,
    base_snapshot: FleetSnapshot,
    start_offset_min: int,
    requested_duration_min: int,
    restart_delay_min: int,
    outside_temp_c: float,
    guardian: SafetyGuardian,
    config: Optional[OptimizerConfig] = None,
    precomputed_snaps: Optional[List[FleetSnapshot]] = None,
) -> Optional[Candidate]:
    """Evaluate one proposed (offset, duration, restart_delay) combination across K=8 futures."""
    if config is None:
        config = OptimizerConfig()

    event_start = base_snapshot.timestamp_min + float(start_offset_min)
    action = CurtailmentAction(
        facility_id=facility.id,
        start_min=int(event_start),
        duration_min=int(requested_duration_min),
    )

    approving_durations: List[float] = []
    restart_slacks: List[float] = []
    verdict_ids: List[str] = []
    approvals_count = 0

    saved_history = {k: list(v) for k, v in guardian.debt_ledger.history.items()}

    if precomputed_snaps is None:
        snaps_dict = precompute_future_snapshots(
            facility, base_snapshot, [start_offset_min], outside_temp_c, config.k_futures
        )
        precomputed_snaps = snaps_dict[start_offset_min]

    for k in range(config.k_futures):
        guardian.debt_ledger.history = {k: list(v) for k, v in saved_history.items()}
        future_snap = precomputed_snaps[k]

        try:
            verdict = guardian.evaluate_request(action, future_snap)
            if verdict.is_approved:
                approvals_count += 1
                approving_durations.append(verdict.approved_duration_min)
                restart_slacks.append(verdict.restart_slack_min)
                verdict_ids.append(verdict.audit_id)
        except Exception:
            pass

    guardian.debt_ledger.history = {k: list(v) for k, v in saved_history.items()}
    approval_fraction = approvals_count / float(config.k_futures)

    if approvals_count < config.min_approvals_required:
        return None

    usable_dur = min(approving_durations)
    if usable_dur < 10.0:
        return None

    mean_slack = float(np.mean(restart_slacks)) if restart_slacks else 0.0

    delta_prof, kw_p10, kw_mean, pk_rebound = query_library_delta(
        facility_id=facility.id,
        outside_temp_c=outside_temp_c,
        load_fraction=base_snapshot.product_load_fraction,
        duration_min=int(usable_dur),
        restart_delay_min=restart_delay_min,
        horizon_min=360,
    )

    fatigue_debt = guardian.debt_ledger.compute_decaying_thermal_debt(facility, event_start)

    return Candidate(
        facility_id=facility.id,
        start_offset_min=start_offset_min,
        duration_min=int(usable_dur),
        restart_delay_min=restart_delay_min,
        approval_fraction=round(approval_fraction, 3),
        approved_min_p10=round(usable_dur, 1),
        expected_kw_cut_p10=round(kw_p10, 2),
        expected_kw_cut_mean=round(kw_mean, 2),
        verdict_id=verdict_ids[0] if verdict_ids else "",
        profile_delta_kw=delta_prof.tolist(),
        restart_slack_min=round(mean_slack, 1),
        fatigue_debt_kwh=round(fatigue_debt, 2),
    )


def generate_phase_robust_candidates(
    facilities: List[Facility],
    snapshots: Dict[str, FleetSnapshot],
    event_duration_min: int,
    outside_temp_c: float,
    guardian: SafetyGuardian,
    config: Optional[OptimizerConfig] = None,
) -> Dict[str, List[Candidate]]:
    """Generate all feasible phase-robust candidates per facility using precomputed snapshot rolling."""
    if config is None:
        config = OptimizerConfig()

    candidates_by_fac: Dict[str, List[Candidate]] = {}

    offsets = [0, 15]
    durations = [15, 30, 45, 60]
    durations = [d for d in durations if d <= event_duration_min or d == min(durations)]
    restart_delays = [0, 5, 10, 15, 20]

    for fac in facilities:
        snap = snapshots.get(fac.id)
        if not snap:
            continue

        fac_candidates: List[Candidate] = []

        # Fast pre-filter: if safe headroom is under 10m, guardian will refuse
        max_safe_lookup = lookup_max_curtail_min(fac.id, snap.temp_c, outside_temp_c)
        if max_safe_lookup < 10.0:
            candidates_by_fac[fac.id] = []
            continue

        # Precompute snapshots across offsets ONCE
        precomputed = precompute_future_snapshots(fac, snap, offsets, outside_temp_c, config.k_futures)

        for offset in offsets:
            snaps_k = precomputed[offset]
            for dur in durations:
                if offset + dur > event_duration_min + 30:
                    continue

                # Evaluate candidate once for duration across K futures
                cand_base = evaluate_candidate_across_futures(
                    facility=fac,
                    base_snapshot=snap,
                    start_offset_min=offset,
                    requested_duration_min=dur,
                    restart_delay_min=0,
                    outside_temp_c=outside_temp_c,
                    guardian=guardian,
                    config=config,
                    precomputed_snaps=snaps_k,
                )
                if cand_base is not None:
                    fac_candidates.append(cand_base)
                    # For feasible candidates, add staggered variations within restart_slack
                    max_slack = cand_base.restart_slack_min
                    for delay in restart_delays:
                        if delay > 0 and delay <= max_slack:
                            delta_prof, kw_p10, kw_mean, _ = query_library_delta(
                                facility_id=fac.id,
                                outside_temp_c=outside_temp_c,
                                load_fraction=snap.product_load_fraction,
                                duration_min=cand_base.duration_min,
                                restart_delay_min=delay,
                                horizon_min=360,
                            )
                            c_stag = copy.copy(cand_base)
                            c_stag.restart_delay_min = delay
                            c_stag.profile_delta_kw = delta_prof.tolist()
                            fac_candidates.append(c_stag)

        candidates_by_fac[fac.id] = fac_candidates

    return candidates_by_fac
