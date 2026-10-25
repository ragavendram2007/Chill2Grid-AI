"""solver.py - Rebound-Aware MILP Dispatch Solver using scipy.optimize.milp (HiGHS).

Formulation:
  Binary decision variables x[f, c] in {0, 1}: at most one candidate per facility.
  Continuous shortfall variable s >= 0: penalty for unmet target kW.
  Continuous rebound peak variable z >= 0: tracks fleet power maximum during recovery.

Constraints:
  1. Facility uniqueness: sum_c x[f, c] <= 1
  2. Shortfall definition: s + sum_{f, c} x[f, c] * cut_{f, c} >= P_target
  3. Rebound tracking: z - sum_{f, c} x[f, c] * delta_P_{f, c}(t) >= P_base(t)
  4. 'No New Peak' constraint: sum_{f, c} x[f, c] * delta_P_{f, c}(t) <= (1 + peak_allowance) * Peak_base - P_base(t)

Objective:
  min W_short * s + W_peak * z + W_debt * sum(fatigue_f * x) - W_slack * sum(slack_{f, c} * x)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from optimizer.types import Candidate, OptimizerConfig


def solve_rebound_milp(
    candidates_by_fac: Dict[str, List[Candidate]],
    target_kw: float,
    event_start_min: int,
    event_dur_min: int,
    baseline_power_curve: List[float],
    config: Optional[OptimizerConfig] = None,
) -> Tuple[List[Candidate], float, float, str]:
    """Solve the rebound-aware dispatch MILP.

    Returns:
        (selected_candidates, shortfall_kw, rebound_peak_kw, solver_status)
    """
    if config is None:
        config = OptimizerConfig()

    # Flatten candidates list
    all_cands: List[Candidate] = []
    fac_indices: Dict[str, List[int]] = {}

    idx = 0
    for fac_id, cands in candidates_by_fac.items():
        fac_indices[fac_id] = []
        for c in cands:
            all_cands.append(c)
            fac_indices[fac_id].append(idx)
            idx += 1

    n_cands = len(all_cands)
    if n_cands == 0:
        return [], float(target_kw), 0.0, "NO_FEASIBLE_CANDIDATES"

    # Time discretization for power profile constraints (sampled every 5 minutes)
    horizon_min = min(len(baseline_power_curve), event_start_min + event_dur_min + 180)
    time_steps = list(range(event_start_min, horizon_min, 5))
    n_time = len(time_steps)

    base_power = np.array([baseline_power_curve[m] for m in time_steps])
    peak_base = float(np.max(baseline_power_curve))
    max_allowed_power = (1.0 + config.peak_allowance) * peak_base

    # Decision variables: [x_0, ..., x_{n-1}, s, z]
    # Total vars: n_cands + 2
    n_vars = n_cands + 2
    idx_s = n_cands
    idx_z = n_cands + 1

    # Objective coefficients
    c_obj = np.zeros(n_vars)
    for i, cand in enumerate(all_cands):
        # Fatigue debt penalty and slack bonus
        c_obj[i] = config.w_debt * cand.fatigue_debt_kwh - config.w_slack * cand.restart_slack_min

    c_obj[idx_s] = config.w_short
    c_obj[idx_z] = config.w_peak

    # Variable bounds and integrality
    integrality = np.zeros(n_vars, dtype=int)
    integrality[:n_cands] = 1  # Binary x_i

    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    lb[idx_s] = 0.0
    ub[idx_s] = np.inf
    lb[idx_z] = 0.0
    ub[idx_z] = np.inf
    bounds = Bounds(lb, ub)

    # Constraints Matrix A
    # 1. At most one candidate per facility (len(fac_indices) rows)
    # 2. Shortfall constraint: sum(x_i * cut_i) + s >= target_kw (1 row)
    # 3. Peak tracking: z - sum(x_i * delta_P_i(t)) >= P_base(t) for t in recovery (n_recovery rows)
    # 4. No new peak: sum(x_i * delta_P_i(t)) <= max_allowed_power - P_base(t) (n_recovery rows)

    recovery_steps = [t for t in time_steps if t >= event_start_min + event_dur_min]
    n_rec = len(recovery_steps)

    rows: List[np.ndarray] = []
    lhs: List[float] = []
    rhs: List[float] = []

    # 1. Facility uniqueness constraints
    for fac_id, indices in fac_indices.items():
        if not indices:
            continue
        row = np.zeros(n_vars)
        for i in indices:
            row[i] = 1.0
        rows.append(row)
        lhs.append(0.0)
        rhs.append(1.0)

    # 2. Shortfall definition
    row_short = np.zeros(n_vars)
    for i, cand in enumerate(all_cands):
        row_short[i] = cand.expected_kw_cut_p10
    row_short[idx_s] = 1.0
    rows.append(row_short)
    lhs.append(float(target_kw))
    rhs.append(np.inf)

    # 3 & 4. Recovery peak tracking and no-new-peak
    for t in recovery_steps:
        m_rel = t - event_start_min
        base_p_t = baseline_power_curve[t] if t < len(baseline_power_curve) else baseline_power_curve[-1]

        # Tracking row: z - sum_i x_i * delta_i(t) >= base_p_t
        row_track = np.zeros(n_vars)
        row_track[idx_z] = 1.0
        for i, cand in enumerate(all_cands):
            # Delta at m_rel from profile
            if 0 <= m_rel < len(cand.profile_delta_kw):
                d_p = cand.profile_delta_kw[m_rel]
            else:
                d_p = 0.0
            row_track[i] = -d_p

        rows.append(row_track)
        lhs.append(base_p_t)
        rhs.append(np.inf)

        # No new peak row: sum_i x_i * delta_i(t) <= max_allowed_power - base_p_t
        row_peak = np.zeros(n_vars)
        for i, cand in enumerate(all_cands):
            if 0 <= m_rel < len(cand.profile_delta_kw):
                d_p = cand.profile_delta_kw[m_rel]
            else:
                d_p = 0.0
            row_peak[i] = d_p

        rows.append(row_peak)
        lhs.append(-np.inf)
        rhs.append(max(0.0, max_allowed_power - base_p_t))

    A = np.array(rows)
    constraints = LinearConstraint(A, lhs, rhs)

    # Solve MILP
    res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

    if res.success:
        x_sol = res.x[:n_cands]
        selected = [all_cands[i] for i, val in enumerate(x_sol) if val > 0.5]
        if len(selected) > 0 or target_kw <= 0:
            shortfall_val = float(res.x[idx_s])
            peak_val = float(res.x[idx_z])
            return selected, shortfall_val, peak_val, "OPTIMAL"

    # Fallback: Relax peak allowance if infeasible or empty under strict no-new-peak
    rhs_relaxed = np.array([max(r, (peak_base * 0.20)) if (r < np.inf and r > -np.inf) else r for r in rhs])
    constraints_rel = LinearConstraint(A, lhs, rhs_relaxed)
    res_rel = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints_rel)

    if res_rel.success:
        x_sol = res_rel.x[:n_cands]
        selected = [all_cands[i] for i, val in enumerate(x_sol) if val > 0.5]
        if len(selected) > 0 or target_kw <= 0:
            return selected, float(res_rel.x[idx_s]), float(res_rel.x[idx_z]), "RELAXED_OPTIMAL"

    # Second fallback: Greedy selection
    return _greedy_fallback(all_cands, target_kw, fac_indices)


def _greedy_fallback(
    all_cands: List[Candidate],
    target_kw: float,
    fac_indices: Dict[str, List[int]],
) -> Tuple[List[Candidate], float, float, str]:
    """Greedy fallback selection prioritizing kW cut per unit fatigue debt."""
    sorted_cands = sorted(
        all_cands,
        key=lambda c: c.expected_kw_cut_p10 / max(1.0, c.fatigue_debt_kwh + 1.0),
        reverse=True,
    )

    selected: List[Candidate] = []
    selected_facs = set()
    current_kw = 0.0

    for c in sorted_cands:
        if c.facility_id in selected_facs:
            continue
        selected.append(c)
        selected_facs.add(c.facility_id)
        current_kw += c.expected_kw_cut_p10
        if current_kw >= target_kw:
            break

    shortfall = max(0.0, target_kw - current_kw)
    return selected, shortfall, 0.0, "GREEDY_FALLBACK"
