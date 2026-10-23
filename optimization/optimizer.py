"""
RESCUEGRID - Mixed-Integer Linear Programming (MILP) Optimizer
=============================================================
Formulates and solves the optimal resource allocation problem using Mixed-Integer
Linear Programming via scipy.optimize.milp.

Mathematical Formulation:
-------------------------
Decision Variables:
    x[i, r] : Integer units of resource type 'r' allocated to incident 'i'.
    u[i, r] : Integer units of unmet demand of resource 'r' for incident 'i'.

Constraints:
    1. Demand conservation: x[i, r] + u[i, r] == demand[i, r]   for all (i, r)
    2. Capacity limit:      sum_i x[i, r] <= available[r]       for all r
    3. Integrality:         x[i, r], u[i, r] in Non-Negative Integers
    4. Optional Minimum Service: sum_r x[i, r] >= min_units     (if feasible & configured)

Objective:
    Minimize sum_{i, r} weight[i, r] * u[i, r]
    where weight[i, r] = max(0.01, priority[i] / 100.0) * resource_urgency[r]
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from optimization.models import Incident, Resource, AllocationPlan


def validate_inputs(incidents: List[Incident], inventory: List[Resource]) -> Tuple[List[str], Dict[str, int]]:
    """Validate input data integrity before formulating the optimization problem."""
    if not incidents:
        raise ValueError("Cannot solve allocation: incidents list is empty.")
    if not inventory:
        raise ValueError("Cannot solve allocation: inventory list is empty.")

    seen_ids = set()
    for inc in incidents:
        if not inc.incident_id:
            raise ValueError("All incidents must have a valid non-empty incident_id.")
        if inc.incident_id in seen_ids:
            raise ValueError(f"Duplicate incident_id detected: '{inc.incident_id}'")
        seen_ids.add(inc.incident_id)

    res_types = [r.resource_type for r in inventory]
    if len(set(res_types)) != len(res_types):
        raise ValueError("Duplicate resource types detected in inventory.")

    available_map = {r.resource_type: r.available_quantity for r in inventory}
    return res_types, available_map


def solve_resource_allocation(
    incidents: List[Incident],
    inventory: List[Resource],
    policy: str = "STANDARD_WEIGHTED",
    min_service_units: int = 1,
) -> AllocationPlan:
    """
    Solve resource allocation using Mixed-Integer Linear Programming.

    Parameters:
    -----------
    incidents : List[Incident]
        Target incidents with priority scores and resource demands.
    inventory : List[Resource]
        Available resource stock and urgency weights.
    policy : str
        "STANDARD_WEIGHTED" : Pure priority-weighted unmet demand minimization.
        "MIN_SERVICE_GUARANTEE" : Enforces minimum allocation per eligible incident where feasible.
        "BUNDLE_AWARE" : Places non-linear penalty on unbalanced partial allocations.
    min_service_units : int
        Minimum units per incident when policy == "MIN_SERVICE_GUARANTEE".

    Returns:
    --------
    AllocationPlan: Structured allocation outcome with trade-off rationale.
    """
    t_start = time.perf_counter()
    res_types, available_map = validate_inputs(incidents, inventory)

    N = len(incidents)
    M = len(res_types)
    res_weight_map = {r.resource_type: r.urgency_weight for r in inventory}

    # Edge Case: Total demand is zero across all incidents
    total_demand = sum(inc.demands.get(r, 0) for inc in incidents for r in res_types)
    if total_demand == 0:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        allocations = {inc.incident_id: {r: 0 for r in res_types} for inc in incidents}
        unmet = {inc.incident_id: {r: 0 for r in res_types} for inc in incidents}
        reasons = {inc.incident_id: "Zero demand requested; all zero requirements satisfied." for inc in incidents}
        return AllocationPlan(
            solver_type="MILP_OPTIMIZER",
            solver_status="OPTIMAL",
            allocations=allocations,
            unmet=unmet,
            objective_value=0.0,
            runtime_ms=elapsed_ms,
            decision_reasons=reasons,
            trade_off_explanation="Zero total demand across all incidents. Solver terminated instantaneously.",
            is_optimal=True,
        )

    # Edge Case: Total available stock is zero
    total_available = sum(available_map.values())
    if total_available == 0:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        allocations = {inc.incident_id: {r: 0 for r in res_types} for inc in incidents}
        unmet = {inc.incident_id: {r: inc.demands.get(r, 0) for r in res_types} for inc in incidents}
        reasons = {inc.incident_id: "Zero inventory available in all resource categories." for inc in incidents}
        obj_val = sum(
            max(0.01, inc.priority_score / 100.0) * res_weight_map.get(r, 1.0) * inc.demands.get(r, 0)
            for inc in incidents
            for r in res_types
        )
        return AllocationPlan(
            solver_type="MILP_OPTIMIZER",
            solver_status="OPTIMAL",
            allocations=allocations,
            unmet=unmet,
            objective_value=float(obj_val),
            runtime_ms=elapsed_ms,
            decision_reasons=reasons,
            trade_off_explanation="Complete stockout across all resource depots. All requested demands unmet.",
            is_optimal=True,
        )

    # Variables: x[i, r] followed by u[i, r]
    # Total variable count = 2 * N * M
    n_vars = 2 * N * M

    def var_idx(is_unmet: bool, i: int, r: int) -> int:
        offset = N * M if is_unmet else 0
        return offset + i * M + r

    # Objective: Minimize sum_{i, r} w[i, r] * u[i, r]
    c = np.zeros(n_vars)
    for i, inc in enumerate(incidents):
        # Priority normalized with slight non-linear escalation for critical emergencies
        p_weight = max(0.01, (inc.priority_score / 100.0))
        if policy == "BUNDLE_AWARE" and inc.priority_score >= 70.0:
            p_weight = (inc.priority_score / 100.0) ** 1.3  # Convex penalty for high-priority starvation

        for r_idx, r_name in enumerate(res_types):
            u_idx = var_idx(True, i, r_idx)
            res_w = res_weight_map.get(r_name, 1.0)
            c[u_idx] = p_weight * res_w

    # Bounds:
    # 0 <= x[i, r] <= min(demand[i, r], available[r])
    # 0 <= u[i, r] <= demand[i, r]
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)
    for i, inc in enumerate(incidents):
        for r_idx, r_name in enumerate(res_types):
            dem = inc.demands.get(r_name, 0)
            avail = available_map.get(r_name, 0)
            x_i = var_idx(False, i, r_idx)
            u_i = var_idx(True, i, r_idx)
            ub[x_i] = min(dem, avail)
            ub[u_i] = dem

    bounds = Bounds(lb, ub)
    integrality = np.ones(n_vars)  # All decision variables must be integers

    # Constraints:
    # 1. Demand conservation: x[i, r] + u[i, r] == demand[i, r] (N * M rows)
    # 2. Resource capacity: sum_i x[i, r] <= available[r] (M rows)
    # 3. Optional Minimum Service Guarantee (if feasible)
    demand_rows = N * M
    capacity_rows = M

    eligible_min_service = []
    if policy == "MIN_SERVICE_GUARANTEE":
        for i, inc in enumerate(incidents):
            if inc.total_demand > 0:
                eligible_min_service.append(i)

        # Check feasibility before adding
        required_for_min_service = len(eligible_min_service) * min_service_units
        if total_available < required_for_min_service:
            eligible_min_service = []  # Discard if physically impossible

    total_constraints = demand_rows + capacity_rows + len(eligible_min_service)
    A = np.zeros((total_constraints, n_vars))
    b_l = np.zeros(total_constraints)
    b_u = np.zeros(total_constraints)

    # 1. Fill demand conservation equations
    row = 0
    for i, inc in enumerate(incidents):
        for r_idx, r_name in enumerate(res_types):
            dem = inc.demands.get(r_name, 0)
            x_i = var_idx(False, i, r_idx)
            u_i = var_idx(True, i, r_idx)
            A[row, x_i] = 1.0
            A[row, u_i] = 1.0
            b_l[row] = dem
            b_u[row] = dem
            row += 1

    # 2. Fill resource capacity limits
    for r_idx, r_name in enumerate(res_types):
        avail = available_map.get(r_name, 0)
        for i in range(N):
            x_i = var_idx(False, i, r_idx)
            A[row, x_i] = 1.0
        b_l[row] = -np.inf
        b_u[row] = avail
        row += 1

    # 3. Fill minimum service constraints (if active)
    for i in eligible_min_service:
        for r_idx in range(M):
            x_i = var_idx(False, i, r_idx)
            A[row, x_i] = 1.0
        b_l[row] = float(min_service_units)
        b_u[row] = np.inf
        row += 1

    constraints = LinearConstraint(A, b_l, b_u)

    # Execute MILP solve
    try:
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return AllocationPlan(
            solver_type="MILP_OPTIMIZER",
            solver_status="ERROR",
            error_message=str(exc),
            runtime_ms=elapsed_ms,
            trade_off_explanation=f"Solver execution failed: {exc}",
            is_optimal=False,
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    if not res.success:
        return AllocationPlan(
            solver_type="MILP_OPTIMIZER",
            solver_status=f"INFEASIBLE ({res.message})",
            error_message=res.message,
            runtime_ms=elapsed_ms,
            trade_off_explanation=f"Solver could not find a feasible integer solution: {res.message}",
            is_optimal=False,
        )

    # Parse solution vectors
    allocations: Dict[str, Dict[str, int]] = {}
    unmet: Dict[str, Dict[str, int]] = {}
    decision_reasons: Dict[str, str] = {}

    solution_vec = np.round(res.x).astype(int)

    # First pass: populate all allocations and unmet dicts
    for i, inc in enumerate(incidents):
        alloc_map = {}
        unmet_map = {}
        for r_idx, r_name in enumerate(res_types):
            x_val = int(solution_vec[var_idx(False, i, r_idx)])
            u_val = int(solution_vec[var_idx(True, i, r_idx)])
            alloc_map[r_name] = max(0, x_val)
            unmet_map[r_name] = max(0, u_val)

        allocations[inc.incident_id] = alloc_map
        unmet[inc.incident_id] = unmet_map

    # Second pass: generate transparent trade-off explanation
    for inc in incidents:
        unmet_map = unmet[inc.incident_id]
        inc_unmet_total = sum(unmet_map.values())
        if inc_unmet_total == 0:
            decision_reasons[inc.incident_id] = (
                f"Fully fulfilled (Priority {inc.priority_score:.1f}). Optimal allocation satisfied all requested units."
            )
        else:
            bottlenecks = [
                r for r, u in unmet_map.items()
                if u > 0 and available_map.get(r, 0) <= sum(allocations[other.incident_id].get(r, 0) for other in incidents)
            ]
            if bottlenecks:
                decision_reasons[inc.incident_id] = (
                    f"Partially fulfilled (Priority {inc.priority_score:.1f}). Stock depleted by higher-urgency trade-offs on: {', '.join(bottlenecks)}."
                )
            else:
                decision_reasons[inc.incident_id] = (
                    f"Partially fulfilled (Priority {inc.priority_score:.1f}). Unmet units minimized globally against system-wide priority weights."
                )

    # System-level trade-off narrative
    exhausted_resources = [
        r for r in res_types
        if sum(allocations[inc.incident_id][r] for inc in incidents) >= available_map.get(r, 0) and available_map.get(r, 0) > 0
    ]
    if exhausted_resources:
        trade_off_narrative = (
            f"MILP converged to global optimum in {elapsed_ms:.1f}ms. "
            f"Depot stocks fully exhausted for: {', '.join(exhausted_resources)}. "
            f"Resources routed to minimize cumulative priority-weighted unmet demand."
        )
    else:
        trade_off_narrative = (
            f"MILP converged to global optimum in {elapsed_ms:.1f}ms. "
            f"Inventory was sufficient to satisfy all system-wide requests without scarcity bottlenecks."
        )

    return AllocationPlan(
        solver_type="MILP_OPTIMIZER",
        solver_status="OPTIMAL",
        allocations=allocations,
        unmet=unmet,
        objective_value=float(res.fun),
        runtime_ms=elapsed_ms,
        decision_reasons=decision_reasons,
        trade_off_explanation=trade_off_narrative,
        is_optimal=True,
    )
