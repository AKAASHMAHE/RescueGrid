"""
RESCUEGRID - Evaluation Metrics & Solver Benchmarking
====================================================
Computes comprehensive statistical metrics for allocation plans, including fulfillment
rates, resource utilization, priority-weighted deficit, and benchmark deltas vs baseline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from optimization.models import Incident, Resource, AllocationPlan, EvaluationMetrics


def compute_evaluation_metrics(
    incidents: List[Incident],
    inventory: List[Resource],
    plan: AllocationPlan,
    baseline_plan: Optional[AllocationPlan] = None,
) -> EvaluationMetrics:
    """
    Calculate statistical metrics for an allocation plan with defensive zero-division checks.

    Parameters:
    -----------
    incidents : List[Incident]
        Incident specifications.
    inventory : List[Resource]
        Available resource inventory.
    plan : AllocationPlan
        Evaluated plan (either MILP or Greedy).
    baseline_plan : Optional[AllocationPlan]
        Baseline plan for comparative benchmarking (optional).

    Returns:
    --------
    EvaluationMetrics: Structured summary of fulfillment, utilization, and deltas.
    """
    res_types = [r.resource_type for r in inventory]
    available_map = {r.resource_type: r.available_quantity for r in inventory}
    res_weight_map = {r.resource_type: r.urgency_weight for r in inventory}

    total_requested = 0
    total_allocated = 0
    total_unmet = 0

    requested_by_res = {r: 0 for r in res_types}
    allocated_by_res = {r: 0 for r in res_types}
    unmet_by_res = {r: 0 for r in res_types}

    incidents_fully_satisfied = 0
    incidents_with_unmet = 0
    priority_weighted_unmet = 0.0

    for inc in incidents:
        inc_req = 0
        inc_alloc = 0
        inc_unmet = 0
        p_weight = max(0.01, (inc.priority_score / 100.0))

        for r in res_types:
            req = inc.demands.get(r, 0)
            alloc = plan.allocations.get(inc.incident_id, {}).get(r, 0)
            unm = plan.unmet.get(inc.incident_id, {}).get(r, 0)

            inc_req += req
            inc_alloc += alloc
            inc_unmet += unm

            requested_by_res[r] += req
            allocated_by_res[r] += alloc
            unmet_by_res[r] += unm

            w = p_weight * res_weight_map.get(r, 1.0)
            priority_weighted_unmet += w * unm

        total_requested += inc_req
        total_allocated += inc_alloc
        total_unmet += inc_unmet

        if inc_unmet == 0:
            incidents_fully_satisfied += 1
        else:
            incidents_with_unmet += 1

    # Safe fulfillment percentages
    overall_fulfillment_pct = (
        round((total_allocated / total_requested) * 100.0, 2)
        if total_requested > 0
        else 100.0
    )

    fulfillment_by_resource = {}
    for r in res_types:
        req = requested_by_res[r]
        alloc = allocated_by_res[r]
        fulfillment_by_resource[r] = (
            round((alloc / req) * 100.0, 2) if req > 0 else 100.0
        )

    # Resource utilization (allocated / available stock)
    resource_utilization = {}
    for r in res_types:
        avail = available_map[r]
        alloc = allocated_by_res[r]
        resource_utilization[r] = (
            round((alloc / avail) * 100.0, 2) if avail > 0 else (100.0 if alloc == 0 else 0.0)
        )

    # Comparative benchmarking vs baseline if supplied
    delta_vs_baseline = None
    if baseline_plan is not None:
        base_metrics = compute_evaluation_metrics(incidents, inventory, baseline_plan, baseline_plan=None)
        unmet_diff = base_metrics.total_unmet - total_unmet
        fulfillment_diff = overall_fulfillment_pct - base_metrics.overall_fulfillment_pct
        weighted_diff = base_metrics.priority_weighted_unmet - priority_weighted_unmet

        if base_metrics.priority_weighted_unmet > 0:
            weighted_reduction_pct = round((weighted_diff / base_metrics.priority_weighted_unmet) * 100.0, 2)
        else:
            weighted_reduction_pct = 0.0

        if base_metrics.total_unmet > 0:
            unmet_reduction_pct = round((unmet_diff / base_metrics.total_unmet) * 100.0, 2)
        else:
            unmet_reduction_pct = 0.0

        identical_allocations = (plan.allocations == baseline_plan.allocations)

        delta_vs_baseline = {
            "identical_allocations": identical_allocations,
            "fulfillment_diff_pct": round(fulfillment_diff, 2),
            "unmet_diff_units": unmet_diff,
            "unmet_reduction_pct": unmet_reduction_pct,
            "weighted_penalty_diff": round(weighted_diff, 2),
            "weighted_reduction_pct": weighted_reduction_pct,
            "baseline_runtime_ms": round(baseline_plan.runtime_ms, 2),
            "optimizer_runtime_ms": round(plan.runtime_ms, 2),
            "speed_ratio": (
                round(plan.runtime_ms / baseline_plan.runtime_ms, 2)
                if baseline_plan.runtime_ms > 0
                else 1.0
            ),
        }

    return EvaluationMetrics(
        total_requested=total_requested,
        total_allocated=total_allocated,
        total_unmet=total_unmet,
        overall_fulfillment_pct=overall_fulfillment_pct,
        fulfillment_by_resource=fulfillment_by_resource,
        priority_weighted_unmet=priority_weighted_unmet,
        incidents_count=len(incidents),
        incidents_fully_satisfied=incidents_fully_satisfied,
        incidents_with_unmet=incidents_with_unmet,
        resource_utilization=resource_utilization,
        runtime_ms=plan.runtime_ms,
        solver_status=plan.solver_status,
        delta_vs_baseline=delta_vs_baseline,
    )
