"""
RESCUEGRID - Greedy Priority-First Allocation Baseline
=====================================================
Transparent greedy baseline algorithm used for benchmark comparison against the MILP solver.

Algorithm:
----------
1. Sort incidents in descending order of priority score (tie-break by incident_id).
2. For each incident, fulfill demands sequentially from available stock until stock is depleted.
3. Record allocations, unmet demands, and decision reasons.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any

from optimization.models import Incident, Resource, AllocationPlan
from optimization.optimizer import validate_inputs


def solve_greedy_baseline(
    incidents: List[Incident],
    inventory: List[Resource],
) -> AllocationPlan:
    """
    Execute transparent greedy priority-first allocation baseline.

    Parameters:
    -----------
    incidents : List[Incident]
        Target incidents.
    inventory : List[Resource]
        Available resource stock.

    Returns:
    --------
    AllocationPlan: Allocation outcome formatted identically to the optimizer plan.
    """
    t_start = time.perf_counter()
    res_types, available_map = validate_inputs(incidents, inventory)

    remaining_stock = {r.resource_type: r.available_quantity for r in inventory}
    res_weight_map = {r.resource_type: r.urgency_weight for r in inventory}

    # Sort incidents strictly by descending priority score, then incident_id
    ordered_incidents = sorted(
        incidents,
        key=lambda inc: (-inc.priority_score, inc.incident_id),
    )

    allocations: Dict[str, Dict[str, int]] = {}
    unmet: Dict[str, Dict[str, int]] = {}
    decision_reasons: Dict[str, str] = {}

    total_objective = 0.0

    for inc in ordered_incidents:
        inc_alloc = {}
        inc_unmet = {}
        inc_reasons = []

        for r_name in res_types:
            requested = inc.demands.get(r_name, 0)
            available = remaining_stock.get(r_name, 0)
            allocated = min(requested, available)
            remaining_stock[r_name] = available - allocated
            deficit = requested - allocated

            inc_alloc[r_name] = allocated
            inc_unmet[r_name] = deficit

            # Priority-weighted penalty accumulation
            w = max(0.01, (inc.priority_score / 100.0)) * res_weight_map.get(r_name, 1.0)
            total_objective += w * deficit

            if deficit > 0:
                inc_reasons.append(f"{r_name} stock exhausted")

        allocations[inc.incident_id] = inc_alloc
        unmet[inc.incident_id] = inc_unmet

        if sum(inc_unmet.values()) == 0:
            decision_reasons[inc.incident_id] = (
                f"Triage Rank #{ordered_incidents.index(inc) + 1} (Priority {inc.priority_score:.1f}). "
                "Fully allocated based on greedy priority queue."
            )
        else:
            decision_reasons[inc.incident_id] = (
                f"Triage Rank #{ordered_incidents.index(inc) + 1} (Priority {inc.priority_score:.1f}). "
                f"Partially allocated; shortages caused by depleted stock: {', '.join(inc_reasons)}."
            )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    exhausted = [r for r, qty in remaining_stock.items() if qty == 0 and available_map.get(r, 0) > 0]
    trade_off = (
        f"Greedy baseline served {len(incidents)} incidents strictly in priority sequence. "
        f"Exhausted stock on: {', '.join(exhausted) if exhausted else 'None'}."
    )

    return AllocationPlan(
        solver_type="GREEDY_BASELINE",
        solver_status="OPTIMAL" if sum(sum(u.values()) for u in unmet.values()) == 0 else "FEASIBLE",
        allocations=allocations,
        unmet=unmet,
        objective_value=float(total_objective),
        runtime_ms=elapsed_ms,
        decision_reasons=decision_reasons,
        trade_off_explanation=trade_off,
        is_optimal=False,
    )
