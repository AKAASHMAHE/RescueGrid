"""
RESCUEGRID - Resource Allocation Engine
---------------------------------------
Emergency Resource Optimization and Decision-Support Engine.
Supports both Mixed-Integer Linear Programming (MILP) solver and Greedy Baseline.

Decision-support prototype. Uses synthetic demo inventory and needs.
It does NOT dispatch real emergency resources.

Run:
    python allocator.py --solver milp
    python allocator.py --solver greedy
    python allocator.py --incidents data/demo_incidents.csv --inventory data/resource_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any

from optimization.models import Incident, Resource, AllocationPlan
from optimization.optimizer import solve_resource_allocation
from optimization.baseline import solve_greedy_baseline
from optimization.metrics import compute_evaluation_metrics


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

REQUIRED_INCIDENT_COLUMNS = {
    "incident_id", "priority_score", "ambulances_needed",
    "rescue_teams_needed", "water_kits_needed"
}
REQUIRED_INVENTORY_COLUMNS = {
    "resource_type", "available_quantity"
}

RESOURCE_COLUMNS = {
    "ambulance": "ambulances_needed",
    "rescue_team": "rescue_teams_needed",
    "water_kit": "water_kits_needed",
}


def read_csv(path: Path) -> List[dict]:
    """Read a CSV file as a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def validate_inputs(incidents: List[dict], inventory: List[dict]) -> None:
    """Validate schemas and numeric values before allocation."""
    if not incidents:
        raise ValueError("Incident file contains no incident rows.")
    if not inventory:
        raise ValueError("Inventory file contains no resource rows.")

    incident_columns = set(incidents[0])
    inventory_columns = set(inventory[0])

    missing_incident = REQUIRED_INCIDENT_COLUMNS - incident_columns
    missing_inventory = REQUIRED_INVENTORY_COLUMNS - inventory_columns

    if missing_incident:
        raise ValueError(
            f"Incident file is missing columns: {sorted(missing_incident)}"
        )
    if missing_inventory:
        raise ValueError(
            f"Inventory file is missing columns: {sorted(missing_inventory)}"
        )

    seen_ids = set()
    for row in incidents:
        incident_id = row["incident_id"].strip()
        if not incident_id:
            raise ValueError("Every incident must have a non-empty incident_id.")
        if incident_id in seen_ids:
            raise ValueError(f"Duplicate incident_id: {incident_id}")
        seen_ids.add(incident_id)

        score = float(row["priority_score"])
        if not 0 <= score <= 100:
            raise ValueError(
                f"priority_score must be between 0 and 100: {incident_id}"
            )

        for column in RESOURCE_COLUMNS.values():
            amount = int(row[column])
            if amount < 0:
                raise ValueError(
                    f"{column} cannot be negative for {incident_id}"
                )

    seen_resources = set()
    for row in inventory:
        resource = row["resource_type"].strip()
        if resource not in RESOURCE_COLUMNS:
            raise ValueError(
                f"Unsupported resource_type '{resource}'. "
                f"Supported types: {sorted(RESOURCE_COLUMNS)}"
            )
        if resource in seen_resources:
            raise ValueError(f"Duplicate inventory row for resource: {resource}")
        seen_resources.add(resource)

        quantity = int(row["available_quantity"])
        if quantity < 0:
            raise ValueError(f"Inventory quantity cannot be negative: {resource}")


def dicts_to_objects(incidents: List[dict], inventory: List[dict]) -> Tuple[List[Incident], List[Resource]]:
    """Convert raw input dictionaries to optimization domain objects."""
    res_objs = [Resource.from_dict(row) for row in inventory]
    res_names = [r.resource_type for r in res_objs]
    inc_objs = [Incident.from_dict(row, res_names) for row in incidents]
    return inc_objs, res_objs


def plan_to_legacy_formats(
    incidents: List[Incident],
    inventory: List[Resource],
    plan: AllocationPlan,
) -> Tuple[List[dict], List[dict]]:
    """Convert AllocationPlan to legacy assignments and shortages dictionaries."""
    assignments = []
    shortages = []

    res_names = [r.resource_type for r in inventory]

    for inc in incidents:
        inc_id = inc.incident_id
        score = inc.priority_score
        alloc_map = plan.allocations.get(inc_id, {})
        unmet_map = plan.unmet.get(inc_id, {})

        incident_assignments = []
        incident_shortages = []

        for r_name in res_names:
            requested = inc.demands.get(r_name, 0)
            allocated = alloc_map.get(r_name, 0)
            unmet = unmet_map.get(r_name, 0)

            incident_assignments.append(f"{r_name}={allocated}/{requested}")

            if unmet > 0:
                incident_shortages.append(f"{r_name}: unmet {unmet}")
                shortages.append({
                    "incident_id": inc_id,
                    "priority_score": score,
                    "resource_type": r_name,
                    "requested": requested,
                    "allocated": allocated,
                    "unmet": unmet,
                })

        assignments.append({
            "incident_id": inc_id,
            "priority_score": score,
            "allocations": "; ".join(incident_assignments),
            "unmet_needs": "; ".join(incident_shortages) or "None",
            "decision_reason": plan.decision_reasons.get(inc_id, plan.trade_off_explanation),
            "review_status": "HUMAN_APPROVAL_REQUIRED",
            "data_status": "SYNTHETIC_DEMO_INPUT",
            "solver_type": plan.solver_type,
        })

    return assignments, shortages


def allocate_resources(
    incidents: List[dict], inventory: List[dict], solver: str = "milp"
) -> Tuple[List[dict], List[dict]]:
    """
    Solve resource allocation using either MILP optimizer or greedy baseline.
    Maintains backward compatibility with teammate's function signature.
    """
    validate_inputs(incidents, inventory)
    inc_objs, res_objs = dicts_to_objects(incidents, inventory)

    if solver.lower() == "greedy":
        plan = solve_greedy_baseline(inc_objs, res_objs)
    else:
        plan = solve_resource_allocation(inc_objs, res_objs)

    return plan_to_legacy_formats(inc_objs, res_objs, plan)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="RESCUEGRID allocation engine")
    parser.add_argument(
        "--incidents", type=Path, default=DATA_DIR / "demo_incidents.csv"
    )
    parser.add_argument(
        "--inventory", type=Path, default=DATA_DIR / "resource_inventory.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "allocation_results.csv"
    )
    parser.add_argument(
        "--solver", type=str, choices=["milp", "greedy"], default="milp",
        help="Allocation algorithm: 'milp' (Mixed-Integer Linear Programming) or 'greedy' (Priority-First Baseline)"
    )
    args = parser.parse_args()

    incidents = read_csv(args.incidents)
    inventory = read_csv(args.inventory)
    assignments, shortages = allocate_resources(incidents, inventory, solver=args.solver)

    write_csv(
        args.output,
        assignments,
        [
            "incident_id", "priority_score", "allocations", "unmet_needs",
            "decision_reason", "review_status", "data_status", "solver_type",
        ],
    )

    shortage_path = args.output.with_name("unmet_needs.csv")
    write_csv(
        shortage_path,
        shortages,
        [
            "incident_id", "priority_score", "resource_type",
            "requested", "allocated", "unmet",
        ],
    )

    print("\nRESCUEGRID allocation summary")
    print("=" * 45)
    print(f"Solver Engine: {args.solver.upper()}")
    print(f"Incidents considered: {len(incidents)}")
    print(f"Resource types: {len(inventory)}")
    print(f"Incidents with unmet needs: {len({x['incident_id'] for x in shortages})}")
    print(f"Allocation results: {args.output}")
    print(f"Unmet needs report: {shortage_path}")
    print("\nRecommendations (human approval required):")
    for row in assignments:
        print(
            f"- {row['incident_id']} | priority {float(row['priority_score']):.1f} | "
            f"{row['allocations']} | Unmet: {row['unmet_needs']}"
        )


if __name__ == "__main__":
    main()
