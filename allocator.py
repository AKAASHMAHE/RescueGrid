"""
RESCUEGRID - Resource Allocation Engine
---------------------------------------
Decision-support prototype. Uses synthetic demo inventory and needs.
It does NOT dispatch real emergency resources.

Run:
    python allocator.py
    python allocator.py --incidents data/demo_incidents.csv --inventory data/resource_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


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


def allocate_resources(
    incidents: List[dict], inventory: List[dict]
) -> Tuple[List[dict], List[dict]]:
    """
    Greedy allocation: process highest priority first and fulfill each
    resource need as far as remaining stock allows.

    This is a transparent baseline, not a globally optimal solver.
    Equal-priority incidents are ordered by incident_id for reproducibility.
    """
    validate_inputs(incidents, inventory)

    remaining: Dict[str, int] = {
        row["resource_type"]: int(row["available_quantity"])
        for row in inventory
    }

    ordered = sorted(
        incidents,
        key=lambda row: (-float(row["priority_score"]), row["incident_id"])
    )

    assignments = []
    shortages = []

    for incident in ordered:
        incident_id = incident["incident_id"]
        score = float(incident["priority_score"])
        incident_assignments = []
        incident_shortages = []

        for resource_type, need_column in RESOURCE_COLUMNS.items():
            requested = int(incident[need_column])
            available_before = remaining.get(resource_type, 0)
            allocated = min(requested, available_before)
            remaining[resource_type] = available_before - allocated
            unmet = requested - allocated

            incident_assignments.append(
                f"{resource_type}={allocated}/{requested}"
            )

            if unmet:
                incident_shortages.append(
                    f"{resource_type}: unmet {unmet}"
                )
                shortages.append({
                    "incident_id": incident_id,
                    "priority_score": score,
                    "resource_type": resource_type,
                    "requested": requested,
                    "allocated": allocated,
                    "unmet": unmet,
                })

        assignments.append({
            "incident_id": incident_id,
            "priority_score": score,
            "allocations": "; ".join(incident_assignments),
            "unmet_needs": "; ".join(incident_shortages) or "None",
            "decision_reason": (
                "Processed in descending priority order; allocations limited "
                "by remaining inventory."
            ),
            "review_status": "HUMAN_APPROVAL_REQUIRED",
            "data_status": "SYNTHETIC_DEMO_INPUT",
        })

    return assignments, shortages


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="RESCUEGRID allocation demo")
    parser.add_argument(
        "--incidents", type=Path, default=DATA_DIR / "demo_incidents.csv"
    )
    parser.add_argument(
        "--inventory", type=Path, default=DATA_DIR / "resource_inventory.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=DATA_DIR / "allocation_results.csv"
    )
    args = parser.parse_args()

    incidents = read_csv(args.incidents)
    inventory = read_csv(args.inventory)
    assignments, shortages = allocate_resources(incidents, inventory)

    write_csv(
        args.output,
        assignments,
        [
            "incident_id", "priority_score", "allocations", "unmet_needs",
            "decision_reason", "review_status", "data_status",
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
    print("=" * 40)
    print(f"Incidents considered: {len(incidents)}")
    print(f"Resource types: {len(inventory)}")
    print(f"Incidents with unmet needs: {len({x['incident_id'] for x in shortages})}")
    print(f"Allocation results: {args.output}")
    print(f"Unmet needs report: {shortage_path}")
    print("\nRecommendations (human approval required):")
    for row in assignments:
        print(
            f"- {row['incident_id']} | priority {row['priority_score']:.1f} | "
            f"{row['allocations']} | Unmet: {row['unmet_needs']}"
        )


if __name__ == "__main__":
    main()
