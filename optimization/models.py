"""
RESCUEGRID - Optimization Data Structures & Models
==================================================
Defines type-safe data structures for incidents, resources, allocation plans,
evaluation metrics, and simulation scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


@dataclass
class Incident:
    """Represents an emergency incident requiring humanitarian resources."""
    incident_id: str
    priority_score: float  # 0.0 to 100.0
    demands: Dict[str, int]  # e.g. {"ambulance": 2, "rescue_team": 1, "water_kit": 3}
    category: str = "urgent_needs"
    event_name: str = "generic_disaster"
    coordinates: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (x, y, z) in 3D grid
    evidence_summary: str = ""
    verified: bool = False

    def __post_init__(self):
        self.priority_score = float(self.priority_score)
        if not (0.0 <= self.priority_score <= 100.0):
            raise ValueError(f"priority_score must be in [0, 100], got {self.priority_score}")
        for r, qty in self.demands.items():
            if int(qty) < 0:
                raise ValueError(f"Demand for {r} cannot be negative ({qty}) for {self.incident_id}")
            self.demands[r] = int(qty)

    @property
    def total_demand(self) -> int:
        return sum(self.demands.values())

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "incident_id": self.incident_id,
            "priority_score": self.priority_score,
            "category": self.category,
            "event_name": self.event_name,
            "coordinates": self.coordinates,
            "evidence_summary": self.evidence_summary,
            "verified": self.verified,
        }
        for r, qty in self.demands.items():
            res[f"{r}_needed"] = qty
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any], resource_types: List[str]) -> Incident:
        demands = {}
        for r in resource_types:
            key_needed = f"{r}s_needed" if f"{r}s_needed" in data else f"{r}_needed"
            if key_needed in data:
                demands[r] = int(data[key_needed])
            elif r in data:
                demands[r] = int(data[r])
            else:
                demands[r] = 0

        # Optional coordinates
        coords = data.get("coordinates", (0.0, 0.0, 0.0))
        if isinstance(coords, list):
            coords = tuple(coords)

        return cls(
            incident_id=str(data["incident_id"]).strip(),
            priority_score=float(data.get("priority_score", 50.0)),
            demands=demands,
            category=str(data.get("category", "urgent_needs")),
            event_name=str(data.get("event_name", "historical_event")),
            coordinates=coords,
            evidence_summary=str(data.get("evidence_summary", "")),
            verified=bool(data.get("verified", False)),
        )


@dataclass
class Resource:
    """Represents an emergency resource inventory category."""
    resource_type: str
    available_quantity: int
    urgency_weight: float = 1.0  # Medical/Life-saving resources have higher priority
    unit_cost: float = 1.0
    color_hex: str = "#00D5E8"
    icon: str = ""

    def __post_init__(self):
        self.available_quantity = int(self.available_quantity)
        if self.available_quantity < 0:
            raise ValueError(f"available_quantity cannot be negative ({self.available_quantity}) for {self.resource_type}")
        self.urgency_weight = float(self.urgency_weight)
        if self.urgency_weight <= 0:
            raise ValueError(f"urgency_weight must be positive for {self.resource_type}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type,
            "available_quantity": self.available_quantity,
            "urgency_weight": self.urgency_weight,
            "unit_cost": self.unit_cost,
            "color_hex": self.color_hex,
            "icon": self.icon,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Resource:
        rtype = str(data["resource_type"]).strip()
        defaults = {
            "ambulance": {"urgency_weight": 1.5, "color_hex": "#F04452", "icon": ""},
            "rescue_team": {"urgency_weight": 1.3, "color_hex": "#F5A623", "icon": ""},
            "water_kit": {"urgency_weight": 1.0, "color_hex": "#00D5E8", "icon": ""},
            "medical_kit": {"urgency_weight": 1.4, "color_hex": "#E056FD", "icon": ""},
            "food_ration": {"urgency_weight": 0.9, "color_hex": "#2ECC71", "icon": ""},
            "shelter_pack": {"urgency_weight": 1.1, "color_hex": "#E67E22", "icon": ""},
            "power_generator": {"urgency_weight": 1.2, "color_hex": "#F1C40F", "icon": ""},
            "hazmat_unit": {"urgency_weight": 1.6, "color_hex": "#9B59B6", "icon": ""},
            "communication_node": {"urgency_weight": 1.1, "color_hex": "#3498DB", "icon": ""},
            "boat_rescue": {"urgency_weight": 1.4, "color_hex": "#1ABC9C", "icon": ""},
        }
        cfg = defaults.get(rtype, {"urgency_weight": 1.0, "color_hex": "#2ECC71", "icon": ""})
        return cls(
            resource_type=rtype,
            available_quantity=int(data["available_quantity"]),
            urgency_weight=float(data.get("urgency_weight", cfg["urgency_weight"])),
            unit_cost=float(data.get("unit_cost", 1.0)),
            color_hex=str(data.get("color_hex", cfg["color_hex"])),
            icon=str(data.get("icon", cfg["icon"])),
        )


@dataclass
class AllocationPlan:
    """Contains the complete allocation solution from an algorithm."""
    solver_type: str  # "MILP_OPTIMIZER" or "GREEDY_BASELINE"
    solver_status: str  # "OPTIMAL", "FEASIBLE", "INFEASIBLE", "ERROR"
    allocations: Dict[str, Dict[str, int]] = field(default_factory=dict)  # inc_id -> {res: qty}
    unmet: Dict[str, Dict[str, int]] = field(default_factory=dict)  # inc_id -> {res: qty}
    objective_value: float = 0.0
    runtime_ms: float = 0.0
    decision_reasons: Dict[str, str] = field(default_factory=dict)
    trade_off_explanation: str = ""
    is_optimal: bool = False
    error_message: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Export tabular summary for UI display and auditing."""
        rows = []
        for inc_id, alloc_map in self.allocations.items():
            unmet_map = self.unmet.get(inc_id, {})
            alloc_str = "; ".join(f"{r}={qty}" for r, qty in alloc_map.items())
            unmet_str = "; ".join(f"{r}={qty}" for r, qty in unmet_map.items() if qty > 0) or "None"
            rows.append({
                "incident_id": inc_id,
                "allocations": alloc_str,
                "unmet_needs": unmet_str,
                "decision_reason": self.decision_reasons.get(inc_id, ""),
                "solver_type": self.solver_type,
                "status": self.solver_status,
            })
        return pd.DataFrame(rows)


@dataclass
class EvaluationMetrics:
    """Statistical summary evaluating an allocation outcome."""
    total_requested: int
    total_allocated: int
    total_unmet: int
    overall_fulfillment_pct: float
    fulfillment_by_resource: Dict[str, float]
    priority_weighted_unmet: float
    incidents_count: int
    incidents_fully_satisfied: int
    incidents_with_unmet: int
    resource_utilization: Dict[str, float]
    runtime_ms: float
    solver_status: str
    is_zero_demand: bool = False
    delta_vs_baseline: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requested": self.total_requested,
            "total_allocated": self.total_allocated,
            "total_unmet": self.total_unmet,
            "overall_fulfillment_pct": self.overall_fulfillment_pct,
            "fulfillment_by_resource": self.fulfillment_by_resource,
            "priority_weighted_unmet": round(self.priority_weighted_unmet, 2),
            "incidents_count": self.incidents_count,
            "incidents_fully_satisfied": self.incidents_fully_satisfied,
            "incidents_with_unmet": self.incidents_with_unmet,
            "resource_utilization": self.resource_utilization,
            "runtime_ms": round(self.runtime_ms, 2),
            "solver_status": self.solver_status,
            "is_zero_demand": self.is_zero_demand,
            "delta_vs_baseline": self.delta_vs_baseline,
        }


@dataclass
class Scenario:
    """A self-contained, reproducible simulation scenario."""
    scenario_id: str
    name: str
    description: str
    incidents: List[Incident]
    inventory: List[Resource]
    notes: str = ""
