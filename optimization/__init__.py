"""
RESCUEGRID Optimization Engine
==============================
Mixed-Integer Linear Programming (MILP) solver and Greedy Allocation Baseline.
"""

from optimization.models import Incident, Resource, AllocationPlan, EvaluationMetrics, Scenario
from optimization.optimizer import solve_resource_allocation, validate_inputs
from optimization.baseline import solve_greedy_baseline
from optimization.metrics import compute_evaluation_metrics
from optimization.scenarios import get_all_scenarios

__all__ = [
    "Incident",
    "Resource",
    "AllocationPlan",
    "EvaluationMetrics",
    "Scenario",
    "solve_resource_allocation",
    "solve_greedy_baseline",
    "validate_inputs",
    "compute_evaluation_metrics",
    "get_all_scenarios",
]
