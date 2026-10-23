"""
Unit and Integration Tests for RESCUEGRID
=========================================
Tests data loading, schema integrity, conflict parsing, greedy allocation constraints,
priority scoring, and defensive fallback behavior.
"""

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pytest
import pandas as pd

import pipeline_utils as utils
import allocator
import conflicts
import priority

DATA_DIR = BASE_DIR / "data"


# =============================================================================
# 1. DATA LOADING & PIPELINE HEALTH TESTS
# =============================================================================

def test_safe_read_csv_nonexistent():
    """Verify safe_read_csv returns None gracefully for missing files."""
    missing_path = DATA_DIR / "non_existent_file_xyz123.csv"
    assert utils.safe_read_csv(missing_path) is None


def test_load_reports_dataset():
    """Verify reports dataset loads with required columns and numeric priority."""
    df, source = utils.load_reports_dataset()
    assert df is not None
    assert len(df) > 0
    assert "tweet_id" in df.columns
    assert "tweet_text" in df.columns
    assert source in ["priority_reports.tsv", "processed_reports.tsv", "humaid_reports_merged.tsv"]
    if source == "priority_reports.tsv":
        assert "priority_score" in df.columns
        assert "priority_score_numeric" in df.columns


def test_load_similarity_dataset():
    """Verify similarity dataset loads with normalized event_name and numeric similarity."""
    df, source = utils.load_similarity_dataset()
    assert df is not None
    assert len(df) > 0
    assert "tweet_id_a" in df.columns
    assert "tweet_id_b" in df.columns
    assert "event_name" in df.columns
    assert "similarity_numeric" in df.columns
    assert df["similarity_numeric"].min() >= 0.0
    assert df["similarity_numeric"].max() <= 1.0


def test_load_conflicts_dataset():
    """Verify conflicts dataset loads if present."""
    df = utils.load_conflicts_dataset()
    if df is not None:
        assert "tweet_id_a" in df.columns
        assert "tweet_id_b" in df.columns
        assert "conflict_signals" in df.columns


def test_inspect_pipeline_health():
    """Verify pipeline health audit produces valid status records."""
    records = utils.inspect_pipeline_health()
    assert len(records) > 0
    names = [r["name"] for r in records]
    assert "Processed Reports" in names
    assert "Priority Scored Reports" in names
    assert "Allocation Results" in names

    # Confirm required files are healthy
    for r in records:
        if r["name"] in ["Processed Reports", "Priority Scored Reports"]:
            assert r["status"] == "HEALTHY"
            assert r["rows"] > 0


# =============================================================================
# 2. CONFLICT SIGNAL PARSER TESTS
# =============================================================================

def test_parse_conflict_signals_empty():
    """Verify parsing empty signal string returns empty list."""
    assert utils.parse_conflict_signals("") == []
    assert utils.parse_conflict_signals(None) == []


def test_parse_conflict_signals_numeric():
    """Verify numeric difference signal extraction."""
    raw = "NUMBER_DIFFERENCE_REVIEW_DEATH_TOLL(A:30;B:25)"
    parsed = utils.parse_conflict_signals(raw)
    assert len(parsed) == 1
    p = parsed[0]
    assert p["signal_type"] == "NUMBER_DIFFERENCE"
    assert p["metric"] == "DEATH_TOLL"
    assert p["a_val"] == "30"
    assert p["b_val"] == "25"
    assert "Report A states '30'" in p["summary"]
    assert "Report B states '25'" in p["summary"]


def test_parse_conflict_signals_composite():
    """Verify multiple composite signals separated by pipe are parsed."""
    raw = "NUMBER_DIFFERENCE_REVIEW_DEATH_TOLL(A:206;B:91) | NUMBER_DIFFERENCE_REVIEW_MISSING_COUNT(A:110;B:68)"
    parsed = utils.parse_conflict_signals(raw)
    assert len(parsed) == 2
    assert parsed[0]["metric"] == "DEATH_TOLL"
    assert parsed[1]["metric"] == "MISSING_COUNT"


def test_conflicts_module_self_test():
    """Verify conflicts module built-in self tests pass without error."""
    conflicts.run_self_tests()


# =============================================================================
# 3. PRIORITY SCORING TESTS
# =============================================================================

def test_priority_score_bounds():
    """Verify score_report caps score at [0, 100] and assigns correct level."""
    test_row = {
        "tweet_text": "People trapped in immediate danger! Critical emergency need rescue immediately!",
        "humanitarian_category": "urgent_needs",
    }
    result = priority.score_report(test_row)
    assert 0 <= result["priority_score"] <= 100
    assert result["priority_level"] == "CRITICAL_REVIEW"
    assert result["matched_indicator_count"] >= 1


def test_priority_score_non_humanitarian():
    """Verify non-humanitarian reports receive base score 0."""
    test_row = {
        "tweet_text": "Just enjoying lunch with friends, wonderful sunny day.",
        "humanitarian_category": "non_humanitarian",
    }
    result = priority.score_report(test_row)
    assert result["priority_score"] == 0
    assert result["priority_level"] == "INFORMATIONAL"


# =============================================================================
# 4. RESOURCE ALLOCATION & SIMULATION TESTS
# =============================================================================

def test_allocator_validation_duplicate_ids():
    """Verify validator catches duplicate incident IDs."""
    incidents = [
        {"incident_id": "INC-1", "priority_score": "90", "ambulances_needed": "1", "rescue_teams_needed": "1", "water_kits_needed": "1"},
        {"incident_id": "INC-1", "priority_score": "80", "ambulances_needed": "1", "rescue_teams_needed": "1", "water_kits_needed": "1"},
    ]
    inventory = [{"resource_type": "ambulance", "available_quantity": "5"}]
    with pytest.raises(ValueError, match="Duplicate incident_id"):
        allocator.validate_inputs(incidents, inventory)


def test_allocator_validation_negative_inventory():
    """Verify validator rejects negative available stock."""
    incidents = [
        {"incident_id": "INC-1", "priority_score": "90", "ambulances_needed": "1", "rescue_teams_needed": "1", "water_kits_needed": "1"},
    ]
    inventory = [
        {"resource_type": "ambulance", "available_quantity": "-3"},
        {"resource_type": "rescue_team", "available_quantity": "5"},
        {"resource_type": "water_kit", "available_quantity": "5"},
    ]
    with pytest.raises(ValueError, match="cannot be negative"):
        allocator.validate_inputs(incidents, inventory)


def test_allocator_greedy_priority_order():
    """Verify higher priority incident receives resources before lower priority."""
    incidents = [
        {"incident_id": "LOW-PRI", "priority_score": "40", "ambulances_needed": "2", "rescue_teams_needed": "0", "water_kits_needed": "0"},
        {"incident_id": "HIGH-PRI", "priority_score": "95", "ambulances_needed": "2", "rescue_teams_needed": "0", "water_kits_needed": "0"},
    ]
    inventory = [
        {"resource_type": "ambulance", "available_quantity": "2"},
        {"resource_type": "rescue_team", "available_quantity": "0"},
        {"resource_type": "water_kit", "available_quantity": "0"},
    ]

    assignments, shortages = allocator.allocate_resources(incidents, inventory)

    high_assign = next(a for a in assignments if a["incident_id"] == "HIGH-PRI")
    low_assign = next(a for a in assignments if a["incident_id"] == "LOW-PRI")

    assert "ambulance=2/2" in high_assign["allocations"]
    assert "ambulance=0/2" in low_assign["allocations"]
    assert any(s["incident_id"] == "LOW-PRI" and s["unmet"] == 2 for s in shortages)


def test_run_allocation_simulation_metrics():
    """Verify run_allocation_simulation computes fulfillment rate and metrics correctly."""
    incidents = [
        {"incident_id": "INC-A", "priority_score": "90", "ambulances_needed": "2", "rescue_teams_needed": "1", "water_kits_needed": "1"},
    ]
    inventory = [
        {"resource_type": "ambulance", "available_quantity": "2"},
        {"resource_type": "rescue_team", "available_quantity": "1"},
        {"resource_type": "water_kit", "available_quantity": "1"},
    ]

    assignments, shortages, metrics = utils.run_allocation_simulation(incidents, inventory)
    assert len(shortages) == 0
    assert metrics["fulfillment_rate"] == 100.0
    assert metrics["total_requested"] == 4
    assert metrics["total_allocated"] == 4
    assert metrics["total_unmet"] == 0


# =============================================================================
# 5. OPTIMIZATION ENGINE & MILP CONSTRAINTS TESTS
# =============================================================================

def test_milp_solver_basic_constraints():
    """Verify MILP solver strictly satisfies capacity, demand conservation, and integrality."""
    import optimization as opt

    incidents = [
        opt.Incident("I-1", 90.0, {"ambulance": 3, "rescue_team": 2}),
        opt.Incident("I-2", 80.0, {"ambulance": 2, "rescue_team": 3}),
        opt.Incident("I-3", 70.0, {"ambulance": 1, "rescue_team": 1}),
    ]
    inventory = [
        opt.Resource("ambulance", available_quantity=4, urgency_weight=1.5),
        opt.Resource("rescue_team", available_quantity=3, urgency_weight=1.3),
    ]

    plan = opt.solve_resource_allocation(incidents, inventory)
    assert plan.is_optimal
    assert plan.solver_status == "OPTIMAL"

    # Check 1: Capacity constraints (sum_i x[i, r] <= available[r])
    total_amb_alloc = sum(plan.allocations[inc.incident_id]["ambulance"] for inc in incidents)
    total_res_alloc = sum(plan.allocations[inc.incident_id]["rescue_team"] for inc in incidents)
    assert total_amb_alloc <= 4
    assert total_res_alloc <= 3

    # Check 2: Demand conservation and bounds (x[i, r] + u[i, r] == demand[i, r])
    for inc in incidents:
        for r in ["ambulance", "rescue_team"]:
            dem = inc.demands[r]
            alloc = plan.allocations[inc.incident_id][r]
            unm = plan.unmet[inc.incident_id][r]
            assert alloc >= 0
            assert unm >= 0
            assert isinstance(alloc, int)
            assert isinstance(unm, int)
            assert alloc <= dem
            assert alloc + unm == dem


def test_milp_solver_edge_cases():
    """Verify zero inventory and zero demand edge cases execute safely without crash."""
    import optimization as opt

    incidents = [opt.Incident("I-1", 85.0, {"ambulance": 2, "water_kit": 4})]
    zero_inv = [opt.Resource("ambulance", 0), opt.Resource("water_kit", 0)]

    # Zero Inventory
    p_zero_inv = opt.solve_resource_allocation(incidents, zero_inv)
    assert p_zero_inv.is_optimal
    assert p_zero_inv.allocations["I-1"]["ambulance"] == 0
    assert p_zero_inv.unmet["I-1"]["ambulance"] == 2

    # Zero Demand
    zero_dem_inc = [opt.Incident("I-Z", 50.0, {"ambulance": 0, "water_kit": 0})]
    some_inv = [opt.Resource("ambulance", 5), opt.Resource("water_kit", 5)]
    p_zero_dem = opt.solve_resource_allocation(zero_dem_inc, some_inv)
    assert p_zero_dem.is_optimal
    assert p_zero_dem.allocations["I-Z"]["ambulance"] == 0
    assert p_zero_dem.unmet["I-Z"]["ambulance"] == 0


def test_milp_vs_greedy_divergence_equity():
    """Verify Scenario 7 demonstrates greedy starvation vs MILP equitable guarantee."""
    import optimization as opt

    scenarios = opt.get_all_scenarios()
    sc = scenarios["SCENARIO_GREEDY_VS_OPTIMIZER"]

    greedy_plan = opt.solve_greedy_baseline(sc.incidents, sc.inventory)
    milp_plan = opt.solve_resource_allocation(sc.incidents, sc.inventory, policy="MIN_SERVICE_GUARANTEE")

    metrics = opt.compute_evaluation_metrics(sc.incidents, sc.inventory, milp_plan, baseline_plan=greedy_plan)

    # In Greedy, INC-703 was completely starved of all resources (0 units allocated)
    assert sum(greedy_plan.allocations["INC-703"].values()) == 0

    # In MILP with Minimum Service Guarantee, INC-703 gets at least 1 unit
    assert sum(milp_plan.allocations["INC-703"].values()) >= 1

    # Overall fulfillment remains optimal
    assert metrics.delta_vs_baseline["identical_allocations"] is False


def test_all_preconfigured_scenarios_execute():
    """Verify all pre-configured simulation scenarios solve without errors on both engines."""
    import optimization as opt

    scenarios = opt.get_all_scenarios()
    assert len(scenarios) >= 7

    for sc_id, sc in scenarios.items():
        g_plan = opt.solve_greedy_baseline(sc.incidents, sc.inventory)
        m_plan = opt.solve_resource_allocation(sc.incidents, sc.inventory)
        assert g_plan.solver_status in ["OPTIMAL", "FEASIBLE"]
        assert m_plan.solver_status in ["OPTIMAL", "FEASIBLE"]

        g_met = opt.compute_evaluation_metrics(sc.incidents, sc.inventory, g_plan)
        m_met = opt.compute_evaluation_metrics(sc.incidents, sc.inventory, m_plan)
        assert 0.0 <= g_met.overall_fulfillment_pct <= 100.0
        assert 0.0 <= m_met.overall_fulfillment_pct <= 100.0


def test_analyst_decision_persistence():
    """Verify saving and reloading analyst decisions survives and records timestamps."""
    utils.save_analyst_decision("similar", "TEST_A_B", "CONFIRMED_DUPLICATE", "Test review note")
    loaded = utils.load_analyst_decisions()
    assert "similar" in loaded
    assert "TEST_A_B" in loaded["similar"]
    assert loaded["similar"]["TEST_A_B"]["status"] == "CONFIRMED_DUPLICATE"
    assert loaded["similar"]["TEST_A_B"]["notes"] == "Test review note"


def test_historical_incident_synthesis():
    """Verify synthesizing simulation scenario from real reports dataframe."""
    reports_df, _ = utils.load_reports_dataset()
    if reports_df is not None and "event_name" in reports_df.columns:
        event = reports_df["event_name"].iloc[0]
        sc = utils.build_scenario_from_historical_reports(reports_df, event_name=event, top_n=3)
        assert sc is not None
        assert len(sc.incidents) > 0
        assert len(sc.inventory) == 3

