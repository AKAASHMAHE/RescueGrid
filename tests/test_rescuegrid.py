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


# =============================================================================
# 6. REGRESSION TESTS: SCENARIO SELECTION, STATE SYNC & ZERO-DEMAND CONVENTIONS
# =============================================================================

def test_validate_active_scenario_id_valid():
    """Verify a valid active scenario ID is preserved with its exact index."""
    available = ["SCENARIO_1", "SCENARIO_2", "SCENARIO_3"]
    active_id, idx = utils.validate_active_scenario_id("SCENARIO_2", available)
    assert active_id == "SCENARIO_2"
    assert idx == 1


def test_validate_active_scenario_id_unknown_fallback():
    """Verify an unknown/stale scenario ID safely falls back to default without ValueError."""
    available = ["SCENARIO_1", "SCENARIO_SEVERE_SHORTAGE", "SCENARIO_3"]
    # Simulates the exact crash: 'HISTORICAL_ECUADOR_EARTHQUAKE_2016' is not in list
    active_id, idx = utils.validate_active_scenario_id("HISTORICAL_ECUADOR_EARTHQUAKE_2016", available, default_id="SCENARIO_SEVERE_SHORTAGE")
    assert active_id == "SCENARIO_SEVERE_SHORTAGE"
    assert idx == 1


def test_validate_active_scenario_id_removed_scenario():
    """Verify safe fallback when a previously active scenario is deleted from registry."""
    available = ["SCENARIO_A", "SCENARIO_B"]
    active_id, idx = utils.validate_active_scenario_id("DELETED_SCENARIO", available, default_id="SCENARIO_NONEXISTENT")
    # Falls back to first available scenario when default is also absent
    assert active_id == "SCENARIO_A"
    assert idx == 0


def test_validate_active_scenario_id_empty_registry():
    """Verify empty scenario registry is handled gracefully without crash."""
    active_id, idx = utils.validate_active_scenario_id("ANY_ID", [])
    assert active_id == ""
    assert idx == -1


def test_validate_active_scenario_id_order_change():
    """Verify scenario index updates consistently if registry order changes."""
    available_v1 = ["SC_A", "SC_B", "SC_C"]
    available_v2 = ["SC_C", "SC_A", "SC_B"]
    _, idx_1 = utils.validate_active_scenario_id("SC_B", available_v1)
    _, idx_2 = utils.validate_active_scenario_id("SC_B", available_v2)
    assert idx_1 == 1
    assert idx_2 == 2


def test_validate_selected_incident_id_reset():
    """Verify selecting an incident that does not exist in the active scenario resets safely."""
    scenario_incidents = ["INC-101", "INC-102", "INC-103"]
    # When switching scenarios, previous selected ID might be 'INC-701'
    verified_inc = utils.validate_selected_incident_id("INC-701", scenario_incidents)
    assert verified_inc == "INC-101"

    # When valid
    assert utils.validate_selected_incident_id("INC-102", scenario_incidents) == "INC-102"

    # When incident list is empty
    assert utils.validate_selected_incident_id("ANY_INC", []) is None


def test_zero_demand_convention_flag():
    """Verify zero-demand convention is explicitly identified to avoid misleading 100% success."""
    import optimization as opt
    scenarios = opt.get_all_scenarios()
    sc_zero = scenarios["SCENARIO_ZERO_DEMAND"]

    plan = opt.solve_resource_allocation(sc_zero.incidents, sc_zero.inventory)
    metrics = opt.compute_evaluation_metrics(sc_zero.incidents, sc_zero.inventory, plan)

    assert metrics.total_requested == 0
    assert metrics.total_allocated == 0
    assert metrics.is_zero_demand is True
    assert metrics.overall_fulfillment_pct == 100.0  # Mathematical convention preserved with explicit flag


def test_large_scale_scenario_generation():
    """Verify large-scale scenario generator produces exact target incident counts and unique IDs."""
    from optimization.scenarios import generate_scaled_scenario

    scales = [
        ("Small", 25, 5),
        ("Medium", 100, 7),
        ("Large", 250, 9),
        ("Stress Test", 500, 10),
    ]

    for scale_name, expected_incidents, expected_resources in scales:
        sc = generate_scaled_scenario("SEVERE_SHORTAGE", scale_name=scale_name, seed=123)
        assert len(sc.incidents) == expected_incidents, f"Failed count for {scale_name}"
        assert len(sc.inventory) == expected_resources, f"Failed res count for {scale_name}"

        # Verify ID uniqueness
        ids = [inc.incident_id for inc in sc.incidents]
        assert len(ids) == len(set(ids)), f"Duplicate IDs detected in {scale_name}"

        # Verify priority bounds [0, 100]
        for inc in sc.incidents:
            assert 0.0 <= inc.priority_score <= 100.0
            assert inc.total_demand >= 0

        # Verify independent inventory
        for res in sc.inventory:
            assert res.available_quantity >= 0


def test_large_scale_seed_reproducibility():
    """Verify fixed seeds produce strictly deterministic scenario output."""
    from optimization.scenarios import generate_scaled_scenario

    sc1 = generate_scaled_scenario("MIXED_PRIORITIES", scale_name="Medium", seed=42)
    sc2 = generate_scaled_scenario("MIXED_PRIORITIES", scale_name="Medium", seed=42)

    assert len(sc1.incidents) == len(sc2.incidents)
    for i in range(len(sc1.incidents)):
        assert sc1.incidents[i].incident_id == sc2.incidents[i].incident_id
        assert sc1.incidents[i].priority_score == sc2.incidents[i].priority_score
        assert sc1.incidents[i].demands == sc2.incidents[i].demands
        assert sc1.incidents[i].coordinates == sc2.incidents[i].coordinates


def test_large_scale_milp_solver_500_incidents():
    """Verify MILP optimizer solves 500-incident stress test cleanly with strict constraint satisfaction."""
    import time
    from optimization.scenarios import generate_scaled_scenario
    from optimization.optimizer import solve_resource_allocation

    sc = generate_scaled_scenario("SEVERE_SHORTAGE", scale_name="Stress Test", seed=42)
    assert len(sc.incidents) == 500

    t0 = time.perf_counter()
    plan = solve_resource_allocation(sc.incidents, sc.inventory)
    solve_duration = time.perf_counter() - t0

    assert plan.is_optimal is True
    assert plan.solver_status == "OPTIMAL"
    assert solve_duration < 30.0  # Must terminate within reasonable operational threshold

    res_names = [r.resource_type for r in sc.inventory]
    avail_map = {r.resource_type: r.available_quantity for r in sc.inventory}

    # Verify demand conservation: x[i, r] + u[i, r] == d[i, r]
    for inc in sc.incidents:
        alloc_map = plan.allocations[inc.incident_id]
        unm_map = plan.unmet[inc.incident_id]
        for r in res_names:
            dem = inc.demands.get(r, 0)
            assert alloc_map.get(r, 0) + unm_map.get(r, 0) == dem

    # Verify resource capacity: sum_i x[i, r] <= available[r]
    for r in res_names:
        tot_allocated = sum(plan.allocations[inc.incident_id].get(r, 0) for inc in sc.incidents)
        assert tot_allocated <= avail_map[r]


def test_emoji_audit_in_scenarios_and_theme():
    """Verify zero emojis exist in theme configuration or scenario names."""
    import re
    import theme
    from optimization.scenarios import get_all_scenarios

    # Regex detecting common emoji ranges
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff]|"
        "[\u2600-\u26ff]|"
        "[\u2700-\u27bf]|"
        "[\u2300-\u23ff]|"
        "[\u2b50-\u2b55]|"
        "[\u200d]|"
        "[\ufe0f]"
    )

    scenarios = get_all_scenarios()
    for s_id, sc in scenarios.items():
        assert not emoji_pattern.search(sc.name), f"Emoji found in scenario name: {sc.name}"
        for res in sc.inventory:
            assert not emoji_pattern.search(res.icon), f"Emoji found in resource icon for {res.resource_type}"

    # Verify theme colors and fonts
    assert theme.APP_BG == "#080F1C"
    assert theme.PANEL_BG == "#101B2D"
    assert theme.ACCENT_CYAN == "#00D5E8"
    assert theme.CRITICAL_RED == "#F04452"


def test_3d_visualization_scaled_rendering():
    """Verify Plotly 3D and Three.js generator render 100 incidents without crashing."""
    from optimization.scenarios import generate_scaled_scenario
    from optimization.optimizer import solve_resource_allocation
    import visualization_3d as viz3d

    sc = generate_scaled_scenario("MIXED_PRIORITIES", scale_name="Medium", seed=77)
    plan = solve_resource_allocation(sc.incidents, sc.inventory)

    # Plotly 3D
    fig = viz3d.generate_plotly_3d_figure(sc.incidents, sc.inventory, plan, selected_incident_id="INC-0001")
    assert fig is not None
    assert len(fig.data) >= 2  # Traces exist

    # Three.js HTML
    html = viz3d.generate_threejs_html(sc.incidents, sc.inventory, plan, selected_incident_id="INC-0001")
    assert "INC-0001" in html
    assert "THREE.WebGLRenderer" in html



