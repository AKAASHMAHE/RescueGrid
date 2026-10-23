"""
RESCUEGRID — 3D Emergency Command Center & Optimization Platform
===============================================================
Tactical emergency operations intelligence and constrained resource allocation platform.
Integrates Mixed-Integer Linear Programming (MILP), Greedy Baseline comparisons,
interactive 3D Digital-Twin situational awareness, and historical evidence analysis.
Strictly adheres to the professional emergency command center design system with zero emojis.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Centralized RESCUEGRID engine imports
import theme
import pipeline_utils as utils
import optimization as opt
import visualization_3d as viz3d

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & DESIGN SYSTEM CSS (Zero Emoji)
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="RESCUEGRID | Emergency Command Center",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(theme.TACTICAL_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DEFENSIVE DATA LOADERS & AUTHORITATIVE SESSION REGISTRY
# -----------------------------------------------------------------------------

reports_df, reports_source = utils.load_reports_dataset()
pairs_df, pairs_source = utils.load_similarity_dataset()
conflicts_df = utils.load_conflicts_dataset()

# Scale Selection State
if "simulation_scale" not in st.session_state:
    st.session_state["simulation_scale"] = "Small"

# Authoritative Persistent Scenario Registry in Session State
if "scenario_registry" not in st.session_state or not st.session_state["scenario_registry"]:
    st.session_state["scenario_registry"] = opt.get_all_scenarios(scale=st.session_state["simulation_scale"])

sc_registry = st.session_state["scenario_registry"]
available_sc_ids = list(sc_registry.keys())

# Safe Active Scenario Validation
saved_sc_id = st.session_state.get("active_scenario_id", "SCENARIO_SEVERE_SHORTAGE")
verified_sc_id, sc_index = utils.validate_active_scenario_id(
    saved_sc_id, available_sc_ids, default_id="SCENARIO_SEVERE_SHORTAGE"
)
st.session_state["active_scenario_id"] = verified_sc_id

# Safe Active Incident Validation
if verified_sc_id in sc_registry:
    active_scenario = sc_registry[verified_sc_id]
    avail_inc_ids = [inc.incident_id for inc in active_scenario.incidents]
    st.session_state["selected_incident_id"] = utils.validate_selected_incident_id(
        st.session_state.get("selected_incident_id"), avail_inc_ids
    )

if "custom_inventory_stock" not in st.session_state:
    st.session_state["custom_inventory_stock"] = {}

if "solver_policy" not in st.session_state:
    st.session_state["solver_policy"] = "STANDARD_WEIGHTED"

if "active_solver" not in st.session_state:
    st.session_state["active_solver"] = "MILP_OPTIMIZER"

if "view_3d_engine" not in st.session_state:
    st.session_state["view_3d_engine"] = "Plotly 3D (Offline)"

# 3D Layer Visibility Flags
for flag in ["vis_critical", "vis_elevated", "vis_standard", "vis_conduits"]:
    if flag not in st.session_state:
        st.session_state[flag] = True

# Persistent Analyst Decisions
analyst_reviews = utils.load_analyst_decisions()


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION — 4 PRIMARY WORKSPACES (Zero Emoji)
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 4px 0 12px 0; border-bottom: 1px solid {theme.PANEL_BORDER}; margin-bottom: 14px;">
            <div style="font-size: 15px; font-weight: 800; letter-spacing: 0.1em; color: {theme.TEXT_MAIN};">
                RESCUEGRID
            </div>
            <div style="font-size: 11px; color: {theme.TEXT_SECONDARY}; margin-top: 2px;">
                Emergency Intelligence & Optimization
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    workspace = st.radio(
        "Operations Workspace",
        [
            "Command Center",
            "Evidence & Conflicts",
            "Resource Optimization",
            "Performance & Evaluation",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Developer & System Diagnostics in Collapsed Expander
    with st.expander("System Diagnostics & Asset Health", expanded=False):
        h_records = utils.inspect_pipeline_health()
        hdf = pd.DataFrame(h_records)[["name", "status", "rows", "size_kb"]]
        hdf.columns = ["Asset", "Status", "Rows", "KB"]
        st.dataframe(hdf, use_container_width=True, hide_index=True)
        st.caption(f"Solver: scipy.optimize.milp (HiGHS) | Python {sys.version.split()[0]}")
        st.caption(f"Scale: {st.session_state['simulation_scale']} | Scenario: {verified_sc_id}")


# -----------------------------------------------------------------------------
# WORKSPACE 1: COMMAND CENTER
# -----------------------------------------------------------------------------

if workspace == "Command Center":

    # Empty Registry Guard
    if not available_sc_ids or sc_index < 0:
        st.error("No scenarios available in the registry.")
        if st.button("Reload Default Scenarios"):
            st.session_state["scenario_registry"] = opt.get_all_scenarios(scale=st.session_state["simulation_scale"])
            st.session_state["active_scenario_id"] = "SCENARIO_SEVERE_SHORTAGE"
            st.rerun()
        st.stop()

    active_scenario = sc_registry[verified_sc_id]

    # Reconstruct inventory with any live user overrides
    current_inventory = []
    for res in active_scenario.inventory:
        custom_qty = st.session_state["custom_inventory_stock"].get(res.resource_type, res.available_quantity)
        current_inventory.append(opt.Resource(
            resource_type=res.resource_type,
            available_quantity=custom_qty,
            urgency_weight=res.urgency_weight,
            unit_cost=res.unit_cost,
            color_hex=res.color_hex,
            icon="",
        ))

    current_incidents = active_scenario.incidents

    # Run Solvers on Identical Inputs
    milp_plan = opt.solve_resource_allocation(
        current_incidents,
        current_inventory,
        policy=st.session_state["solver_policy"],
    )
    greedy_plan = opt.solve_greedy_baseline(
        current_incidents,
        current_inventory,
    )

    active_plan = milp_plan if st.session_state["active_solver"] == "MILP_OPTIMIZER" else greedy_plan
    eval_metrics = opt.compute_evaluation_metrics(
        current_incidents,
        current_inventory,
        active_plan,
        baseline_plan=greedy_plan if st.session_state["active_solver"] == "MILP_OPTIMIZER" else None,
    )

    # Top Status Bar
    status_badge = "badge-green" if active_plan.is_optimal else "badge-red"
    solver_tag = "MILP (OPTIMAL)" if active_plan.is_optimal else f"SOLVER ({active_plan.solver_status})"
    if st.session_state["active_solver"] == "GREEDY_BASELINE":
        solver_tag = "GREEDY BASELINE"
        status_badge = "badge-amber"

    st.markdown(
        f"""
        <div class="top-command-header">
            <div class="header-brand">
                <span>RESCUEGRID COMMAND CENTER</span>
                <span class="badge badge-cyan">{active_scenario.name}</span>
                <span class="badge {status_badge}">{solver_tag}</span>
            </div>
            <div class="header-meta">
                <span>Scale: <b style="color: {theme.TEXT_MAIN};">{st.session_state['simulation_scale']} ({len(current_incidents)} Sites)</b></span>
                <span>Runtime: <b style="color: {theme.ACCENT_CYAN};">{active_plan.runtime_ms:.1f} ms</b></span>
                <span>Depot Stock: <b style="color: {theme.TEXT_MAIN};">{sum(r.available_quantity for r in current_inventory)} units</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Ribbon (6 Aligned Cards, Zero Emoji, N/A on Zero Demand)
    crit_count = sum(1 for inc in current_incidents if inc.priority_score >= 80.0)
    avail_tot = sum(r.available_quantity for r in current_inventory)
    req_tot = eval_metrics.total_requested
    alc_tot = eval_metrics.total_allocated
    unm_tot = eval_metrics.total_unmet
    fulfillment_disp = "N/A" if eval_metrics.is_zero_demand else f"{eval_metrics.overall_fulfillment_pct:.1f}%"

    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-label">Total Incidents</div>
                <div class="kpi-value">{len(current_incidents)}</div>
                <div class="kpi-subtext">Active target sites</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Critical Incidents</div>
                <div class="kpi-value" style="color: {theme.CRITICAL_RED};">{crit_count}</div>
                <div class="kpi-subtext">Priority >= 80.0</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Depot Stock</div>
                <div class="kpi-value">{avail_tot}</div>
                <div class="kpi-subtext">{len(current_inventory)} resource types</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Allocated Units</div>
                <div class="kpi-value" style="color: {theme.SUCCESS_GREEN};">{alc_tot}</div>
                <div class="kpi-subtext">Of {req_tot} requested</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Unmet Deficit</div>
                <div class="kpi-value" style="color: {theme.CRITICAL_RED if unm_tot > 0 else theme.SUCCESS_GREEN};">{unm_tot}</div>
                <div class="kpi-subtext">Shortage volume</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Fulfillment</div>
                <div class="kpi-value" style="color: {theme.ACCENT_CYAN};">{fulfillment_disp}</div>
                <div class="kpi-subtext">{"Informational" if eval_metrics.is_zero_demand else "Overall satisfied"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- THREE-COLUMN OPERATIONAL LAYOUT ---
    col_ctrl, col_center, col_dossier = st.columns([1.1, 2.5, 1.2])

    # LEFT PANEL: Simulation Controls
    with col_ctrl:
        st.markdown(
            f"""
            <div class="panel-header">Simulation Controls</div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Scale Selector
        scale_opts = ["Small", "Medium", "Large", "Stress Test"]
        curr_sc_scale = st.session_state["simulation_scale"]
        sel_scale = st.selectbox(
            "Simulation Scale",
            scale_opts,
            index=scale_opts.index(curr_sc_scale) if curr_sc_scale in scale_opts else 0,
            key="cmd_scale_select",
        )
        if sel_scale != curr_sc_scale:
            st.session_state["simulation_scale"] = sel_scale
            st.session_state["scenario_registry"] = opt.get_all_scenarios(scale=sel_scale)
            st.session_state["custom_inventory_stock"] = {}
            st.session_state["selected_incident_id"] = None
            st.rerun()

        # 2. Scenario Dropdown
        chosen_sc_id = st.selectbox(
            "Scenario Preset",
            available_sc_ids,
            format_func=lambda k: sc_registry[k].name,
            index=sc_index,
            key="cmd_sc_select",
        )
        if chosen_sc_id != verified_sc_id:
            st.session_state["active_scenario_id"] = chosen_sc_id
            st.session_state["custom_inventory_stock"] = {}
            st.session_state["selected_incident_id"] = None
            st.rerun()

        # 3. Solver Selection
        st.markdown("**Allocation Algorithm**")
        algo_choice = st.radio(
            "Algorithm",
            ["MILP_OPTIMIZER", "GREEDY_BASELINE"],
            format_func=lambda x: "MILP Optimizer (Optimal)" if x == "MILP_OPTIMIZER" else "Greedy Priority Baseline",
            index=0 if st.session_state["active_solver"] == "MILP_OPTIMIZER" else 1,
            label_visibility="collapsed",
            key="cmd_algo_select",
        )
        if algo_choice != st.session_state["active_solver"]:
            st.session_state["active_solver"] = algo_choice
            st.rerun()

        # 4. Optimization Policy
        if st.session_state["active_solver"] == "MILP_OPTIMIZER":
            policy_options = ["STANDARD_WEIGHTED", "MIN_SERVICE_GUARANTEE", "BUNDLE_AWARE"]
            curr_pol = st.session_state["solver_policy"]
            pol_idx = policy_options.index(curr_pol) if curr_pol in policy_options else 0
            policy_choice = st.selectbox(
                "Optimization Policy",
                policy_options,
                format_func=lambda p: {
                    "STANDARD_WEIGHTED": "Standard Priority-Weighted Deficit",
                    "MIN_SERVICE_GUARANTEE": "Anti-Starvation Service Guarantee",
                    "BUNDLE_AWARE": "Bundle-Aware Critical Escalation",
                }.get(p, p),
                index=pol_idx,
                key="cmd_policy_select",
            )
            if policy_choice != st.session_state["solver_policy"]:
                st.session_state["solver_policy"] = policy_choice
                st.rerun()

        # 5. Layer Visibility Toggles
        with st.expander("Layer Visibility", expanded=False):
            st.session_state["vis_critical"] = st.checkbox("Critical Incidents (Red)", value=st.session_state["vis_critical"])
            st.session_state["vis_elevated"] = st.checkbox("High Priority (Amber)", value=st.session_state["vis_elevated"])
            st.session_state["vis_standard"] = st.checkbox("Standard Priority (Green)", value=st.session_state["vis_standard"])
            st.session_state["vis_conduits"] = st.checkbox("Allocation Conduits", value=st.session_state["vis_conduits"])

        # 6. Depot Inventory Sliders
        with st.expander("Depot Stock Adjustments", expanded=False):
            for res in active_scenario.inventory:
                curr_val = st.session_state["custom_inventory_stock"].get(res.resource_type, res.available_quantity)
                max_slider = max(50, res.available_quantity * 3, 20)
                new_val = st.slider(
                    res.resource_type.replace("_", " ").title(),
                    min_value=0,
                    max_value=max_slider,
                    value=curr_val,
                    key=f"stock_slider_{res.resource_type}",
                )
                if new_val != curr_val:
                    st.session_state["custom_inventory_stock"][res.resource_type] = new_val

            if st.button("Reset Stock Levels", use_container_width=True):
                st.session_state["custom_inventory_stock"] = {}
                st.rerun()

        # 7. 3D Engine Selection
        st.markdown("**3D Engine**")
        engine_opts = ["Plotly 3D (Offline)", "Three.js WebGL"]
        curr_eng = st.session_state["view_3d_engine"]
        eng_idx = engine_opts.index(curr_eng) if curr_eng in engine_opts else 0
        new_eng = st.selectbox("Renderer", engine_opts, index=eng_idx, label_visibility="collapsed", key="cmd_engine_select")
        if new_eng != curr_eng:
            st.session_state["view_3d_engine"] = new_eng
            st.rerun()

    # CENTER PANEL: 3D Tactical Response Environment
    with col_center:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div class="panel-header" style="border: none; margin: 0; padding: 0;">Tactical Response Grid</div>
                <div style="font-size: 11px; color: {theme.TEXT_SECONDARY};">Engine: {st.session_state['view_3d_engine']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sel_inc_id = st.session_state["selected_incident_id"]

        if st.session_state["view_3d_engine"] == "Three.js WebGL":
            threejs_code = viz3d.generate_threejs_html(
                current_incidents,
                current_inventory,
                active_plan,
                selected_incident_id=sel_inc_id,
            )
            st.components.v1.html(threejs_code, height=540)
        else:
            plotly_fig = viz3d.generate_plotly_3d_figure(
                current_incidents,
                current_inventory,
                active_plan,
                selected_incident_id=sel_inc_id,
                show_critical=st.session_state["vis_critical"],
                show_elevated=st.session_state["vis_elevated"],
                show_standard=st.session_state["vis_standard"],
                show_conduits=st.session_state["vis_conduits"],
            )
            st.plotly_chart(plotly_fig, use_container_width=True, height=540)

        # Synchronized Node Selector
        inc_ids = [inc.incident_id for inc in current_incidents]
        sel_idx = inc_ids.index(sel_inc_id) if sel_inc_id in inc_ids else 0
        c_sel_1, c_sel_2 = st.columns([3, 1])
        chosen_inc = c_sel_1.selectbox(
            "Select Site to Inspect",
            inc_ids,
            index=sel_idx,
            key="cmd_site_selector",
            label_visibility="collapsed",
        )
        if chosen_inc != sel_inc_id:
            st.session_state["selected_incident_id"] = chosen_inc
            st.rerun()

        if c_sel_2.button("Reset View", use_container_width=True):
            st.session_state["selected_incident_id"] = inc_ids[0] if inc_ids else None
            st.rerun()

    # RIGHT PANEL: Tactical Dossier & Depot Capacity
    with col_dossier:
        st.markdown(
            f"""
            <div class="panel-header">Tactical Dossier</div>
            """,
            unsafe_allow_html=True,
        )

        target_inc = next((inc for inc in current_incidents if inc.incident_id == st.session_state["selected_incident_id"]), current_incidents[0])
        alloc_map = active_plan.allocations.get(target_inc.incident_id, {})
        unmet_map = active_plan.unmet.get(target_inc.incident_id, {})

        p_badge = "badge-red" if target_inc.priority_score >= 80 else ("badge-amber" if target_inc.priority_score >= 50 else "badge-green")
        border_col = theme.CRITICAL_RED if target_inc.priority_score >= 80 else theme.ACCENT_CYAN

        st.markdown(
            f"""
            <div class="command-panel" style="border-left: 3px solid {border_col}; margin-bottom: 10px;">
                <div style="font-size: 11px; font-weight: 700; color: {theme.TEXT_SECONDARY};">SITE DOSSIER</div>
                <div style="font-size: 18px; font-weight: 700; color: {theme.TEXT_MAIN};">{target_inc.incident_id}</div>
                <div style="margin-top: 5px;">
                    <span class="badge {p_badge}">Priority: {target_inc.priority_score:.1f}</span>
                    <span class="badge badge-cyan">{target_inc.category.replace('_', ' ')}</span>
                </div>
                <div style="font-size: 11px; color: {theme.TEXT_MUTED}; margin-top: 6px;">
                    Event: {target_inc.event_name}<br>
                    Coords: ({target_inc.coordinates[0]:.0f}, {target_inc.coordinates[1]:.0f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Evidence Report**")
        st.caption(f"\"{target_inc.evidence_summary}\"" if target_inc.evidence_summary else "No textual dispatch notes recorded for this site.")

        st.markdown("**Resource Allocations**")
        for res in current_inventory:
            r_name = res.resource_type
            req = target_inc.demands.get(r_name, 0)
            alc = alloc_map.get(r_name, 0)
            unm = unmet_map.get(r_name, 0)
            if req > 0:
                unm_color = theme.SUCCESS_GREEN if unm == 0 else theme.CRITICAL_RED
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px; padding: 4px 6px; background: {theme.APP_BG}; border-radius: 3px;">
                        <span>{r_name.replace('_', ' ').title()}</span>
                        <span>Allocated: <b>{alc}/{req}</b> | Unmet: <b style="color: {unm_color};">{unm}</b></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("**Decision Reason**")
        st.caption(active_plan.decision_reasons.get(target_inc.incident_id, "Standard priority triage."))

        st.markdown("---")
        st.markdown("**Depot Stock Utilization**")
        for res in current_inventory:
            r_name = res.resource_type
            avail = res.available_quantity
            used = sum(active_plan.allocations.get(inc.incident_id, {}).get(r_name, 0) for inc in current_incidents)
            util_pct = (used / avail * 100.0) if avail > 0 else 100.0
            st.progress(min(1.0, util_pct / 100.0), text=f"{r_name.replace('_', ' ').title()}: {used}/{avail} units ({util_pct:.0f}%)")

    # --- BOTTOM OPERATIONAL SUMMARY ---
    st.markdown("---")
    st.markdown(
        f"""
        <div class="panel-header">Operational Summary & Shortage Radar</div>
        """,
        unsafe_allow_html=True,
    )

    sum_c1, sum_c2, sum_c3 = st.columns([1.2, 1.2, 1.6])

    with sum_c1:
        st.markdown("**Priority Elevation Distribution**")
        p_df = pd.DataFrame([{"Site": inc.incident_id, "Priority": inc.priority_score} for inc in current_incidents[:40]])
        fig_p = px.bar(p_df, x="Site", y="Priority", color="Priority", color_continuous_scale="Reds", range_y=[0, 105])
        fig_p.update_layout(theme.get_tactical_plotly_layout(height=200, show_legend=False))
        st.plotly_chart(fig_p, use_container_width=True)

    with sum_c2:
        st.markdown("**Resource Demand vs. Stock Summary**")
        sum_data = pd.DataFrame({
            "Metric": ["Requested", "Allocated", "Unmet"],
            "Units": [req_tot, alc_tot, unm_tot]
        })
        fig_sum = px.bar(sum_data, x="Metric", y="Units", color="Metric", color_discrete_map={"Requested": theme.ACCENT_CYAN, "Allocated": theme.SUCCESS_GREEN, "Unmet": theme.CRITICAL_RED})
        fig_sum.update_layout(theme.get_tactical_plotly_layout(height=200, show_legend=False))
        st.plotly_chart(fig_sum, use_container_width=True)

    with sum_c3:
        st.markdown("**Compact Shortage Alert List**")
        shortage_rows = []
        for inc in current_incidents:
            u_map = active_plan.unmet.get(inc.incident_id, {})
            for r, u in u_map.items():
                if u > 0:
                    shortage_rows.append({"Site": inc.incident_id, "Priority": f"{inc.priority_score:.1f}", "Resource": r.replace('_', ' ').title(), "Deficit": u})
        if shortage_rows:
            st.dataframe(pd.DataFrame(shortage_rows).head(50), use_container_width=True, hide_index=True, height=180)
        else:
            st.markdown(
                f"""
                <div style="background: {theme.PANEL_BG}; border: 1px solid {theme.SUCCESS_GREEN}; border-radius: 4px; padding: 14px; text-align: center; color: {theme.SUCCESS_GREEN}; font-size: 12px; font-weight: 600;">
                    Zero unmet deficits. All incident resource requests are fully satisfied.
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# WORKSPACE 2: EVIDENCE & CONFLICTS (Zero Emoji)
# -----------------------------------------------------------------------------

elif workspace == "Evidence & Conflicts":
    st.markdown(
        f"""
        <div class="top-command-header" style="border-left-color: {theme.CRITICAL_RED};">
            <div class="header-brand">
                <span>EVIDENCE RECONCILIATION & CONFLICT REVIEW</span>
                <span class="badge badge-red">Discrepancy Triage Queue</span>
            </div>
            <div class="header-meta">
                <span>Total Candidate Signals: <b style="color: {theme.TEXT_MAIN};">{len(conflicts_df) if conflicts_df is not None else 0}</b></span>
                <span>Analyst Action Required</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_conflicts, tab_duplicates, tab_analytics = st.tabs([
        "Potential Conflict Signals",
        "Candidate Similar Reports",
        "Evidence Analytics",
    ])

    with tab_conflicts:
        if conflicts_df is None or conflicts_df.empty:
            st.info("No conflict flags detected in potential_conflicts.tsv.")
        else:
            cdf = conflicts_df.copy()

            cf_c1, cf_c2 = st.columns(2)
            ev_list = sorted(cdf["event_name"].unique()) if "event_name" in cdf.columns else []
            sel_ev = cf_c1.selectbox("Filter by Disaster Event", ["All Events"] + ev_list, key="conf_ev_filter")
            if sel_ev != "All Events":
                cdf = cdf[cdf["event_name"] == sel_ev]

            st.caption(f"Displaying {len(cdf)} candidate discrepancy pairs.")

            for idx, row in cdf.iterrows():
                id_a = row.get("tweet_id_a", "A")
                id_b = row.get("tweet_id_b", "B")
                sim_val = float(row.get("similarity", 0.0))
                signals = utils.parse_conflict_signals(row.get("conflict_signals", ""))

                pair_key = f"{id_a}_{id_b}"
                saved_status = analyst_reviews.get("conflicts", {}).get(pair_key, {}).get("status", "UNREVIEWED")

                st.markdown(
                    f"""
                    <div class="command-panel" style="border-left: 3px solid {theme.CRITICAL_RED}; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <div>
                                <span style="font-weight: 700; color: {theme.TEXT_MAIN};">PAIR #{idx + 1} | {row.get('event_name', 'Unknown')}</span>
                                <span class="badge badge-cyan" style="margin-left: 8px;">Cosine Sim: {sim_val:.3f}</span>
                            </div>
                            <span class="badge badge-red">DISCREPANCY DETECTED</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for sig in signals:
                    st.markdown(f"**Mismatch Reason:** `{sig['summary']}`")

                t_a, t_b = st.columns(2)
                with t_a:
                    st.markdown(f"**Report A (#{id_a})** | Category: `{row.get('category_a', 'N/A')}`")
                    st.info(row.get("tweet_text_a", "Text unavailable"))

                with t_b:
                    st.markdown(f"**Report B (#{id_b})** | Category: `{row.get('category_b', 'N/A')}`")
                    st.info(row.get("tweet_text_b", "Text unavailable"))

                # Review Logger
                rc1, rc2, rc3 = st.columns([2, 2, 1])
                status_choices = [
                    "UNREVIEWED",
                    "TEMPORAL_UPDATE (Toll Revised Over Time)",
                    "GEOGRAPHIC_VARIATION (Distinct Sub-Localities)",
                    "TRUE_CONTRADICTION (Conflicting Ground Truth)",
                ]
                st_idx = status_choices.index(saved_status) if saved_status in status_choices else 0
                new_status = rc1.selectbox(f"Analyst Determination #{idx + 1}", status_choices, index=st_idx, key=f"conf_status_{pair_key}")
                note = rc2.text_input("Notes", value=analyst_reviews.get("conflicts", {}).get(pair_key, {}).get("notes", ""), key=f"conf_note_{pair_key}")

                if rc3.button("Save Decision", key=f"conf_btn_{pair_key}", use_container_width=True):
                    utils.save_analyst_decision("conflicts", pair_key, new_status, note)
                    st.success("Decision recorded.")

                st.markdown(f"<hr style='border-color: {theme.PANEL_BORDER}; margin: 12px 0;'>", unsafe_allow_html=True)

    with tab_duplicates:
        if pairs_df is None or pairs_df.empty:
            st.info("Similarity dataset not found.")
        else:
            pdf = pairs_df.copy()
            sim_col = "similarity_numeric" if "similarity_numeric" in pdf.columns else "similarity"
            thresh = st.slider("Similarity Threshold", 0.65, 1.0, 0.75, 0.01, key="dup_thresh")
            pdf = pdf[pdf[sim_col].astype(float) >= thresh]
            st.caption(f"Found {len(pdf):,} pairs exceeding similarity {thresh}.")
            p_cols = [c for c in ["similarity", "event_name", "category_a", "category_b", "tweet_id_a", "tweet_id_b", "tweet_text_a", "tweet_text_b"] if c in pdf.columns]
            st.dataframe(pdf.head(80)[p_cols], use_container_width=True, hide_index=True)

    with tab_analytics:
        st.markdown("#### Evidence Distribution Analytics")
        if conflicts_df is not None and not conflicts_df.empty:
            ea1, ea2 = st.columns(2)
            with ea1:
                all_sigs = [s.strip().split("(")[0] for row in conflicts_df["conflict_signals"] for s in str(row).split("|") if s.strip()]
                sig_counts = pd.Series(all_sigs).value_counts().reset_index()
                sig_counts.columns = ["Signal Type", "Count"]
                fig_sig = px.bar(sig_counts, x="Count", y="Signal Type", orientation="h", color="Count", color_continuous_scale="Reds", title="Conflict Signals by Type")
                fig_sig.update_layout(theme.get_tactical_plotly_layout(height=280))
                st.plotly_chart(fig_sig, use_container_width=True)

            with ea2:
                rev_statuses = [v.get("status", "UNREVIEWED") for v in analyst_reviews.get("conflicts", {}).values()]
                if not rev_statuses:
                    rev_statuses = ["UNREVIEWED"] * len(conflicts_df)
                rev_counts = pd.Series(rev_statuses).value_counts().reset_index()
                rev_counts.columns = ["Status", "Count"]
                fig_rev = px.pie(rev_counts, names="Status", values="Count", hole=0.45, color_discrete_sequence=[theme.CRITICAL_RED, theme.WARNING_AMBER, theme.SUCCESS_GREEN, theme.TEXT_SECONDARY], title="Analyst Triage Status")
                fig_rev.update_layout(theme.get_tactical_plotly_layout(height=280))
                st.plotly_chart(fig_rev, use_container_width=True)


# -----------------------------------------------------------------------------
# WORKSPACE 3: RESOURCE OPTIMIZATION (Zero Emoji)
# -----------------------------------------------------------------------------

elif workspace == "Resource Optimization":
    st.markdown(
        f"""
        <div class="top-command-header">
            <div class="header-brand">
                <span>RESOURCE OPTIMIZATION ANALYTICS</span>
                <span class="badge badge-cyan">{active_scenario.name}</span>
            </div>
            <div class="header-meta">
                <span>Algorithm Comparison: MILP vs Greedy Baseline</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if verified_sc_id in sc_registry:
        active_scenario = sc_registry[verified_sc_id]
        current_inventory = [
            opt.Resource(
                res.resource_type,
                st.session_state["custom_inventory_stock"].get(res.resource_type, res.available_quantity),
                res.urgency_weight,
                res.unit_cost,
                res.color_hex,
                icon=""
            ) for res in active_scenario.inventory
        ]
        current_incidents = active_scenario.incidents

        milp_plan = opt.solve_resource_allocation(current_incidents, current_inventory, policy=st.session_state["solver_policy"])
        greedy_plan = opt.solve_greedy_baseline(current_incidents, current_inventory)
        eval_metrics = opt.compute_evaluation_metrics(current_incidents, current_inventory, milp_plan, baseline_plan=greedy_plan)

        # 5.1 Comparison Summary Metrics
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Selected Algorithm", "MILP (Optimal)", delta=f"{milp_plan.runtime_ms:.1f} ms")
        m2.metric("Total Requested", f"{eval_metrics.total_requested} units")
        m3.metric("Allocated Units", f"{eval_metrics.total_allocated} units", delta=f"{eval_metrics.overall_fulfillment_pct:.1f}%")
        m4.metric("Unmet Deficit", f"{eval_metrics.total_unmet} units", delta="Shortage", delta_color="inverse")
        m5.metric("Weighted Penalty", f"{eval_metrics.priority_weighted_unmet:.2f}")
        m6.metric("Underserved Sites", f"{eval_metrics.incidents_with_unmet} of {len(current_incidents)}")

        st.markdown("---")

        # 5.2 Required Charts A & B
        o_c1, o_c2 = st.columns(2)

        with o_c1:
            st.markdown("#### Chart A: Allocation Comparison (MILP vs Greedy)")
            comp_bar_df = pd.DataFrame([
                {"Algorithm": "MILP Optimizer", "Metric": "Allocated", "Units": eval_metrics.total_allocated},
                {"Algorithm": "MILP Optimizer", "Metric": "Unmet", "Units": eval_metrics.total_unmet},
                {"Algorithm": "Greedy Baseline", "Metric": "Allocated", "Units": sum(sum(a.values()) for a in greedy_plan.allocations.values())},
                {"Algorithm": "Greedy Baseline", "Metric": "Unmet", "Units": sum(sum(u.values()) for u in greedy_plan.unmet.values())},
            ])
            fig_comp = px.bar(comp_bar_df, x="Algorithm", y="Units", color="Metric", barmode="group", color_discrete_map={"Allocated": theme.SUCCESS_GREEN, "Unmet": theme.CRITICAL_RED})
            fig_comp.update_layout(theme.get_tactical_plotly_layout(height=280))
            st.plotly_chart(fig_comp, use_container_width=True)

        with o_c2:
            st.markdown("#### Chart B: Resource Utilization & Remaining Stock")
            util_data = []
            for res in current_inventory:
                avail = res.available_quantity
                used = sum(milp_plan.allocations.get(inc.incident_id, {}).get(res.resource_type, 0) for inc in current_incidents)
                rem = max(0, avail - used)
                util_data.append({"Resource": res.resource_type.replace('_', ' ').title(), "State": "Allocated", "Units": used})
                util_data.append({"Resource": res.resource_type.replace('_', ' ').title(), "State": "Remaining", "Units": rem})
            util_df = pd.DataFrame(util_data)
            fig_util = px.bar(util_df, x="Resource", y="Units", color="State", barmode="stack", color_discrete_map={"Allocated": theme.ACCENT_CYAN, "Remaining": theme.PANEL_BORDER})
            fig_util.update_layout(theme.get_tactical_plotly_layout(height=280))
            st.plotly_chart(fig_util, use_container_width=True)

        st.markdown("---")

        # Required Charts D & E
        o_c3, o_c4 = st.columns(2)

        with o_c3:
            st.markdown("#### Chart D: Shortage Analysis (Unmet Demand by Resource)")
            shortage_by_res = []
            for res in current_inventory:
                r_name = res.resource_type
                tot_u = sum(milp_plan.unmet.get(inc.incident_id, {}).get(r_name, 0) for inc in current_incidents)
                shortage_by_res.append({"Resource": r_name.replace('_', ' ').title(), "Unmet Units": tot_u})
            fig_sh = px.bar(pd.DataFrame(shortage_by_res), x="Resource", y="Unmet Units", color="Unmet Units", color_continuous_scale="Reds")
            fig_sh.update_layout(theme.get_tactical_plotly_layout(height=280))
            st.plotly_chart(fig_sh, use_container_width=True)

        with o_c4:
            st.markdown("#### Chart E: Priority-Weighted Penalty Contribution")
            penalty_contrib = []
            for inc in current_incidents:
                u_map = milp_plan.unmet.get(inc.incident_id, {})
                inc_pen = sum(max(0.01, inc.priority_score / 100.0) * r.urgency_weight * u_map.get(r.resource_type, 0) for r in current_inventory)
                if inc_pen > 0:
                    penalty_contrib.append({"Site": inc.incident_id, "Priority": inc.priority_score, "Penalty": inc_pen})
            if penalty_contrib:
                fig_pen = px.pie(pd.DataFrame(penalty_contrib).head(15), names="Site", values="Penalty", color="Priority", color_continuous_scale="Reds", hole=0.45)
                fig_pen.update_layout(theme.get_tactical_plotly_layout(height=280))
                st.plotly_chart(fig_pen, use_container_width=True)
            else:
                st.markdown(
                    f"""
                    <div style="background: {theme.PANEL_BG}; border: 1px solid {theme.SUCCESS_GREEN}; border-radius: 4px; padding: 20px; text-align: center; color: {theme.SUCCESS_GREEN}; font-size: 13px;">
                        Zero weighted penalty incurred. All incident demands fully satisfied.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # Required Chart F: Resource Demand vs Supply
        st.markdown("#### Chart F: Resource Demand versus Available Supply")
        supply_demand_data = []
        for res in current_inventory:
            r_name = res.resource_type
            d_tot = sum(inc.demands.get(r_name, 0) for inc in current_incidents)
            s_tot = res.available_quantity
            supply_demand_data.append({"Resource": r_name.replace('_', ' ').title(), "Type": "Requested Demand", "Units": d_tot})
            supply_demand_data.append({"Resource": r_name.replace('_', ' ').title(), "Type": "Available Supply", "Units": s_tot})
        fig_sd = px.bar(pd.DataFrame(supply_demand_data), x="Resource", y="Units", color="Type", barmode="group", color_discrete_map={"Requested Demand": theme.WARNING_AMBER, "Available Supply": theme.ACCENT_CYAN})
        fig_sd.update_layout(theme.get_tactical_plotly_layout(height=260))
        st.plotly_chart(fig_sd, use_container_width=True)

        st.markdown("---")

        # Chart C: Incident Allocation Matrix (with pagination for large scale)
        st.markdown("#### Chart C: Incident-Resource Allocation Matrix")
        matrix_rows = []
        for inc in current_incidents:
            row_dict = {"Site": inc.incident_id, "Priority": inc.priority_score}
            for res in current_inventory:
                r_name = res.resource_type
                req = inc.demands.get(r_name, 0)
                alc = milp_plan.allocations.get(inc.incident_id, {}).get(r_name, 0)
                unm = milp_plan.unmet.get(inc.incident_id, {}).get(r_name, 0)
                row_dict[f"{r_name.replace('_', ' ').title()} (Alc/Req)"] = f"{alc}/{req}"
                row_dict[f"{r_name.replace('_', ' ').title()} (Unmet)"] = unm
            matrix_rows.append(row_dict)

        m_df = pd.DataFrame(matrix_rows)
        # Search and filter
        search_kw = st.text_input("Filter Incident by ID or Priority", "", key="matrix_filter")
        if search_kw:
            m_df = m_df[m_df["Site"].str.contains(search_kw, case=False, na=False)]

        st.dataframe(m_df.head(100), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# WORKSPACE 4: PERFORMANCE & EVALUATION (Zero Emoji)
# -----------------------------------------------------------------------------

elif workspace == "Performance & Evaluation":
    st.markdown(
        f"""
        <div class="top-command-header">
            <div class="header-brand">
                <span>PERFORMANCE & EVALUATION BENCHMARK</span>
                <span class="badge badge-cyan">Scale: {st.session_state['simulation_scale']}</span>
            </div>
            <div class="header-meta">
                <span>Cross-Scenario Evaluation Suite</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Benchmark Execution
    bench_results = []
    f_data = []
    p_comp_data = []

    for s_id, s in sc_registry.items():
        pol = "MIN_SERVICE_GUARANTEE" if "GREEDY_VS_OPTIMIZER" in s_id or "STARVATION" in s_id else "STANDARD_WEIGHTED"
        m_res = opt.solve_resource_allocation(s.incidents, s.inventory, policy=pol)
        g_res = opt.solve_greedy_baseline(s.incidents, s.inventory)

        m_met = opt.compute_evaluation_metrics(s.incidents, s.inventory, m_res, baseline_plan=g_res)
        g_met = opt.compute_evaluation_metrics(s.incidents, s.inventory, g_res)

        diff = m_met.delta_vs_baseline.get("identical_allocations", True) if m_met.delta_vs_baseline else True

        bench_results.append({
            "Scenario": s.name,
            "Sites": len(s.incidents),
            "Requested": m_met.total_requested,
            "Depot Stock": sum(r.available_quantity for r in s.inventory),
            "MILP Fulfillment": f"{m_met.overall_fulfillment_pct:.1f}%" if not m_met.is_zero_demand else "N/A",
            "Greedy Fulfillment": f"{g_met.overall_fulfillment_pct:.1f}%" if not g_met.is_zero_demand else "N/A",
            "MILP Penalty": f"{m_met.priority_weighted_unmet:.2f}",
            "Greedy Penalty": f"{g_met.priority_weighted_unmet:.2f}",
            "MILP Runtime": f"{m_met.runtime_ms:.1f} ms",
            "Greedy Runtime": f"{g_met.runtime_ms:.1f} ms",
            "Outcome": "Optimal Match" if diff else "MILP Equity Advantage",
        })

        if not m_met.is_zero_demand:
            sc_short = s.name[:24] + "..." if len(s.name) > 24 else s.name
            f_data.append({"Scenario": sc_short, "Solver": "MILP", "Fulfillment (%)": m_met.overall_fulfillment_pct})
            f_data.append({"Scenario": sc_short, "Solver": "Greedy", "Fulfillment (%)": g_met.overall_fulfillment_pct})

            p_comp_data.append({"Scenario": sc_short, "Solver": "MILP", "Penalty": m_res.objective_value})
            p_comp_data.append({"Scenario": sc_short, "Solver": "Greedy", "Penalty": g_res.objective_value})

    st.markdown("#### Scenario Benchmark Matrix")
    st.dataframe(pd.DataFrame(bench_results), use_container_width=True, hide_index=True)

    st.markdown("---")
    be_c1, be_c2 = st.columns(2)

    with be_c1:
        st.markdown("#### Fulfillment Rate Comparison (%)")
        if f_data:
            fig_f = px.bar(pd.DataFrame(f_data), x="Scenario", y="Fulfillment (%)", color="Solver", barmode="group", color_discrete_map={"MILP": theme.ACCENT_CYAN, "Greedy": theme.WARNING_AMBER}, range_y=[0, 105])
            fig_f.update_layout(theme.get_tactical_plotly_layout(height=280))
            st.plotly_chart(fig_f, use_container_width=True)

    with be_c2:
        st.markdown("#### Objective Penalty Comparison (Lower is Better)")
        if p_comp_data:
            fig_pen_comp = px.bar(pd.DataFrame(p_comp_data), x="Scenario", y="Penalty", color="Solver", barmode="group", color_discrete_map={"MILP": theme.ACCENT_CYAN, "Greedy": theme.WARNING_AMBER})
            fig_pen_comp.update_layout(theme.get_tactical_plotly_layout(height=280))
            st.plotly_chart(fig_pen_comp, use_container_width=True)


# -----------------------------------------------------------------------------
# GLOBAL FOOTER (Zero Emoji)
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: {theme.TEXT_MUTED}; font-size: 11px; padding: 4px;">
        RESCUEGRID &bull; Operations Research & Constrained Emergency Resource Optimization Platform &bull; HiGHS Solver &bull; Decision Support Only
    </div>
    """,
    unsafe_allow_html=True,
)
