"""
RESCUEGRID — 3D Emergency Command Center & Optimization Engine
=============================================================
Next-generation tactical command center for disaster social media intelligence,
conflict reconciliation, priority scoring, and mixed-integer linear programming (MILP)
resource allocation under severe supply constraints.

Decision-Support Prototype. All conflict signals and priority scores are heuristic.
Inventory values and incident resource demands are synthetic demonstration inputs.
Human authorization is required for all real-world dispatch decisions.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Centralized RESCUEGRID engine imports
import pipeline_utils as utils
import optimization as opt
import visualization_3d as viz3d

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & DIGITAL TWIN COMMAND THEME CSS
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="RESCUEGRID | 3D Emergency Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tactical Dark Command Center CSS (inspired by NEXAI Digital Twin aesthetic)
TACTICAL_THEME_CSS = """
<style>
    /* Dark Slate & Navy Command Background */
    .stApp {
        background-color: #070B14;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid #1E293B;
    }
    
    /* Top Status Bar */
    .top-status-bar {
        background: linear-gradient(90deg, #0F172A 0%, #131E32 50%, #0F172A 100%);
        border: 1px solid #1E293B;
        border-bottom: 2px solid #00F0FF;
        border-radius: 6px;
        padding: 10px 18px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0, 240, 255, 0.08);
    }
    
    .status-brand {
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: #00F0FF;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .status-meta {
        font-size: 12px;
        color: #94A3B8;
        display: flex;
        gap: 16px;
    }
    
    /* Tactical KPI Cards */
    .tactical-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 6px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    
    .card-title {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 4px;
    }
    
    .card-value {
        font-size: 22px;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    
    .card-subtext {
        font-size: 11px;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Badges */
    .badge-cyan { background-color: rgba(0, 240, 255, 0.15); color: #00F0FF; border: 1px solid rgba(0, 240, 255, 0.4); }
    .badge-red { background-color: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-orange { background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-green { background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4); }

    .tag-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    
    /* Streamlit tabs and headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1E293B;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0F172A;
        border-radius: 4px 4px 0 0;
        color: #94A3B8;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #00F0FF !important;
        font-weight: 600;
    }
</style>
"""
st.markdown(TACTICAL_THEME_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA LOADERS & SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------

reports_df, reports_source = utils.load_reports_dataset()
pairs_df, pairs_source = utils.load_similarity_dataset()
conflicts_df = utils.load_conflicts_dataset()
all_scenarios = opt.get_all_scenarios()

# Session State
if "active_scenario_id" not in st.session_state:
    st.session_state["active_scenario_id"] = "SCENARIO_SEVERE_SHORTAGE"

if "selected_incident_id" not in st.session_state:
    st.session_state["selected_incident_id"] = None

if "custom_inventory_stock" not in st.session_state:
    st.session_state["custom_inventory_stock"] = {}

if "solver_policy" not in st.session_state:
    st.session_state["solver_policy"] = "STANDARD_WEIGHTED"

if "active_solver" not in st.session_state:
    st.session_state["active_solver"] = "MILP_OPTIMIZER"

if "view_3d_engine" not in st.session_state:
    st.session_state["view_3d_engine"] = "Three.js WebGL"

# Load persisted analyst reviews
analyst_reviews = utils.load_analyst_decisions()


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🛰️ RESCUEGRID 3D")
    st.caption("Tactical Operations & Resource Optimization • v3.0")

    app_mode = st.radio(
        "Operations Module",
        [
            "🕹️ 3D Command Center",
            "⚖️ Evidence & Conflict Review",
            "🔍 Historical Report Intelligence",
            "🛡️ System Health & Diagnostics",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### Evidence Attribution Guardrail")
    st.markdown(
        """
        <div style="font-size: 11px; color: #94A3B8; line-height: 1.5; background: #0F172A; padding: 10px; border-radius: 6px; border-left: 3px solid #00F0FF;">
            <b>• DATA-OBSERVED:</b> Real HumAID tweets, events, labels.<br>
            <b>• RULE-BASED:</b> Discrepancy flags, priority scoring.<br>
            <b>• MODEL-DERIVED:</b> TF-IDF similarity vectors.<br>
            <b>• SYNTHETIC:</b> 3D coords, inventories, resource demands.<br>
            <i>Human commander approval required for dispatch.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("NEXAI 2026 Operations Prototype • VIT")


# -----------------------------------------------------------------------------
# MODULE 1: 🕹️ 3D COMMAND CENTER & OPTIMIZATION ENGINE
# -----------------------------------------------------------------------------

if app_mode == "🕹️ 3D Command Center":

    # --- A. TOP STATUS BAR ---
    active_sc_id = st.session_state["active_scenario_id"]
    active_scenario = all_scenarios.get(active_sc_id, all_scenarios["SCENARIO_SEVERE_SHORTAGE"])

    # Check if custom inventory overrides exist
    current_inventory = []
    for res in active_scenario.inventory:
        custom_qty = st.session_state["custom_inventory_stock"].get(res.resource_type, res.available_quantity)
        current_inventory.append(opt.Resource(
            resource_type=res.resource_type,
            available_quantity=custom_qty,
            urgency_weight=res.urgency_weight,
            color_hex=res.color_hex,
            icon=res.icon,
        ))

    current_incidents = active_scenario.incidents

    # Run Solvers: Both MILP and Greedy
    milp_plan = opt.solve_resource_allocation(
        current_incidents,
        current_inventory,
        policy=st.session_state["solver_policy"],
    )
    greedy_plan = opt.solve_greedy_baseline(
        current_incidents,
        current_inventory,
    )

    # Active plan based on user selection
    active_plan = milp_plan if st.session_state["active_solver"] == "MILP_OPTIMIZER" else greedy_plan
    eval_metrics = opt.compute_evaluation_metrics(
        current_incidents,
        current_inventory,
        active_plan,
        baseline_plan=greedy_plan if st.session_state["active_solver"] == "MILP_OPTIMIZER" else None,
    )

    # Top Status Bar Render
    status_color = "#10B981" if eval_metrics.overall_fulfillment_pct >= 90 else ("#F59E0B" if eval_metrics.overall_fulfillment_pct >= 50 else "#EF4444")
    st.markdown(
        f"""
        <div class="top-status-bar">
            <div class="status-brand">
                <span>🛰️</span> RESCUEGRID DIGITAL TWIN COMMAND
                <span class="tag-pill badge-cyan" style="font-size: 10px; margin-left: 8px;">{active_scenario.name}</span>
            </div>
            <div class="status-meta">
                <span>INCIDENTS: <b style="color: #F8FAFC;">{eval_metrics.incidents_count}</b></span>
                <span>DEPOT STOCK: <b style="color: #00F0FF;">{sum(r.available_quantity for r in current_inventory)} units</b></span>
                <span>FULFILLMENT: <b style="color: {status_color};">{eval_metrics.overall_fulfillment_pct}%</b></span>
                <span>SOLVER: <b style="color: #00F0FF;">{st.session_state['active_solver']} ({eval_metrics.runtime_ms:.1f}ms)</b></span>
                <span class="tag-pill badge-orange">SYNTHETIC DEMO</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- B. THREE-COLUMN TACTICAL LAYOUT ---
    col_ctrl, col_center, col_dossier = st.columns([1.1, 2.4, 1.2])

    # ---------------------------------------------------------
    # LEFT PANEL: TACTICAL SIMULATION CONTROLS
    # ---------------------------------------------------------
    with col_ctrl:
        st.markdown("#### 🎛️ Simulation Controls")

        # 1. Scenario Selection
        sc_names = {s_id: s.name for s_id, s in all_scenarios.items()}
        chosen_sc_id = st.selectbox(
            "Select Scenario",
            list(sc_names.keys()),
            format_func=lambda k: sc_names[k],
            index=list(sc_names.keys()).index(active_sc_id),
        )
        if chosen_sc_id != active_sc_id:
            st.session_state["active_scenario_id"] = chosen_sc_id
            st.session_state["custom_inventory_stock"] = {}
            st.session_state["selected_incident_id"] = None
            st.rerun()

        # Optional Historical Synthesis Event Selector
        if reports_df is not None and "event_name" in reports_df.columns:
            with st.expander("🔗 Synthesize from Historical Reports", expanded=False):
                hist_events = sorted(reports_df["event_name"].unique())
                sel_hist_event = st.selectbox("Historical Event", hist_events)
                top_k = st.slider("Top Reports to Cluster", 3, 8, 4)
                if st.button("Generate Historical Scenario", use_container_width=True):
                    hist_scenario = utils.build_scenario_from_historical_reports(reports_df, sel_hist_event, top_n=top_k)
                    if hist_scenario:
                        all_scenarios[hist_scenario.scenario_id] = hist_scenario
                        st.session_state["active_scenario_id"] = hist_scenario.scenario_id
                        st.session_state["custom_inventory_stock"] = {}
                        st.rerun()

        st.markdown("---")

        # 2. Allocation Algorithm Selection
        st.markdown("**Allocation Engine**")
        algo_choice = st.radio(
            "Solver",
            ["MILP_OPTIMIZER", "GREEDY_BASELINE"],
            format_func=lambda x: "🚀 MILP Optimizer (scipy.optimize.milp)" if x == "MILP_OPTIMIZER" else "⚡ Greedy Priority-First Baseline",
            index=0 if st.session_state["active_solver"] == "MILP_OPTIMIZER" else 1,
            label_visibility="collapsed",
        )
        if algo_choice != st.session_state["active_solver"]:
            st.session_state["active_solver"] = algo_choice
            st.rerun()

        # 3. Solver Policy Options (if MILP)
        if st.session_state["active_solver"] == "MILP_OPTIMIZER":
            policy_choice = st.selectbox(
                "Optimization Policy",
                ["STANDARD_WEIGHTED", "MIN_SERVICE_GUARANTEE", "BUNDLE_AWARE"],
                format_func=lambda p: {
                    "STANDARD_WEIGHTED": "Standard Priority-Weighted Deficit",
                    "MIN_SERVICE_GUARANTEE": "Minimum Service Guarantee (Anti-Starvation)",
                    "BUNDLE_AWARE": "Bundle-Aware Critical Escalation",
                }.get(p, p),
                index=["STANDARD_WEIGHTED", "MIN_SERVICE_GUARANTEE", "BUNDLE_AWARE"].index(st.session_state["solver_policy"]),
            )
            if policy_choice != st.session_state["solver_policy"]:
                st.session_state["solver_policy"] = policy_choice
                st.rerun()

        st.markdown("---")

        # 4. Interactive Regional Inventory Adjustments
        st.markdown("**Depot Stock Levels**")
        for res in active_scenario.inventory:
            curr_val = st.session_state["custom_inventory_stock"].get(res.resource_type, res.available_quantity)
            new_val = st.slider(
                f"{res.icon} {res.resource_type.title()}",
                min_value=0,
                max_value=max(30, res.available_quantity * 3),
                value=curr_val,
                key=f"slider_{res.resource_type}",
            )
            if new_val != curr_val:
                st.session_state["custom_inventory_stock"][res.resource_type] = new_val

        col_act1, col_act2 = st.columns(2)
        if col_act1.button("🔄 Reset Stock", use_container_width=True):
            st.session_state["custom_inventory_stock"] = {}
            st.rerun()

        # 5. 3D View Engine Toggle
        engine_opt = col_act2.selectbox(
            "3D View",
            ["Three.js WebGL", "Plotly 3D"],
            index=0 if st.session_state["view_3d_engine"] == "Three.js WebGL" else 1,
            label_visibility="collapsed",
        )
        if engine_opt != st.session_state["view_3d_engine"]:
            st.session_state["view_3d_engine"] = engine_opt
            st.rerun()

    # ---------------------------------------------------------
    # CENTER PANEL: INTERACTIVE 3D DIGITAL TWIN OPERATIONS SCENE
    # ---------------------------------------------------------
    with col_center:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #F8FAFC; font-size: 14px;">3D TACTICAL RESPONSE GRID</span>
                <span style="font-size: 11px; color: #64748B;">RENDERER: {st.session_state['view_3d_engine'].upper()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sel_inc_id = st.session_state["selected_incident_id"]

        # Render 3D Scene based on active engine
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
            )
            st.plotly_chart(plotly_fig, use_container_width=True, height=540)

        # Tactical Node Quick Selector below 3D viewport
        inc_ids = [inc.incident_id for inc in current_incidents]
        sel_idx = inc_ids.index(sel_inc_id) if sel_inc_id in inc_ids else 0
        chosen_inc = st.selectbox(
            "Select Incident to Inspect in Dossier Panel",
            inc_ids,
            index=sel_idx,
            key="node_selector",
        )
        if chosen_inc != sel_inc_id:
            st.session_state["selected_incident_id"] = chosen_inc
            st.rerun()

    # ---------------------------------------------------------
    # RIGHT PANEL: INCIDENT & LOGISTICS DOSSIER
    # ---------------------------------------------------------
    with col_dossier:
        st.markdown("#### 📋 Operational Dossier")

        # Get active incident
        target_inc = next((inc for inc in current_incidents if inc.incident_id == st.session_state["selected_incident_id"]), current_incidents[0])

        alloc_map = active_plan.allocations.get(target_inc.incident_id, {})
        unmet_map = active_plan.unmet.get(target_inc.incident_id, {})
        inc_req_tot = target_inc.total_demand
        inc_alloc_tot = sum(alloc_map.values())
        inc_unmet_tot = sum(unmet_map.values())

        p_badge = "badge-red" if target_inc.priority_score >= 80 else ("badge-orange" if target_inc.priority_score >= 50 else "badge-green")

        st.markdown(
            f"""
            <div class="tactical-card" style="border-left: 3px solid {'#EF4444' if target_inc.priority_score >= 80 else '#00F0FF'};">
                <div class="card-title">INCIDENT DOSSIER</div>
                <div class="card-value" style="font-size: 18px;">{target_inc.incident_id}</div>
                <div style="margin-top: 6px;">
                    <span class="tag-pill {p_badge}">Priority: {target_inc.priority_score:.1f}/100</span>
                    <span class="tag-pill badge-cyan">{target_inc.category}</span>
                </div>
                <div class="card-subtext" style="margin-top: 8px;">
                    Event: <b>{target_inc.event_name}</b><br>
                    Coordinates: ({target_inc.coordinates[0]:.0f}, {target_inc.coordinates[1]:.0f}, {target_inc.coordinates[2]:.0f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Evidence Report Excerpt:**")
        st.info(f"\"{target_inc.evidence_summary}\"" if target_inc.evidence_summary else "No textual dispatch notes recorded for this site.")

        st.markdown("**Resource Allocations & Needs:**")
        for res in current_inventory:
            r_name = res.resource_type
            req = target_inc.demands.get(r_name, 0)
            alc = alloc_map.get(r_name, 0)
            unm = unmet_map.get(r_name, 0)

            c_text = "#10B981" if unm == 0 else "#EF4444"
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; padding: 4px 6px; background: #070B14; border-radius: 4px;">
                    <span>{res.icon} {r_name.title()}</span>
                    <span>Allocated: <b>{alc}/{req}</b> | Unmet: <b style="color: {c_text};">{unm}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("**Allocation Decision Reason:**")
        reason_text = active_plan.decision_reasons.get(target_inc.incident_id, "Standard triage priority.")
        st.caption(f"💡 {reason_text}")

        st.markdown("---")
        st.markdown("**Depot Capacity Utilization**")
        for res in current_inventory:
            r_name = res.resource_type
            avail = res.available_quantity
            used = sum(active_plan.allocations.get(inc.incident_id, {}).get(r_name, 0) for inc in current_incidents)
            util_pct = (used / avail * 100.0) if avail > 0 else 100.0
            st.progress(min(1.0, util_pct / 100.0), text=f"{res.icon} {r_name.title()}: {used}/{avail} units ({util_pct:.0f}%)")

    # --- C. BOTTOM RESULTS PANEL: COMPARISON & EVALUATION ---
    st.markdown("---")
    st.subheader("📊 Optimization vs. Greedy Baseline Benchmarking")

    tab_eval, tab_comp, tab_narrative = st.tabs([
        "📈 Comparative Performance Metrics",
        "📋 Incident Assignment Matrix",
        "💡 Important Allocation Trade-Offs",
    ])

    with tab_eval:
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Total Requested", f"{eval_metrics.total_requested} units")
        m_col2.metric("Total Allocated", f"{eval_metrics.total_allocated} units", delta=f"{eval_metrics.overall_fulfillment_pct}% fulfilled")
        m_col3.metric("Total Unmet Deficit", f"{eval_metrics.total_unmet} units", delta="-Shortage", delta_color="inverse")
        m_col4.metric("Priority-Weighted Penalty", f"{eval_metrics.priority_weighted_unmet:.2f}")
        m_col5.metric("Solver Speed", f"{eval_metrics.runtime_ms:.2f} ms", delta=eval_metrics.solver_status)

        # Comparative Delta Card (if comparing MILP vs Baseline)
        if eval_metrics.delta_vs_baseline:
            d = eval_metrics.delta_vs_baseline
            is_identical = d["identical_allocations"]

            if is_identical:
                st.info(
                    "ℹ️ **Empirical Finding:** Both the MILP Optimizer and the Greedy Priority Baseline produced the **identical** allocation plan for this scenario. "
                    "This occurs when high-priority requests can be ordered linearly without conflicting multi-resource bottleneck trade-offs."
                )
            else:
                st.success(
                    f"🎯 **Algorithmic Divergence Detected!** "
                    f"The MILP Optimizer produced a distinct allocation strategy compared to the Greedy Baseline. "
                    f"Penalty difference: `{d['weighted_penalty_diff']:+.2f}` | "
                    f"Unmet unit difference: `{d['unmet_diff_units']:+d}` units."
                )

        # Bar charts: Fulfillment by Resource
        res_data = [
            {"Resource": r.title(), "Fulfillment (%)": eval_metrics.fulfillment_by_resource.get(r, 0.0)}
            for r in current_inventory[0].demands.keys() if hasattr(current_inventory[0], 'demands')
        ] if False else [
            {"Resource": r.resource_type.title(), "Fulfillment (%)": eval_metrics.fulfillment_by_resource.get(r.resource_type, 0.0)}
            for r in current_inventory
        ]
        res_df = pd.DataFrame(res_data)

        b_c1, b_c2 = st.columns([1, 1])
        with b_c1:
            fig_bar = px.bar(
                res_df,
                x="Resource",
                y="Fulfillment (%)",
                color="Fulfillment (%)",
                color_continuous_scale="Teal",
                range_y=[0, 105],
                title="Resource Fulfillment Rates (%)",
            )
            fig_bar.update_layout(
                paper_bgcolor="#070B14",
                plot_bgcolor="#0F172A",
                font_color="#CBD5E1",
                height=260,
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with b_c2:
            # Shortage per incident bar
            inc_shortages = []
            for inc in current_incidents:
                u_tot = sum(active_plan.unmet.get(inc.incident_id, {}).values())
                inc_shortages.append({"Incident": inc.incident_id, "Priority": inc.priority_score, "Unmet Units": u_tot})
            short_df = pd.DataFrame(inc_shortages)

            fig_short = px.bar(
                short_df,
                x="Incident",
                y="Unmet Units",
                color="Priority",
                color_continuous_scale="Reds",
                title="Unmet Deficits Across Competing Sites",
            )
            fig_short.update_layout(
                paper_bgcolor="#070B14",
                plot_bgcolor="#0F172A",
                font_color="#CBD5E1",
                height=260,
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_short, use_container_width=True)

    with tab_comp:
        st.markdown("#### Side-by-Side Algorithm Output Comparison")
        comp_rows = []
        for inc in current_incidents:
            m_alc = "; ".join(f"{r}={q}" for r, q in milp_plan.allocations.get(inc.incident_id, {}).items())
            m_unm = "; ".join(f"{r}={q}" for r, q in milp_plan.unmet.get(inc.incident_id, {}).items() if q > 0) or "None"
            g_alc = "; ".join(f"{r}={q}" for r, q in greedy_plan.allocations.get(inc.incident_id, {}).items())
            g_unm = "; ".join(f"{r}={q}" for r, q in greedy_plan.unmet.get(inc.incident_id, {}).items() if q > 0) or "None"

            comp_rows.append({
                "Incident ID": inc.incident_id,
                "Priority": inc.priority_score,
                "Total Demand": inc.total_demand,
                "MILP Allocated": m_alc,
                "MILP Unmet": m_unm,
                "Greedy Allocated": g_alc,
                "Greedy Unmet": g_unm,
                "Differs": (m_alc != g_alc),
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    with tab_narrative:
        st.markdown("#### Algorithmic Trade-Off Explanation")
        st.markdown(
            f"""
            <div style="background: #0F172A; padding: 14px 18px; border-radius: 6px; border-left: 3px solid #00F0FF; font-size: 13px; line-height: 1.6; color: #CBD5E1;">
                <b>Active Solver Narrative:</b><br>
                {active_plan.trade_off_explanation}<br><br>
                <b>Core Optimization Mechanics:</b><br>
                • <b>Decision Variables:</b> x[i, r] denotes units of resource 'r' assigned to incident 'i'. Unmet variable u[i, r] captures unfulfilled demand.<br>
                • <b>Objective Function:</b> Minimize cumulative priority-weighted unmet demand: <code>min sum(weight[i, r] * u[i, r])</code>.<br>
                • <b>Policy Enforcement:</b> When <i>Minimum Service Guarantee</i> is selected, constraints ensure that no site with positive demand is completely starved if sufficient units exist.<br>
                • <b>Safety Notice:</b> Synthetic scenario calculations are decision-support only. Live field dispatch requires certified emergency commander authorization.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# MODULE 2: ⚖️ EVIDENCE & CONFLICT REVIEW
# -----------------------------------------------------------------------------

elif app_mode == "⚖️ Evidence & Conflict Review":
    st.title("⚖️ Evidence & Conflict Review")
    st.caption("Inspect candidate report pairs, evaluate potential opposing claims, and log persistent analyst decisions.")

    st.markdown(
        """
        <div class="top-status-bar" style="border-bottom-color: #EF4444;">
            <div class="status-brand">
                <span>⚠️</span> CONFLICT RECONCILIATION & HUMAN TRIAGE
            </div>
            <div class="status-meta">
                <span>RULE-BASED SIGNALS • UNVERIFIED HYPOTHESES FOR ANALYST VERIFICATION</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_conflicts, tab_duplicates = st.tabs(["⚖️ Potential Discrepancies & Conflicts", "🔗 Candidate Similar Reports"])

    # SUBTAB 1: Potential Conflicts
    with tab_conflicts:
        if conflicts_df is None or conflicts_df.empty:
            st.info("No conflict flags detected in `potential_conflicts.tsv`. Run `python conflicts.py`.")
        else:
            cdf = conflicts_df.copy()

            # Filters
            f1, f2 = st.columns(2)
            events = sorted(cdf["event_name"].unique()) if "event_name" in cdf.columns else []
            sel_ev = f1.selectbox("Filter Discrepancies by Event", ["All Events"] + events)
            if sel_ev != "All Events":
                cdf = cdf[cdf["event_name"] == sel_ev]

            st.markdown(f"**Flagged Conflict Pairs:** `{len(cdf)}`")

            for idx, row in cdf.iterrows():
                id_a = row.get("tweet_id_a", "A")
                id_b = row.get("tweet_id_b", "B")
                sim_val = float(row.get("similarity", 0.0))
                signals = utils.parse_conflict_signals(row.get("conflict_signals", ""))

                pair_key = f"{id_a}_{id_b}"
                saved_status = analyst_reviews.get("conflicts", {}).get(pair_key, {}).get("status", "UNREVIEWED")

                st.markdown(
                    f"""
                    <div class="tactical-card" style="border-left: 3px solid #EF4444; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div>
                                <span style="font-weight: 700; color: #F8FAFC; font-size: 14px;">PAIR #{idx + 1} | {row.get('event_name', 'Unknown Event')}</span>
                                <span class="tag-pill badge-cyan" style="margin-left: 8px;">Cosine Sim: {sim_val:.3f}</span>
                            </div>
                            <span class="tag-pill badge-red">POTENTIAL CONFLICT SIGNAL</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                for sig in signals:
                    st.markdown(f"🚩 **Mismatch Indicator:** `{sig['summary']}`")

                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown(f"**Report A (#{id_a})** | Category: `{row.get('category_a', 'N/A')}`")
                    st.info(row.get("tweet_text_a", "Text unavailable"))

                with c_b:
                    st.markdown(f"**Report B (#{id_b})** | Category: `{row.get('category_b', 'N/A')}`")
                    st.info(row.get("tweet_text_b", "Text unavailable"))

                # Analyst Decision Logger
                rev_c1, rev_c2, rev_c3 = st.columns([2, 2, 1])
                new_status = rev_c1.selectbox(
                    f"Analyst Determination for #{idx + 1}",
                    [
                        "UNREVIEWED",
                        "TEMPORAL_UPDATE (Toll Revised Over Time)",
                        "GEOGRAPHIC_VARIATION (Distinct Sub-Localities)",
                        "TRUE_CONTRADICTION (Conflicting Ground Truth)",
                    ],
                    index=[
                        "UNREVIEWED",
                        "TEMPORAL_UPDATE (Toll Revised Over Time)",
                        "GEOGRAPHIC_VARIATION (Distinct Sub-Localities)",
                        "TRUE_CONTRADICTION (Conflicting Ground Truth)",
                    ].index(saved_status) if saved_status in [
                        "UNREVIEWED",
                        "TEMPORAL_UPDATE (Toll Revised Over Time)",
                        "GEOGRAPHIC_VARIATION (Distinct Sub-Localities)",
                        "TRUE_CONTRADICTION (Conflicting Ground Truth)",
                    ] else 0,
                    key=f"status_sel_{pair_key}",
                )

                note = rev_c2.text_input("Analyst Notes", value=analyst_reviews.get("conflicts", {}).get(pair_key, {}).get("notes", ""), key=f"note_{pair_key}")

                if rev_c3.button("Save Review", key=f"btn_save_{pair_key}", use_container_width=True):
                    utils.save_analyst_decision("conflicts", pair_key, new_status, note)
                    st.success("Decision recorded to local audit log.")

                st.markdown("<hr style='border-color: #1E293B; margin: 16px 0;'>", unsafe_allow_html=True)

    # SUBTAB 2: Candidate Similar Reports
    with tab_duplicates:
        if pairs_df is None or pairs_df.empty:
            st.info("Similarity dataset not found. Run `python group.py` and `python inspect_pairs.py`.")
        else:
            pdf = pairs_df.copy()
            sim_col = "similarity_numeric" if "similarity_numeric" in pdf.columns else "similarity"

            s_col1, s_col2 = st.columns(2)
            min_sim_thresh = s_col1.slider("Minimum Similarity Threshold", 0.65, 1.0, 0.75, 0.01)
            pdf = pdf[pdf[sim_col].astype(float) >= min_sim_thresh]

            st.caption(f"Showing candidate similar pairs matching threshold ({len(pdf):,} pairs).")

            preview_cols = [c for c in ["similarity", "event_name", "category_a", "category_b", "tweet_id_a", "tweet_id_b", "tweet_text_a", "tweet_text_b"] if c in pdf.columns]
            st.dataframe(pdf.head(80)[preview_cols], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# MODULE 3: 🔍 HISTORICAL REPORT INTELLIGENCE
# -----------------------------------------------------------------------------

elif app_mode == "🔍 Historical Report Intelligence":
    st.title("🔍 Historical Report Intelligence")
    st.caption("Search, filter, and inspect processed HumAID disaster reports with bounded pagination.")

    if reports_df is None:
        st.warning("Processed reports dataset not found. Run `python extract.py`.")
    else:
        rdf = reports_df.copy()

        # Top Filters
        fc1, fc2, fc3 = st.columns(3)
        ev_list = sorted(rdf["event_name"].unique()) if "event_name" in rdf.columns else []
        sel_evs = fc1.multiselect("Filter by Disaster Event", ev_list)

        cat_list = sorted(rdf["humanitarian_category"].unique()) if "humanitarian_category" in rdf.columns else []
        sel_cats = fc2.multiselect("Filter by Category", cat_list)

        txt_query = fc3.text_input("Search Text or ID", placeholder="e.g. trapped, medical, flood...")

        if sel_evs:
            rdf = rdf[rdf["event_name"].isin(sel_evs)]
        if sel_cats:
            rdf = rdf[rdf["humanitarian_category"].isin(sel_cats)]
        if txt_query:
            q = txt_query.strip().lower()
            mask = pd.Series(False, index=rdf.index)
            for c in ["tweet_text", "clean_text", "tweet_id"]:
                if c in rdf.columns:
                    mask |= rdf[c].astype(str).str.lower().str.contains(q, regex=False)
            rdf = rdf[mask]

        total_matching = len(rdf)

        # Bounded Pagination
        pg_c1, pg_c2 = st.columns([3, 1])
        page_sz = 50
        max_p = max(1, (total_matching + page_sz - 1) // page_sz)
        curr_p = pg_c2.number_input("Page", min_value=1, max_value=max_p, value=1, step=1)

        st_idx = (curr_p - 1) * page_sz
        end_idx = min(st_idx + page_sz, total_matching)
        pg_c1.markdown(f"**Matching Reports:** `{total_matching:,}` | Showing `{st_idx + 1}` to `{end_idx}` (Page `{curr_p}` of `{max_p}`)")

        disp_cols = [c for c in ["tweet_id", "event_name", "humanitarian_category", "priority_score", "priority_level", "tweet_text"] if c in rdf.columns]
        st.dataframe(rdf.iloc[st_idx:end_idx][disp_cols], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# MODULE 4: 🛡️ SYSTEM HEALTH & DIAGNOSTICS
# -----------------------------------------------------------------------------

elif app_mode == "🛡️ System Health & Diagnostics":
    st.title("🛡️ System Health & Diagnostics")
    st.caption("Verifies pipeline data integrity, schema compliance, runtime versions, and execution commands.")

    health_records = utils.inspect_pipeline_health()
    hdf = pd.DataFrame(health_records)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline Assets", len(hdf))
    c2.metric("Operational (Healthy)", (hdf["status"] == "HEALTHY").sum())
    c3.metric("Optimization Engine", "MILP (scipy.optimize.milp)")
    c4.metric("3D Graphics Engine", "WebGL + Three.js / Plotly")

    st.markdown("---")
    st.markdown("#### Pipeline Asset Audit Matrix")
    disp_h = hdf[["name", "type", "status", "rows", "size_kb", "modified", "script"]].copy()
    disp_h.columns = ["Asset", "Pipeline Stage", "Status", "Rows", "Size (KB)", "Last Modified", "Script"]
    st.dataframe(disp_h, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Execution Guide")
    st.markdown(
        """
        - **Run MILP Allocator:** `python allocator.py --solver milp`
        - **Run Greedy Baseline:** `python allocator.py --solver greedy`
        - **Run Unit Test Suite:** `pytest -v`
        - **Start Dashboard:** `streamlit run app.py`
        """
    )


# -----------------------------------------------------------------------------
# GLOBAL FOOTER
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #475569; font-size: 11px; padding: 10px;">
        RESCUEGRID 3D • Tactical Emergency Intelligence & Resource Optimization Engine • NEXAI 2026<br>
        Mixed-Integer Linear Programming • Evidence-Aware Reconciliation • Human Commander Review Required
    </div>
    """,
    unsafe_allow_html=True,
)
