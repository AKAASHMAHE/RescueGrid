"""
RESCUEGRID - Evidence-Aware Emergency Intelligence Dashboard
============================================================
Emergency Operations Intelligence Platform for Disaster Social Media Analysis,
Similarity Grouping, Potential Conflict Review, Priority Scoring, and Resource Allocation.

Decision-Support Prototype. All conflict signals and priority scores are heuristic.
All resource inventory and incident needs in the allocator are synthetic demo inputs.
Human approval is required for all actions.
"""

from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import centralized pipeline utilities
import pipeline_utils as utils
import allocator

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & THEME CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="RESCUEGRID | Emergency Intelligence Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Emergency Intelligence Dark Theme CSS
CUSTOM_CSS = """
<style>
    /* Dark Navy & Slate Command Center Theme */
    .stApp {
        background-color: #0B1120;
        color: #F1F5F9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Header & Titles */
    h1, h2, h3, h4 {
        color: #F8FAFC !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Top Banner / Cards */
    .emergency-banner {
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
        border-left: 4px solid #EF4444;
        padding: 14px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    
    .emergency-banner-title {
        font-size: 15px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .emergency-banner-subtitle {
        font-size: 13px;
        color: #94A3B8;
        line-height: 1.4;
    }
    
    /* KPI Card Component */
    .kpi-card {
        background-color: #131E32;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .kpi-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 6px;
    }
    
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    
    .kpi-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
    }
    
    .badge-data { background-color: #0369A1; color: #E0F2FE; }
    .badge-rule { background-color: #B45309; color: #FEF3C7; }
    .badge-model { background-color: #6D28D9; color: #EDE9FE; }
    .badge-demo { background-color: #BE123C; color: #FFE4E6; }
    .badge-safe { background-color: #047857; color: #D1FAE5; }

    /* Dossier Card for Comparisons */
    .dossier-card {
        background-color: #131E32;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    /* Subdued Divider */
    hr {
        border-color: #1E293B;
        margin: 24px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA INITIALIZATION & CACHED INGESTION
# -----------------------------------------------------------------------------

reports_df, reports_source = utils.load_reports_dataset()
pairs_df, pairs_source = utils.load_similarity_dataset()
conflicts_df = utils.load_conflicts_dataset()
allocation_bundle = utils.load_allocation_data()

allocations_df = allocation_bundle["results"]
shortages_df = allocation_bundle["unmet"]
inventory_df = allocation_bundle["inventory"]
incidents_df = allocation_bundle["incidents"]

# Session State for Human Review
if "similar_reviews" not in st.session_state:
    st.session_state["similar_reviews"] = {}

if "conflict_reviews" not in st.session_state:
    st.session_state["conflict_reviews"] = {}


# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & PLATFORM CONTROLS
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🛡️ RESCUEGRID")
    st.caption("Evidence-Aware Emergency Intelligence • v2.4")

    st.markdown(
        """
        <div style="background-color: #1E293B; padding: 10px; border-radius: 6px; border-left: 3px solid #EF4444; margin-bottom: 15px;">
            <div style="font-size: 11px; font-weight: 700; color: #EF4444;">OPERATIONAL ADVISORY</div>
            <div style="font-size: 11px; color: #CBD5E1; margin-top: 4px;">
                Human-in-the-loop decision support. Heuristic conflict signals & synthetic demo inventory.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Operations Center",
        [
            "🛰️ Executive Overview",
            "🔍 Report Explorer",
            "🔗 Similarity & Duplicates",
            "⚖️ Conflict Review",
            "📈 Priority Analytics",
            "📦 Resource Allocation",
            "🛡️ Pipeline Health",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### Platform Readiness")
    st.markdown(f"- **Reports Data:** {'✅ Ready' if reports_df is not None else '⏳ Pending'}")
    st.markdown(f"- **Similarity Pairs:** {'✅ Ready' if pairs_df is not None else '⏳ Pending'}")
    st.markdown(f"- **Conflict Flags:** {'✅ Ready' if conflicts_df is not None else '⏳ Pending'}")
    st.markdown(f"- **Resource Allocator:** {'✅ Ready' if allocations_df is not None else '⏳ Pending'}")

    st.markdown("---")
    st.caption("RESCUEGRID Operations • Multi-Disaster Historical Intelligence")


# Helper for rendering standardized KPI card
def render_kpi(col, title: str, value: str, badge_text: str, badge_class: str):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <span class="kpi-badge {badge_class}">{badge_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 1: 🛰️ EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------

if page == "🛰️ Executive Overview":
    st.title("🛰️ Operational Executive Overview")
    st.caption("High-level multi-disaster situational intelligence, pipeline coverage, and resource readiness.")

    # Top advisory banner
    st.markdown(
        """
        <div class="emergency-banner">
            <div class="emergency-banner-title">
                <span>⚠️</span> SYSTEM TRUTH & EVIDENCE ATTRIBUTION
            </div>
            <div class="emergency-banner-subtitle">
                All statistics below are computed directly from available project datasets.
                Metrics distinguish <b>Data-Observed records</b> (HumAID disaster tweets),
                <b>Rule-Based signals</b> (conflict checks and priority scores),
                <b>Model-Derived similarities</b> (TF-IDF nearest neighbors), and
                <b>Synthetic Demonstration data</b> (inventory & simulated needs).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute genuine KPIs
    total_reports = len(reports_df) if reports_df is not None else 0
    total_events = reports_df["event_name"].nunique() if reports_df is not None and "event_name" in reports_df.columns else 0
    total_pairs = len(pairs_df) if pairs_df is not None else 0
    total_conflicts = len(conflicts_df) if conflicts_df is not None else 0
    total_incidents = len(allocations_df) if allocations_df is not None else 0

    unmet_count = 0
    if shortages_df is not None and not shortages_df.empty and "incident_id" in shortages_df.columns:
        unmet_count = shortages_df["incident_id"].nunique()

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    render_kpi(k1, "Reports Ingested", f"{total_reports:,}" if total_reports else "Not available yet", "Data-Observed", "badge-data")
    render_kpi(k2, "Disaster Events", f"{total_events}" if total_events else "Not available yet", "Data-Observed", "badge-data")
    render_kpi(k3, "Candidate Pairs", f"{total_pairs:,}" if total_pairs else "Not available yet", "Model-Derived", "badge-model")
    render_kpi(k4, "Potential Conflicts", f"{total_conflicts}" if total_conflicts else "Not available yet", "Rule-Based Signal", "badge-rule")
    render_kpi(k5, "Shortage Incidents", f"{unmet_count}/{total_incidents}" if total_incidents else "Not available yet", "Synthetic Demo", "badge-demo")

    st.markdown("---")

    # Visual Analytics Grid
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Humanitarian Information Categories")
        if reports_df is not None and "humanitarian_category" in reports_df.columns:
            cat_counts = (
                reports_df["humanitarian_category"]
                .replace("", "unmapped")
                .value_counts()
                .reset_index()
            )
            cat_counts.columns = ["Category", "Reports"]

            fig_cat = px.bar(
                cat_counts.sort_values("Reports", ascending=True),
                x="Reports",
                y="Category",
                orientation="h",
                color="Reports",
                color_continuous_scale="Blues",
                title="Reports by Humanitarian Category (Data-Derived)",
            )
            fig_cat.update_layout(
                paper_bgcolor="#0B1120",
                plot_bgcolor="#131E32",
                font_color="#CBD5E1",
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Humanitarian categories not available yet. Run `python extract.py`.")

    with col_right:
        st.subheader("Disaster Event Distribution")
        if reports_df is not None and "event_name" in reports_df.columns:
            event_counts = (
                reports_df["event_name"]
                .replace("", "unknown")
                .value_counts()
                .reset_index()
            )
            event_counts.columns = ["Event", "Reports"]

            fig_event = px.bar(
                event_counts.sort_values("Reports", ascending=True),
                x="Reports",
                y="Event",
                orientation="h",
                color="Reports",
                color_continuous_scale="Teal",
                title="Report Volume by Disaster Event (Data-Derived)",
            )
            fig_event.update_layout(
                paper_bgcolor="#0B1120",
                plot_bgcolor="#131E32",
                font_color="#CBD5E1",
                height=380,
                margin=dict(l=10, r=10, t=40, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_event, use_container_width=True)
        else:
            st.info("Event data not available yet.")

    st.markdown("---")

    # Lower Grid: Triage Status and Module Verification
    col_triage, col_pipeline = st.columns([1, 1])

    with col_triage:
        st.subheader("Triage Review & Priority Levels")
        hint_col = "priority_level" if (reports_df is not None and "priority_level" in reports_df.columns) else "review_hint"
        if reports_df is not None and hint_col in reports_df.columns:
            triage_counts = reports_df[hint_col].value_counts().reset_index()
            triage_counts.columns = ["Status", "Count"]

            fig_triage = px.pie(
                triage_counts,
                names="Status",
                values="Count",
                hole=0.45,
                color_discrete_sequence=["#EF4444", "#F59E0B", "#0284C7", "#10B981", "#64748B"],
                title=f"Distribution of {hint_col.replace('_', ' ').title()} (Rule-Based)",
            )
            fig_triage.update_layout(
                paper_bgcolor="#0B1120",
                font_color="#CBD5E1",
                height=340,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_triage, use_container_width=True)
            st.caption("Rule-based review priority hints are configurable indicators, not ground-truth severity.")
        else:
            st.info("Review hints / priority levels not available yet.")

    with col_pipeline:
        st.subheader("Pipeline Module Readiness")
        health_records = utils.inspect_pipeline_health()
        health_df = pd.DataFrame(health_records)[["name", "type", "rows", "status", "modified"]]
        health_df.columns = ["Module Output", "Stage", "Rows", "Health", "Last Modified"]
        st.dataframe(health_df, use_container_width=True, hide_index=True, height=340)

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 13px; color: #94A3B8; line-height: 1.6;">
            <b>Operational Guardrails Summary:</b><br>
            1. Text reports reflect historical social media submissions and may be unconfirmed.<br>
            2. High similarity scores identify candidate duplicates for deduplication review, not validated identical incidents.<br>
            3. Discrepancy signals flag numerical or status divergence for analyst verification; they do not automatically denote misinformation.<br>
            4. Allocator recommendations use demo inventories and a greedy baseline. Real-world dispatch is never triggered.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 2: 🔍 REPORT EXPLORER
# -----------------------------------------------------------------------------

elif page == "🔍 Report Explorer":
    st.title("🔍 Emergency Report Explorer")
    st.caption("Search, filter, and inspect disaster reports with bounded memory pagination and detailed dossier views.")

    if reports_df is None:
        st.warning("Processed reports dataset is not available yet. Please run `python extract.py`.")
    else:
        df = reports_df.copy()

        # Top Filter Controls
        with st.expander("Filter & Search Controls", expanded=True):
            f1, f2, f3 = st.columns(3)

            # Event filter
            events = sorted(x for x in df["event_name"].unique() if x) if "event_name" in df.columns else []
            sel_events = f1.multiselect("Disaster Event", events)

            # Category filter
            categories = sorted(x for x in df["humanitarian_category"].unique() if x) if "humanitarian_category" in df.columns else []
            sel_cats = f2.multiselect("Humanitarian Category", categories)

            # Priority / Hint filter
            filter_hint_col = "priority_level" if "priority_level" in df.columns else "review_hint"
            hints = sorted(x for x in df[filter_hint_col].unique() if x) if filter_hint_col in df.columns else []
            sel_hints = f3.multiselect(f"Review Level ({filter_hint_col})", hints)

            # Search bar
            search_query = st.text_input("Text or Tweet ID Search (case-insensitive)", placeholder="e.g., stranded, bridge collapsed, 903371...")

        # Apply Filters
        if sel_events:
            df = df[df["event_name"].isin(sel_events)]
        if sel_cats:
            df = df[df["humanitarian_category"].isin(sel_cats)]
        if sel_hints:
            df = df[df[filter_hint_col].isin(sel_hints)]

        if search_query:
            q = search_query.strip()
            mask = pd.Series(False, index=df.index)
            for c in ("tweet_text", "clean_text", "tweet_id"):
                if c in df.columns:
                    mask |= df[c].astype(str).str.contains(q, case=False, regex=False, na=False)
            df = df[mask]

        total_matched = len(df)

        # Pagination & Bounded Rendering
        p1, p2, p3 = st.columns([2, 1, 1])
        page_size = p2.selectbox("Page Size", [25, 50, 100, 200], index=0)
        total_pages = max(1, (total_matched + page_size - 1) // page_size)
        curr_page = p3.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

        start_idx = (curr_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_matched)

        p1.markdown(f"**Matched Reports:** `{total_matched:,}` | Showing items `{start_idx + 1}` to `{end_idx}` (Page `{curr_page}` of `{total_pages}`)")

        # Prepare Display Table
        display_cols = [
            c for c in [
                "tweet_id", "event_name", "humanitarian_category", "priority_score",
                "priority_level", "review_hint", "need_indicators", "tweet_text"
            ] if c in df.columns
        ]

        page_slice = df.iloc[start_idx:end_idx][display_cols]
        st.dataframe(page_slice, use_container_width=True, hide_index=True, height=400)

        # Detailed Report Inspector
        st.markdown("---")
        st.subheader("Detailed Report Inspector")

        inspector_ids = page_slice["tweet_id"].tolist() if "tweet_id" in page_slice.columns else []
        if inspector_ids:
            selected_id = st.selectbox("Select Report to Inspect from Current Page", inspector_ids)
            report_match = df[df["tweet_id"] == selected_id]

            if not report_match.empty:
                rep = report_match.iloc[0]

                st.markdown(
                    f"""
                    <div class="dossier-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-size: 16px; font-weight: 700; color: #F8FAFC;">REPORT DOSSIER: #{rep.get('tweet_id', 'N/A')}</span>
                            <span class="kpi-badge badge-data">Event: {rep.get('event_name', 'Unknown')}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("#### Text Representations")
                    st.markdown("**Original Raw Text:**")
                    st.info(rep.get("tweet_text", "N/A"))

                    st.markdown("**Cleaned Formatted Text:**")
                    st.text(rep.get("clean_text", "N/A"))

                    st.markdown("**Normalized Tokens:**")
                    st.code(rep.get("normalized_text", "N/A"))

                with d2:
                    st.markdown("#### Classification & Priority Attribution")
                    st.markdown(f"- **Humanitarian Category:** `{rep.get('humanitarian_category', 'N/A')}` *(Original: {rep.get('original_label', 'N/A')})*")
                    st.markdown(f"- **Review Hint:** `{rep.get('review_hint', 'N/A')}`")
                    st.markdown(f"- **Need Indicators:** `{rep.get('need_indicators', 'None detected')}`")

                    if "priority_score" in rep.index:
                        st.markdown(f"- **Priority Score:** `{rep.get('priority_score', 'N/A')} / 100`")
                        st.markdown(f"- **Priority Level:** `{rep.get('priority_level', 'N/A')}`")
                        st.markdown(f"- **Scoring Factors:** `{rep.get('priority_reasons', 'N/A')}`")
                        if "indicator_details" in rep.index and rep.get("indicator_details"):
                            st.markdown(f"- **Matched Indicator Phrases:** `{rep.get('indicator_details')}`")
                    st.caption("Source label is observed from HumAID. Priority score is a rule-based heuristic.")


# -----------------------------------------------------------------------------
# PAGE 3: 🔗 SIMILARITY & DUPLICATE CANDIDATES
# -----------------------------------------------------------------------------

elif page == "🔗 Similarity & Duplicates":
    st.title("🔗 Similar Report Pairs & Duplicate Candidates")
    st.caption("Inspect TF-IDF cosine similarity candidates, evaluate repetition, and conduct human duplicate verification.")

    if pairs_df is None:
        st.warning("Similarity pairs dataset not found. Please run `python group.py` and `python inspect_pairs.py`.")
    else:
        df = pairs_df.copy()

        # Top threshold slider & filter
        c1, c2, c3 = st.columns([2, 1, 1])
        min_sim = c1.slider("Cosine Similarity Threshold", min_value=0.65, max_value=1.0, value=0.70, step=0.01)

        event_col = "event_name" if "event_name" in df.columns else ("event_a" if "event_a" in df.columns else None)
        events = sorted(df[event_col].dropna().unique()) if event_col else []
        sel_event = c2.selectbox("Filter Event", ["All Events"] + events)

        # Filter dataset
        sim_col = "similarity_numeric" if "similarity_numeric" in df.columns else "similarity"
        df = df[df[sim_col].astype(float) >= min_sim]
        if sel_event != "All Events" and event_col:
            df = df[df[event_col] == sel_event]

        c3.metric("Candidate Pairs", f"{len(df):,}")

        # Similarity Score Distribution Chart
        st.markdown("#### Similarity Score Distribution")
        fig_sim = px.histogram(
            df,
            x=sim_col,
            nbins=35,
            color_discrete_sequence=["#0284C7"],
            title="Distribution of Cosine Similarity Across Candidate Pairs",
        )
        fig_sim.update_layout(
            paper_bgcolor="#0B1120",
            plot_bgcolor="#131E32",
            font_color="#CBD5E1",
            height=260,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis_title="Similarity Score",
            yaxis_title="Pair Count",
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        st.markdown("---")
        st.subheader("Side-by-Side Candidate Pair Comparison")

        if df.empty:
            st.info("No report pairs match the selected similarity threshold and filters.")
        else:
            # Dropdown selector for pair inspection
            pair_options = [
                f"Pair #{i+1}: Sim {float(row[sim_col]):.3f} | [{row.get('tweet_id_a', 'A')[:8]}... vs {row.get('tweet_id_b', 'B')[:8]}...]"
                for i, (_, row) in enumerate(df.head(200).iterrows())
            ]
            sel_pair_idx = st.selectbox("Select Candidate Pair to Compare", range(len(pair_options)), format_func=lambda i: pair_options[i])
            chosen_pair = df.iloc[sel_pair_idx]

            id_a = chosen_pair.get("tweet_id_a", "N/A")
            id_b = chosen_pair.get("tweet_id_b", "N/A")
            score_val = float(chosen_pair.get(sim_col, 0))

            # Header info
            st.markdown(
                f"""
                <div class="emergency-banner" style="border-left-color: #0284C7;">
                    <div class="emergency-banner-title">
                        <span>🔍</span> CANDIDATE PAIR COMPARISON — SIMILARITY SCORE: {score_val:.4f}
                    </div>
                    <div class="emergency-banner-subtitle">
                        <b>Important:</b> Text similarity is a candidate signal. Similar reports may describe the same incident,
                        quote the same news wire, or describe distinct events with similar wording.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Two-column comparison
            comp_a, comp_b = st.columns(2)
            with comp_a:
                st.markdown(
                    f"""
                    <div class="dossier-card">
                        <div class="kpi-title">REPORT A (ID: {id_a})</div>
                        <div style="font-size: 13px; color: #38BDF8; margin-bottom: 8px;">
                            Category: {chosen_pair.get('category_a', 'N/A')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info(chosen_pair.get("tweet_text_a", "Text unavailable"))

            with comp_b:
                st.markdown(
                    f"""
                    <div class="dossier-card">
                        <div class="kpi-title">REPORT B (ID: {id_b})</div>
                        <div style="font-size: 13px; color: #38BDF8; margin-bottom: 8px;">
                            Category: {chosen_pair.get('category_b', 'N/A')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info(chosen_pair.get("tweet_text_b", "Text unavailable"))

            # Analyst Decision Interface
            st.markdown("#### Analyst Review Decision")
            pair_key = f"{id_a}_{id_b}"
            current_status = st.session_state["similar_reviews"].get(pair_key, "UNREVIEWED")

            r_col1, r_col2 = st.columns([2, 1])
            new_decision = r_col1.radio(
                "Human Verification Status",
                ["UNREVIEWED", "POTENTIAL_DUPLICATE_REPORT", "DISTINCT_INCIDENTS", "NEEDS_GROUND_VERIFICATION"],
                index=["UNREVIEWED", "POTENTIAL_DUPLICATE_REPORT", "DISTINCT_INCIDENTS", "NEEDS_GROUND_VERIFICATION"].index(current_status),
                horizontal=True,
            )

            if r_col2.button("Save Analyst Decision", use_container_width=True):
                st.session_state["similar_reviews"][pair_key] = new_decision
                st.success(f"Saved review for Pair {id_a} / {id_b}: `{new_decision}`")

            # Table preview
            st.markdown("---")
            st.markdown("#### Bounded Candidate Pairs Table")
            preview_cols = [c for c in ["similarity", "event_name", "category_a", "category_b", "tweet_id_a", "tweet_id_b", "tweet_text_a", "tweet_text_b"] if c in df.columns]
            st.dataframe(df.head(100)[preview_cols], use_container_width=True, hide_index=True, height=350)


# -----------------------------------------------------------------------------
# PAGE 4: ⚖️ CONFLICT REVIEW
# -----------------------------------------------------------------------------

elif page == "⚖️ Conflict Review":
    st.title("⚖️ Evidence-Aware Conflict Review")
    st.caption("Inspect rule-based opposing claims and metric discrepancies across similar reports.")

    # Prominent disclaimer
    st.markdown(
        """
        <div class="emergency-banner">
            <div class="emergency-banner-title">
                <span>⚠️</span> POTENTIAL CONFLICT SIGNALS — HUMAN REVIEW REQUIRED
            </div>
            <div class="emergency-banner-subtitle">
                Conflict flags are generated by transparent, rule-based heuristics comparing similar reports
                (e.g., opposing keywords like 'road open' vs 'road closed', or differing numbers for casualty/injury counts).
                <b>A flag is an alert for human triage, not proof of contradiction or misinformation.</b>
                Reports may describe different times, revised estimates, or distinct micro-locations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if conflicts_df is None or conflicts_df.empty:
        st.info("No conflict flags detected or `potential_conflicts.tsv` not found. Run `python conflicts.py`.")
    else:
        df = conflicts_df.copy()

        # Filters
        f1, f2 = st.columns([1, 1])
        all_signals = sorted({sig.strip() for val in df["conflict_signals"] for sig in str(val).split("|") if sig.strip()}) if "conflict_signals" in df.columns else []
        sel_sig = f1.selectbox("Filter by Conflict Signal Type", ["All Signals"] + all_signals)

        events = sorted(df["event_name"].unique()) if "event_name" in df.columns else []
        sel_evt = f2.selectbox("Filter by Event", ["All Events"] + events)

        if sel_sig != "All Signals":
            df = df[df["conflict_signals"].astype(str).str.contains(sel_sig, regex=False)]
        if sel_evt != "All Events":
            df = df[df["event_name"] == sel_evt]

        st.markdown(f"**Flagged Conflict Pairs:** `{len(df)}`")

        # Dossier view of each flagged pair
        for idx, row in df.iterrows():
            sim_val = float(row.get("similarity", 0.0))
            raw_sig = row.get("conflict_signals", "DISCREPANCY")
            parsed_signals = utils.parse_conflict_signals(raw_sig)

            st.markdown(
                f"""
                <div class="dossier-card" style="border-left: 4px solid #EF4444;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div>
                            <span style="font-weight: 700; color: #F8FAFC; font-size: 15px;">PAIR #{idx + 1} | {row.get('event_name', 'Unknown Event')}</span>
                            <span class="kpi-badge badge-rule" style="margin-left: 8px;">Similarity: {sim_val:.3f}</span>
                        </div>
                        <span class="kpi-badge badge-demo">POTENTIAL CONFLICT SIGNAL — NOT VERIFIED</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Signal summary tags
            for ps in parsed_signals:
                st.markdown(f"🚩 **Conflict Reason:** `{ps['summary']}`")

            t_a, t_b = st.columns(2)
            with t_a:
                st.markdown(f"**Report A (ID: {row.get('tweet_id_a', 'N/A')})** | Category: `{row.get('category_a', 'N/A')}`")
                st.info(row.get("tweet_text_a", "Text unavailable"))

            with t_b:
                st.markdown(f"**Report B (ID: {row.get('tweet_id_b', 'N/A')})** | Category: `{row.get('category_b', 'N/A')}`")
                st.info(row.get("tweet_text_b", "Text unavailable"))

            # Review status
            rev_key = f"conflict_{row.get('tweet_id_a')}_{row.get('tweet_id_b')}"
            existing_rev = st.session_state["conflict_reviews"].get(rev_key, "PENDING_ANALYST_TRIAGE")

            rev_col1, rev_col2 = st.columns([3, 1])
            new_rev = rev_col1.radio(
                f"Triage Assessment for Pair #{idx + 1}",
                [
                    "PENDING_ANALYST_TRIAGE",
                    "TEMPORAL_UPDATE (Toll/Status Revised Over Time)",
                    "GEOGRAPHIC_VARIATION (Different Localities)",
                    "AUTHENTIC_CONTRADICTION (Further Investigation Required)",
                ],
                index=[
                    "PENDING_ANALYST_TRIAGE",
                    "TEMPORAL_UPDATE (Toll/Status Revised Over Time)",
                    "GEOGRAPHIC_VARIATION (Different Localities)",
                    "AUTHENTIC_CONTRADICTION (Further Investigation Required)",
                ].index(existing_rev),
                key=f"radio_{rev_key}",
                horizontal=True,
            )

            if rev_col2.button("Record Triage", key=f"btn_{rev_key}", use_container_width=True):
                st.session_state["conflict_reviews"][rev_key] = new_rev
                st.success(f"Recorded triage status: {new_rev}")

            st.markdown("<hr style='margin: 16px 0; border-color: #1E293B;'>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 5: 📈 PRIORITY ANALYTICS
# -----------------------------------------------------------------------------

elif page == "📈 Priority Analytics":
    st.title("📈 Emergency Report Priority Analytics")
    st.caption("Inspect transparent rule-based review priorities, urgency indicator triggers, and queue distributions.")

    # Explanation of priority score factors
    st.markdown(
        """
        <div class="emergency-banner" style="border-left-color: #F59E0B;">
            <div class="emergency-banner-title">
                <span>ℹ️</span> SCORING METHODOLOGY & LIMITATIONS
            </div>
            <div class="emergency-banner-subtitle">
                Priority scores [0–100] are calculated using transparent heuristics:
                <b>Category Base Score</b> (Urgent Needs: 45, Casualties: 40, Missing People: 45, Evacuation: 35, etc.)
                plus <b>Urgency Indicator Regex Boosts</b> (Immediate Danger: +35, Urgent Request: +25, Medical Emergency: +25, Evacuation Need: +15, Essential Supplies: +10).
                Scores are capped at 100.
                <b>This is a triage ranking for review queues, NOT a validated physical severity estimate or automatic dispatch order.</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if reports_df is None or "priority_score" not in reports_df.columns:
        st.info("Priority scored dataset not found (`priority_reports.tsv`). Please run `python priority.py`.")
    else:
        df = reports_df.copy()
        df["priority_score_num"] = pd.to_numeric(df["priority_score"], errors="coerce").fillna(0)

        # Overview Score Distribution
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Scored Reports", f"{len(df):,}")
        m2.metric("Critical Review Queue (>=70)", f"{(df['priority_score_num'] >= 70).sum():,}")
        m3.metric("High Review Queue (45-69)", f"{((df['priority_score_num'] >= 45) & (df['priority_score_num'] < 70)).sum():,}")
        m4.metric("Mean Priority Score", f"{df['priority_score_num'].mean():.1f} / 100")

        st.markdown("---")
        g1, g2 = st.columns([1, 1])

        with g1:
            st.subheader("Priority Score Histogram")
            fig_hist = px.histogram(
                df,
                x="priority_score_num",
                nbins=25,
                color_discrete_sequence=["#EF4444"],
                title="Report Distribution Across Score Spectrum",
            )
            fig_hist.update_layout(
                paper_bgcolor="#0B1120",
                plot_bgcolor="#131E32",
                font_color="#CBD5E1",
                height=340,
                xaxis_title="Rule-Based Priority Score",
                yaxis_title="Report Volume",
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with g2:
            st.subheader("Priority Level Breakdown")
            if "priority_level" in df.columns:
                level_order = ["CRITICAL_REVIEW", "HIGH_REVIEW", "MEDIUM_REVIEW", "LOW_REVIEW", "INFORMATIONAL"]
                lvl_counts = df["priority_level"].value_counts().reindex(level_order, fill_value=0).reset_index()
                lvl_counts.columns = ["Priority Level", "Count"]

                fig_lvl = px.bar(
                    lvl_counts,
                    x="Priority Level",
                    y="Count",
                    color="Priority Level",
                    color_discrete_map={
                        "CRITICAL_REVIEW": "#EF4444",
                        "HIGH_REVIEW": "#F59E0B",
                        "MEDIUM_REVIEW": "#0284C7",
                        "LOW_REVIEW": "#10B981",
                        "INFORMATIONAL": "#64748B",
                    },
                    title="Report Volume by Review Band",
                )
                fig_lvl.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#131E32",
                    font_color="#CBD5E1",
                    height=340,
                    margin=dict(l=10, r=10, t=35, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_lvl, use_container_width=True)

        st.markdown("---")
        st.subheader("High-Priority Emergency Review Queue")

        # Filters for high priority reports
        p_filter_col1, p_filter_col2 = st.columns(2)
        events = sorted(df["event_name"].unique()) if "event_name" in df.columns else []
        sel_ev = p_filter_col1.selectbox("Filter Queue by Disaster Event", ["All Events"] + events)

        min_score = p_filter_col2.slider("Minimum Priority Score for Queue", min_value=0, max_value=100, value=70, step=5)

        queue_df = df[df["priority_score_num"] >= min_score]
        if sel_ev != "All Events":
            queue_df = queue_df[queue_df["event_name"] == sel_ev]

        st.markdown(f"**Reports in Queue:** `{len(queue_df):,}`")

        queue_display_cols = [
            c for c in [
                "priority_score", "priority_level", "event_name", "humanitarian_category",
                "priority_reasons", "indicator_details", "tweet_text"
            ] if c in queue_df.columns
        ]

        st.dataframe(queue_df.head(100)[queue_display_cols], use_container_width=True, hide_index=True, height=380)


# -----------------------------------------------------------------------------
# PAGE 6: 📦 RESOURCE ALLOCATION
# -----------------------------------------------------------------------------

elif page == "📦 Resource Allocation":
    st.title("📦 Evidence-Aware Resource Allocation Engine")
    st.caption("Greedy priority-first resource allocation baseline with interactive scenario simulation.")

    # Prominent synthetic notice
    st.markdown(
        """
        <div class="emergency-banner" style="border-left-color: #EF4444;">
            <div class="emergency-banner-title">
                <span>🚨</span> SYNTHETIC DEMONSTRATION DATA NOTICE
            </div>
            <div class="emergency-banner-subtitle">
                All inventory quantities, incident resource requests, and priority scores in this module are <b>synthetic demo inputs</b>.
                The allocation algorithm is a transparent <b>greedy priority-first baseline</b> (it fulfills requests in descending order of priority score
                until stock is depleted). It is not guaranteed to be globally optimal.
                <b>All recommendations require human commander review. Real resource dispatch is never executed.</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_static, tab_simulator = st.tabs(["📋 Current Baseline Allocation", "🎛️ Scenario Simulator & Sandbox"])

    # TAB 1: Baseline Static Results
    with tab_static:
        st.subheader("Current Baseline Allocation Results")

        if allocations_df is None or allocations_df.empty:
            st.info("Allocation results not available yet. Please run `python allocator.py`.")
        else:
            # Metrics
            total_incidents = len(allocations_df)
            shortage_incidents = shortages_df["incident_id"].nunique() if shortages_df is not None and not shortages_df.empty else 0

            b1, b2, b3 = st.columns(3)
            b1.metric("Incidents Considered", f"{total_incidents}")
            b2.metric("Incidents with Unmet Needs", f"{shortage_incidents}")
            b3.metric("Baseline Solver Heuristic", "Greedy Priority-First")

            st.markdown("#### Baseline Incident Assignments")
            st.dataframe(allocations_df, use_container_width=True, hide_index=True)

            if shortages_df is not None and not shortages_df.empty:
                st.markdown("#### Unmet Resource Shortages")
                sh_df = shortages_df.copy()
                sh_df["unmet_numeric"] = pd.to_numeric(sh_df["unmet"], errors="coerce").fillna(0)

                sh_chart_col, sh_table_col = st.columns([1, 1])

                with sh_chart_col:
                    shortage_summary = sh_df.groupby("resource_type")["unmet_numeric"].sum().reset_index()
                    fig_shortage = px.bar(
                        shortage_summary,
                        x="resource_type",
                        y="unmet_numeric",
                        color="resource_type",
                        color_discrete_sequence=["#EF4444", "#F59E0B", "#0284C7"],
                        title="Unmet Deficit by Resource Category",
                    )
                    fig_shortage.update_layout(
                        paper_bgcolor="#0B1120",
                        plot_bgcolor="#131E32",
                        font_color="#CBD5E1",
                        height=280,
                        margin=dict(l=10, r=10, t=35, b=10),
                        showlegend=False,
                        xaxis_title="Resource Type",
                        yaxis_title="Unfulfilled Units",
                    )
                    st.plotly_chart(fig_shortage, use_container_width=True)

                with sh_table_col:
                    st.dataframe(sh_df[["incident_id", "priority_score", "resource_type", "requested", "allocated", "unmet"]], use_container_width=True, hide_index=True, height=280)

    # TAB 2: Interactive Scenario Simulator
    with tab_simulator:
        st.subheader("Interactive Scenario Simulator")
        st.caption("Adjust demo inventory reserves or incident requests to evaluate allocation outcomes and shortage reduction.")

        # Default incident & inventory lists
        default_inventory = [
            {"resource_type": "ambulance", "available_quantity": 3},
            {"resource_type": "rescue_team", "available_quantity": 3},
            {"resource_type": "water_kit", "available_quantity": 6},
        ]

        default_incidents = [
            {"incident_id": "DEMO-001", "priority_score": 95, "ambulances_needed": 2, "rescue_teams_needed": 1, "water_kits_needed": 3},
            {"incident_id": "DEMO-002", "priority_score": 82, "ambulances_needed": 1, "rescue_teams_needed": 2, "water_kits_needed": 4},
            {"incident_id": "DEMO-003", "priority_score": 68, "ambulances_needed": 2, "rescue_teams_needed": 1, "water_kits_needed": 2},
        ]

        with st.form("simulation_form"):
            st.markdown("#### 1. Adjust Scenario Inventory")
            inv_c1, inv_c2, inv_c3 = st.columns(3)
            amb_qty = inv_c1.number_input("Ambulance Stock", min_value=0, max_value=50, value=3)
            rescue_qty = inv_c2.number_input("Rescue Team Stock", min_value=0, max_value=50, value=3)
            water_qty = inv_c3.number_input("Water Kit Stock", min_value=0, max_value=50, value=6)

            st.markdown("#### 2. Incident Scenario Requests")
            st.caption("Incident priority scores determine triage order in the greedy allocation solver.")

            sim_incidents = []
            for i, d_inc in enumerate(default_incidents):
                st.markdown(f"**Incident: {d_inc['incident_id']}**")
                col_s, col_a, col_r, col_w = st.columns(4)
                score = col_s.number_input(f"Priority Score ({d_inc['incident_id']})", 0, 100, int(d_inc["priority_score"]), key=f"score_{i}")
                amb = col_a.number_input(f"Ambulances Needed", 0, 20, int(d_inc["ambulances_needed"]), key=f"amb_{i}")
                res = col_r.number_input(f"Rescue Teams Needed", 0, 20, int(d_inc["rescue_teams_needed"]), key=f"res_{i}")
                wat = col_w.number_input(f"Water Kits Needed", 0, 20, int(d_inc["water_kits_needed"]), key=f"wat_{i}")

                sim_incidents.append({
                    "incident_id": d_inc["incident_id"],
                    "priority_score": str(score),
                    "ambulances_needed": str(amb),
                    "rescue_teams_needed": str(res),
                    "water_kits_needed": str(wat),
                })

            sim_submitted = st.form_submit_button("🚀 Run Scenario Allocation Simulation", use_container_width=True)

        if sim_submitted:
            sim_inventory = [
                {"resource_type": "ambulance", "available_quantity": str(amb_qty)},
                {"resource_type": "rescue_team", "available_quantity": str(rescue_qty)},
                {"resource_type": "water_kit", "available_quantity": str(water_qty)},
            ]

            try:
                assignments, shortages, metrics = utils.run_allocation_simulation(sim_incidents, sim_inventory)

                st.markdown("---")
                st.subheader("Simulation Results")

                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Requested Units", f"{metrics['total_requested']}")
                sm2.metric("Available Stock", f"{metrics['total_inventory']}")
                sm3.metric("Fulfillment Rate", f"{metrics['fulfillment_rate']}%")
                sm4.metric("Incidents with Shortage", f"{metrics['incidents_with_unmet']} / {metrics['incidents_count']}")

                st.markdown("#### Scenario Assignment Output")
                sim_res_df = pd.DataFrame(assignments)
                st.dataframe(sim_res_df, use_container_width=True, hide_index=True)

                if shortages:
                    st.markdown("#### Scenario Unmet Shortages")
                    st.dataframe(pd.DataFrame(shortages), use_container_width=True, hide_index=True)
                else:
                    st.success("🎉 All requested resources were 100% fulfilled in this scenario simulation!")

            except Exception as exc:
                st.error(f"Simulation error: {exc}")


# -----------------------------------------------------------------------------
# PAGE 7: 🛡️ PIPELINE HEALTH & SYSTEM DIAGNOSTICS
# -----------------------------------------------------------------------------

elif page == "🛡️ Pipeline Health":
    st.title("🛡️ Pipeline Health & Architecture Diagnostics")
    st.caption("Inspect pipeline file integrity, schema compliance, execution dependencies, and system telemetry.")

    st.markdown(
        """
        <div class="emergency-banner" style="border-left-color: #10B981;">
            <div class="emergency-banner-title">
                <span>🩺</span> SYSTEM DIAGNOSTIC STATUS
            </div>
            <div class="emergency-banner-subtitle">
                RESCUEGRID audits file paths, record volumes, column schemas, and module integrity on every launch.
                Review the audit matrix below for pipeline verification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    health_data = utils.inspect_pipeline_health()
    health_df = pd.DataFrame(health_data)

    # Health Metrics
    total_files = len(health_df)
    healthy_files = (health_df["status"] == "HEALTHY").sum()
    warning_files = (health_df["status"] == "SCHEMA_WARNING").sum()
    missing_files = (health_df["status"] == "MISSING").sum()

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Audited Pipeline Assets", f"{total_files}")
    h2.metric("Fully Operational", f"{healthy_files}", delta="Healthy", delta_color="normal")
    h3.metric("Schema Warnings", f"{warning_files}")
    h4.metric("Pending Modules", f"{missing_files}")

    st.markdown("---")
    st.subheader("Pipeline Asset Health Matrix")

    # Format table for display
    display_matrix = health_df[[
        "name", "type", "status", "rows", "size_kb", "modified", "script"
    ]].copy()
    display_matrix.columns = [
        "Asset Name", "Pipeline Stage", "Health Status", "Row Count", "Size (KB)", "Last Modified", "Generator Script"
    ]

    st.dataframe(display_matrix, use_container_width=True, hide_index=True)

    # Missing Columns or Detailed Errors if any
    warnings = [item for item in health_data if item["missing_cols"]]
    if warnings:
        st.warning("The following files are missing expected columns:")
        for w in warnings:
            st.write(f"- `{w['filename']}`: Missing `{w['missing_cols']}`")

    st.markdown("---")
    st.subheader("Runtime Environment & Dependencies")

    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(f"**Python:** `{sys.version.split()[0]}`")
    e2.markdown(f"**Streamlit:** `{st.__version__}`")
    e3.markdown(f"**Pandas:** `{pd.__version__}`")
    e4.markdown(f"**Plotly:** `{px.__version__ if hasattr(px, '__version__') else 'Installed'}`")

    st.markdown("---")
    st.subheader("Pipeline Module Execution Guide")
    st.markdown(
        """
        To regenerate or update any stage of the RESCUEGRID pipeline, execute the respective command from the project root:
        - **1. Merge Raw HumAID Data:** `python prepare_humaid.py`
        - **2. Clean & Extract Categories:** `python extract.py`
        - **3. Generate Similar Pairs:** `python group.py`
        - **4. Join Text for Pair Review:** `python inspect_pairs.py`
        - **5. Detect Potential Conflicts:** `python conflicts.py`
        - **6. Compute Priority Scores:** `python priority.py`
        - **7. Run Resource Allocation:** `python allocator.py`
        - **8. Launch Dashboard:** `streamlit run app.py`
        """
    )


# -----------------------------------------------------------------------------
# GLOBAL FOOTER
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 12px; padding: 10px;">
        RESCUEGRID: Evidence-Aware Emergency Intelligence and Resource Allocation Under Conflicting Reports<br>
        Developed for Emergency Decision Support • Historical Research Prototype • Human Review Required
    </div>
    """,
    unsafe_allow_html=True,
)
