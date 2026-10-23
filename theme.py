"""
RESCUEGRID - Tactical Command Design System & Theme Configuration
=================================================================
Centralized color palette, typographic scales, CSS stylesheets,
and Plotly figure themes for the RESCUEGRID operations platform.
Strictly adheres to the professional emergency command center design system.
"""

from __future__ import annotations
from typing import Dict, Any
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. CORE COLOR PALETTE
# -----------------------------------------------------------------------------
APP_BG = "#080F1C"              # Deep navy application background
PANEL_BG = "#101B2D"            # Primary panel background
PANEL_BG_SECONDARY = "#142238"  # Secondary panel background
PANEL_BORDER = "#26374D"        # Muted blue-gray borders

ACCENT_CYAN = "#00D5E8"         # Primary interactive accent / selected node
CRITICAL_RED = "#F04452"        # Critical incident (Priority >= 80) / Unmet deficit
WARNING_AMBER = "#F5A623"       # High-priority warning (Priority 50-79) / Greedy
SUCCESS_GREEN = "#2ECC71"       # Successful allocation / Standard (Priority < 50)
DEPOT_GOLD = "#F5A623"          # Logistics depot node

TEXT_MAIN = "#F2F6FC"           # Main primary text
TEXT_SECONDARY = "#9BAEC5"      # Secondary muted labels
TEXT_MUTED = "#607289"          # Tertiary / disabled text

# Resource Urgency & Color Configuration
RESOURCE_COLORS: Dict[str, str] = {
    "ambulance": "#F04452",
    "rescue_team": "#F5A623",
    "water_kit": "#00D5E8",
    "medical_kit": "#E056FD",
    "food_ration": "#2ECC71",
    "shelter_pack": "#E67E22",
    "power_generator": "#F1C40F",
    "hazmat_unit": "#9B59B6",
    "communication_node": "#3498DB",
    "boat_rescue": "#1ABC9C",
}

# -----------------------------------------------------------------------------
# 2. TACTICAL CSS STYLESHEET
# -----------------------------------------------------------------------------
TACTICAL_CSS = f"""
<style>
    /* Dark Navy Command Center Background */
    .stApp {{
        background-color: {APP_BG};
        color: {TEXT_MAIN};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {PANEL_BG};
        border-right: 1px solid {PANEL_BORDER};
    }}

    /* Global Typography & Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_MAIN} !important;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}

    p, span, label {{
        color: {TEXT_SECONDARY};
    }}

    /* Top Command Header */
    .top-command-header {{
        background-color: {PANEL_BG};
        border: 1px solid {PANEL_BORDER};
        border-left: 3px solid {ACCENT_CYAN};
        border-radius: 4px;
        padding: 10px 16px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .header-brand {{
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: {TEXT_MAIN};
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .header-meta {{
        font-size: 12px;
        color: {TEXT_SECONDARY};
        display: flex;
        align-items: center;
        gap: 16px;
    }}

    /* KPI Ribbon Cards */
    .kpi-container {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
        margin-bottom: 14px;
    }}

    .kpi-card {{
        background-color: {PANEL_BG};
        border: 1px solid {PANEL_BORDER};
        border-radius: 4px;
        padding: 10px 12px;
        transition: border-color 0.15s ease;
    }}

    .kpi-card:hover {{
        border-color: {ACCENT_CYAN};
    }}

    .kpi-label {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {TEXT_SECONDARY};
        margin-bottom: 4px;
    }}

    .kpi-value {{
        font-size: 22px;
        font-weight: 700;
        color: {TEXT_MAIN};
        line-height: 1.1;
    }}

    .kpi-subtext {{
        font-size: 11px;
        color: {TEXT_MUTED};
        margin-top: 3px;
    }}

    /* Command Panels */
    .command-panel {{
        background-color: {PANEL_BG};
        border: 1px solid {PANEL_BORDER};
        border-radius: 4px;
        padding: 14px;
        margin-bottom: 12px;
    }}

    .panel-header {{
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {TEXT_MAIN};
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid {PANEL_BORDER};
    }}

    /* Badges & Tags (Zero Emoji) */
    .badge {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}

    .badge-cyan {{
        background-color: rgba(0, 213, 232, 0.12);
        color: {ACCENT_CYAN};
        border: 1px solid rgba(0, 213, 232, 0.35);
    }}

    .badge-red {{
        background-color: rgba(240, 68, 82, 0.12);
        color: {CRITICAL_RED};
        border: 1px solid rgba(240, 68, 82, 0.35);
    }}

    .badge-amber {{
        background-color: rgba(245, 166, 35, 0.12);
        color: {WARNING_AMBER};
        border: 1px solid rgba(245, 166, 35, 0.35);
    }}

    .badge-green {{
        background-color: rgba(46, 204, 113, 0.12);
        color: {SUCCESS_GREEN};
        border: 1px solid rgba(46, 204, 113, 0.35);
    }}

    .badge-neutral {{
        background-color: rgba(155, 174, 197, 0.1);
        color: {TEXT_SECONDARY};
        border: 1px solid {PANEL_BORDER};
    }}

    /* Streamlit Tabs Navigation */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid {PANEL_BORDER};
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: {PANEL_BG};
        border: 1px solid {PANEL_BORDER};
        border-bottom: none;
        border-radius: 4px 4px 0 0;
        color: {TEXT_SECONDARY};
        padding: 6px 16px;
        font-size: 12px;
        font-weight: 600;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {PANEL_BG_SECONDARY} !important;
        color: {ACCENT_CYAN} !important;
        border-color: {ACCENT_CYAN} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {PANEL_BG_SECONDARY};
        color: {TEXT_MAIN};
        border: 1px solid {PANEL_BORDER};
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        transition: all 0.15s ease;
    }}

    .stButton > button:hover {{
        background-color: {PANEL_BORDER};
        border-color: {ACCENT_CYAN};
        color: {ACCENT_CYAN};
    }}

    .stButton > button:active {{
        background-color: {ACCENT_CYAN};
        color: {APP_BG};
    }}

    /* Dataframe tables */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {PANEL_BORDER};
        border-radius: 4px;
        background-color: {PANEL_BG};
    }}
</style>
"""


# -----------------------------------------------------------------------------
# 3. PLOTLY TACTICAL LAYOUT GENERATOR
# -----------------------------------------------------------------------------
def get_tactical_plotly_layout(
    height: int = 280,
    margin: Dict[str, int] = None,
    show_legend: bool = True,
) -> Dict[str, Any]:
    """Return unified tactical styling dictionary for Plotly charts."""
    if margin is None:
        margin = dict(l=35, r=15, t=30, b=35)

    return dict(
        paper_bgcolor=APP_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=TEXT_SECONDARY,
            size=11,
        ),
        margin=margin,
        height=height,
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(16, 27, 45, 0.8)",
            bordercolor=PANEL_BORDER,
            borderwidth=1,
            font=dict(size=10, color=TEXT_SECONDARY),
        ),
        xaxis=dict(
            gridcolor=PANEL_BORDER,
            zerolinecolor=PANEL_BORDER,
            tickfont=dict(size=10, color=TEXT_SECONDARY),
        ),
        yaxis=dict(
            gridcolor=PANEL_BORDER,
            zerolinecolor=PANEL_BORDER,
            tickfont=dict(size=10, color=TEXT_SECONDARY),
        ),
    )
