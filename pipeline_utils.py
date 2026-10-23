"""
RESCUEGRID - Pipeline Utilities & Helper Functions
=================================================
Provides centralized, defensive data loading, caching, schema validation,
metric calculations, signal parsing, and allocation simulation for RESCUEGRID.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Optional Streamlit cache decorator (falls back gracefully in headless/test environments)
try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    def cache_data(func):
        return func


# -----------------------------------------------------------------------------
# 1. DEFENSIVE FILE LOADING & CACHING
# -----------------------------------------------------------------------------

def safe_read_csv(
    path: Path,
    sep: str = ",",
    dtype: Any = str,
    nrows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Defensively read a CSV/TSV file with fallback encodings and empty handling."""
    if not path.exists():
        return None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                dtype=dtype,
                keep_default_na=False,
                encoding=encoding,
                nrows=nrows,
                low_memory=False,
            )
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    return None


@cache_data(show_spinner=False)
def load_reports_dataset() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load the best available reports dataset.
    Priority order:
    1. priority_reports.tsv (rich: contains priority scores, categories, text)
    2. processed_reports.tsv (contains extracted categories, hints)
    3. humaid_reports_merged.tsv (raw merged dataset)
    Returns: (DataFrame, source_file_name)
    """
    priority_file = DATA_DIR / "priority_reports.tsv"
    processed_file = DATA_DIR / "processed_reports.tsv"
    merged_file = DATA_DIR / "humaid_reports_merged.tsv"

    if priority_file.exists():
        df = safe_read_csv(priority_file, sep="\t")
        if df is not None:
            if "priority_score" in df.columns:
                df["priority_score_numeric"] = pd.to_numeric(df["priority_score"], errors="coerce").fillna(0)
            return df, "priority_reports.tsv"

    if processed_file.exists():
        df = safe_read_csv(processed_file, sep="\t")
        if df is not None:
            return df, "processed_reports.tsv"

    if merged_file.exists():
        df = safe_read_csv(merged_file, sep="\t")
        if df is not None:
            return df, "humaid_reports_merged.tsv"

    return None, "none"


@cache_data(show_spinner=False)
def load_similarity_dataset() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load candidate similarity pairs.
    Prefers review_similar_pairs.tsv (contains joined text), falls back to similar_report_pairs.tsv.
    Ensures normalized column names: 'event_name' is always available.
    """
    review_pairs_file = DATA_DIR / "review_similar_pairs.tsv"
    pairs_file = DATA_DIR / "similar_report_pairs.tsv"

    if review_pairs_file.exists():
        df = safe_read_csv(review_pairs_file, sep="\t")
        if df is not None:
            if "event_a" in df.columns and "event_name" not in df.columns:
                df["event_name"] = df["event_a"]
            if "similarity" in df.columns:
                df["similarity_numeric"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)
            return df, "review_similar_pairs.tsv"

    if pairs_file.exists():
        df = safe_read_csv(pairs_file, sep="\t")
        if df is not None:
            if "similarity" in df.columns:
                df["similarity_numeric"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)
            return df, "similar_report_pairs.tsv"

    return None, "none"


@cache_data(show_spinner=False)
def load_conflicts_dataset() -> Optional[pd.DataFrame]:
    """Load potential conflicts dataset."""
    conflicts_file = DATA_DIR / "potential_conflicts.tsv"
    if conflicts_file.exists():
        df = safe_read_csv(conflicts_file, sep="\t")
        if df is not None and "similarity" in df.columns:
            df["similarity_numeric"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)
        return df
    return None


@cache_data(show_spinner=False)
def load_allocation_data() -> Dict[str, Optional[pd.DataFrame]]:
    """Load all resource allocation related tables."""
    return {
        "results": safe_read_csv(DATA_DIR / "allocation_results.csv"),
        "unmet": safe_read_csv(DATA_DIR / "unmet_needs.csv"),
        "inventory": safe_read_csv(DATA_DIR / "resource_inventory.csv"),
        "incidents": safe_read_csv(DATA_DIR / "demo_incidents.csv"),
    }


# -----------------------------------------------------------------------------
# 2. CONFLICT SIGNAL PARSER
# -----------------------------------------------------------------------------

def parse_conflict_signals(signal_str: str) -> List[Dict[str, str]]:
    """
    Parse composite conflict signal strings into structured dictionaries.
    Example:
    'NUMBER_DIFFERENCE_REVIEW_DEATH_TOLL(A:30;B:25) | OPPOSING_ROAD_STATUS(A:open;B:closed)'
    Returns list of dicts with keys: raw, signal_type, metric, a_val, b_val, summary
    """
    if not signal_str:
        return []

    signals = [s.strip() for s in str(signal_str).split("|") if s.strip()]
    parsed = []

    for sig in signals:
        m = re.match(r"^(?P<full_type>[A-Za-z0-9_]+)\s*(?:\(A:(?P<a_val>[^;]*);B:(?P<b_val>[^)]*)\))?$", sig)
        if m:
            full_type = m.group("full_type") or "SIGNAL"
            a_val = (m.group("a_val") or "").strip()
            b_val = (m.group("b_val") or "").strip()

            if full_type.startswith("NUMBER_DIFFERENCE_REVIEW_"):
                stype = "NUMBER_DIFFERENCE"
                metric = full_type.replace("NUMBER_DIFFERENCE_REVIEW_", "")
            elif full_type.startswith("OPPOSING_"):
                stype = "OPPOSING_CLAIM"
                metric = full_type.replace("OPPOSING_", "")
            else:
                stype = full_type
                metric = full_type

            clean_metric = metric.replace("_", " ").title()
            if a_val and b_val:
                summary = f"{clean_metric} mismatch: Report A states '{a_val}', Report B states '{b_val}'"
            elif a_val or b_val:
                summary = f"{clean_metric} variance noted (A: '{a_val}', B: '{b_val}')"
            else:
                summary = f"Opposing claim detected in {clean_metric}"

            parsed.append({
                "raw": sig,
                "signal_type": stype,
                "metric": metric,
                "a_val": a_val,
                "b_val": b_val,
                "summary": summary,
            })
        else:
            parsed.append({
                "raw": sig,
                "signal_type": "GENERAL_SIGNAL",
                "metric": sig,
                "a_val": "",
                "b_val": "",
                "summary": sig,
            })

    return parsed


# -----------------------------------------------------------------------------
# 3. INTERACTIVE ALLOCATION SIMULATION
# -----------------------------------------------------------------------------

def run_allocation_simulation(
    incidents: List[dict],
    inventory: List[dict],
) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    """
    Execute allocation algorithm with input validation and summary metrics.
    Imports allocator.py directly to preserve teammate's logic.
    """
    import allocator

    # Defensive copy and validation
    assignments, shortages = allocator.allocate_resources(incidents, inventory)

    # Compute key performance metrics for the simulation
    total_requested = sum(
        int(inc.get(col, 0))
        for inc in incidents
        for col in allocator.RESOURCE_COLUMNS.values()
    )

    total_inventory = sum(
        int(inv.get("available_quantity", 0))
        for inv in inventory
    )

    total_unmet = sum(
        int(sh.get("unmet", 0))
        for sh in shortages
    )

    total_allocated = total_requested - total_unmet

    fulfillment_rate = (
        (total_allocated / total_requested * 100.0) if total_requested > 0 else 100.0
    )

    metrics = {
        "total_requested": total_requested,
        "total_inventory": total_inventory,
        "total_allocated": total_allocated,
        "total_unmet": total_unmet,
        "fulfillment_rate": round(fulfillment_rate, 1),
        "incidents_count": len(incidents),
        "incidents_with_unmet": len({sh["incident_id"] for sh in shortages}),
    }

    return assignments, shortages, metrics


# -----------------------------------------------------------------------------
# 4. SYSTEM HEALTH INSPECTION
# -----------------------------------------------------------------------------

def inspect_pipeline_health() -> List[Dict[str, Any]]:
    """
    Audit all expected files, columns, row counts, and health status in RESCUEGRID.
    """
    file_specs = [
        {
            "name": "Raw Merged Reports",
            "path": DATA_DIR / "humaid_reports_merged.tsv",
            "sep": "\t",
            "script": "python prepare_humaid.py",
            "type": "Data Pipeline",
            "required_cols": ["tweet_id", "tweet_text", "original_label", "event_name"],
        },
        {
            "name": "Processed Reports",
            "path": DATA_DIR / "processed_reports.tsv",
            "sep": "\t",
            "script": "python extract.py",
            "type": "Extraction & Cleaning",
            "required_cols": ["tweet_id", "clean_text", "humanitarian_category", "review_hint"],
        },
        {
            "name": "Similarity Pairs",
            "path": DATA_DIR / "similar_report_pairs.tsv",
            "sep": "\t",
            "script": "python group.py",
            "type": "Candidate Generation",
            "required_cols": ["tweet_id_a", "tweet_id_b", "similarity"],
        },
        {
            "name": "Reviewed Similar Pairs",
            "path": DATA_DIR / "review_similar_pairs.tsv",
            "sep": "\t",
            "script": "python inspect_pairs.py",
            "type": "Review Candidate Join",
            "required_cols": ["tweet_id_a", "tweet_id_b", "similarity", "tweet_text_a", "tweet_text_b"],
        },
        {
            "name": "Potential Conflicts",
            "path": DATA_DIR / "potential_conflicts.tsv",
            "sep": "\t",
            "script": "python conflicts.py",
            "type": "Contradiction Detection",
            "required_cols": ["tweet_id_a", "tweet_id_b", "conflict_signals", "event_name"],
        },
        {
            "name": "Priority Scored Reports",
            "path": DATA_DIR / "priority_reports.tsv",
            "sep": "\t",
            "script": "python priority.py",
            "type": "Urgency & Priority Scoring",
            "required_cols": ["tweet_id", "priority_score", "priority_level", "priority_reasons"],
        },
        {
            "name": "Demo Incident Requests",
            "path": DATA_DIR / "demo_incidents.csv",
            "sep": ",",
            "script": "Manual Scenario Input",
            "type": "Synthetic Demo Scenario",
            "required_cols": ["incident_id", "priority_score", "ambulances_needed"],
        },
        {
            "name": "Resource Inventory",
            "path": DATA_DIR / "resource_inventory.csv",
            "sep": ",",
            "script": "Manual Scenario Input",
            "type": "Synthetic Inventory",
            "required_cols": ["resource_type", "available_quantity"],
        },
        {
            "name": "Allocation Results",
            "path": DATA_DIR / "allocation_results.csv",
            "sep": ",",
            "script": "python allocator.py",
            "type": "Resource Decision Engine",
            "required_cols": ["incident_id", "priority_score", "allocations", "unmet_needs"],
        },
        {
            "name": "Unmet Needs Shortages",
            "path": DATA_DIR / "unmet_needs.csv",
            "sep": ",",
            "script": "python allocator.py",
            "type": "Shortage Report",
            "required_cols": ["incident_id", "resource_type", "unmet"],
        },
    ]

    results = []
    for spec in file_specs:
        path = spec["path"]
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        mod_time = (
            pd.to_datetime(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
            if exists
            else "N/A"
        )

        rows = 0
        cols_found = []
        missing_cols = []
        status = "MISSING"

        if exists:
            try:
                # Read sample to verify columns quickly
                sample = pd.read_csv(
                    path,
                    sep=spec["sep"],
                    nrows=5,
                    keep_default_na=False,
                    encoding="utf-8-sig",
                )
                cols_found = list(sample.columns)
                missing = [c for c in spec["required_cols"] if c not in cols_found]
                missing_cols = missing

                # Read line count for total rows
                with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
                    rows = max(0, sum(1 for _ in f) - 1)

                if missing:
                    status = "SCHEMA_WARNING"
                else:
                    status = "HEALTHY"
            except Exception as e:
                status = f"ERROR: {e}"

        results.append({
            "name": spec["name"],
            "filename": path.name,
            "type": spec["type"],
            "exists": exists,
            "status": status,
            "rows": rows,
            "size_kb": round(size_bytes / 1024, 1),
            "modified": mod_time,
            "missing_cols": missing_cols,
            "script": spec["script"],
        })

    return results
