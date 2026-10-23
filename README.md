# RESCUEGRID — Evidence-Aware Emergency Intelligence and Resource Allocation Under Conflicting Reports

**RESCUEGRID** is a modern emergency operations intelligence platform designed for disaster response coordinators and incident analysts. The system analyzes disaster-related social media reports, classifies humanitarian needs, identifies candidate duplicate reports, flags potentially contradictory claims, prioritizes reports for human triage, and demonstrates transparent resource allocation under limited inventory.

---

## 🛰️ System Architecture & Pipeline Flow

The platform follows a modular, verifiable pipeline where every module produces transparent, audit-ready data files:

```
[HumAID Event & Combined Datasets]
                │
                ▼
      prepare_humaid.py   ──► data/humaid_reports_merged.tsv (43,409 rows)
                │
                ▼
          extract.py      ──► data/processed_reports.tsv (Categories & Keywords)
                │
         ┌──────┴────────────────────────┐
         ▼                               ▼
      group.py                       priority.py
  (TF-IDF Nearest Neighbors)     (Rule-Based Triage Scoring)
         │                               │
         ▼                               ▼
data/similar_report_pairs.tsv    data/priority_reports.tsv (Scored [0-100])
         │
         ├───────────────────────────────┐
         ▼                               ▼
  inspect_pairs.py                 conflicts.py
(Joined Review Text)         (Contradiction & Metric Mismatch)
         │                               │
         ▼                               ▼
data/review_similar_pairs.tsv    data/potential_conflicts.tsv (Flagged Signals)
         │                               │
         └───────────────┬───────────────┘
                         ▼
                   allocator.py
         (Greedy Priority-First Allocator)
                         │
                         ▼
             data/allocation_results.csv & unmet_needs.csv
                         │
                         ▼
               Streamlit Dashboard
                    (app.py)
```

---

## 🏷️ System Truth & Evidence Attribution

RESCUEGRID strictly distinguishes between four categories of evidence to ensure data integrity:

| Evidence Level | Description | Color Tag |
| :--- | :--- | :--- |
| **Data-Observed** | Verified text, disaster event names, and humanitarian class labels from the official HumAID dataset. | 🔵 Blue Badge |
| **Rule-Based Heuristic** | Keyword indicators, conflict signal detection (numeric/status mismatch), and review priority scores [0–100]. | 🟡 Amber Badge |
| **Model-Derived** | Character n-gram TF-IDF embeddings and cosine similarity candidate pairs. | 🟣 Purple Badge |
| **Synthetic Demo** | Demonstration inventory quantities, simulated incident resource requests, and baseline greedy allocations. | 🔴 Red Badge |

> [!IMPORTANT]
> **Operational Guardrail**: This application is a decision-support prototype. Priority scores indicate review queue urgency, not physical ground-truth disaster severity. Conflict signals highlight potential discrepancies for analyst triage (e.g. death toll updates over time or across localities) and do not denote confirmed misinformation. All resource recommendations require human approval; **real dispatch is never triggered**.

---

## 🚀 Quickstart & Setup

### 1. Requirements & Virtual Environment

Python 3.10+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `streamlit>=1.30.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `plotly>=5.18.0`
- `pytest>=8.0.0`

### 2. Launch the Streamlit Intelligence Dashboard

Run from the project root:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## 📋 Running Pipeline Modules Manually

All data artifacts are pre-generated in the `data/` directory. If you wish to re-execute any stage:

1. **Merge HumAID Records**:
   ```bash
   python prepare_humaid.py
   ```
2. **Clean Text & Extract Humanitarian Categories**:
   ```bash
   python extract.py
   ```
3. **Find Candidate Similar Report Pairs**:
   ```bash
   python group.py
   ```
4. **Prepare Joined Text for Pair Review**:
   ```bash
   python inspect_pairs.py
   ```
5. **Detect Potential Conflict Signals**:
   ```bash
   python conflicts.py
   # Run self-tests:
   python conflicts.py --self-test
   ```
6. **Assign Explainable Priority Scores**:
   ```bash
   python priority.py
   ```
7. **Run Resource Allocation**:
   ```bash
   python allocator.py
   ```

---

## 🧪 Running Automated Unit Tests

A comprehensive unit and regression test suite is located in `tests/test_rescuegrid.py`:

```bash
pytest -v
```

Tests cover:
- Defensive dataset loading and encoding fallbacks (`pipeline_utils.py`)
- Schema completeness across all TSV/CSV data assets
- Regex parsing of composite conflict signals
- Invariant bounds on priority scores `[0, 100]`
- Input validation on allocator (duplicate IDs, negative quantities, missing columns)
- Greedy allocation prioritization logic and shortage accounting

---

## 🖥️ Dashboard Pages Overview

1. **🛰️ Executive Overview**: System-wide telemetry, total reports processed (43,409), events represented, candidate pair counts, conflict flags, humanitarian category distributions, and pipeline readiness.
2. **🔍 Report Explorer**: Fast bounded search and filtering across events, categories, and priority levels. Includes a deep-dive Report Dossier with raw, cleaned, and normalized text.
3. **🔗 Similarity & Duplicates**: Interactive cosine similarity slider, similarity score distribution, side-by-side text comparison, and session-based human duplicate review decisions.
4. **⚖️ Conflict Review**: Rule-based conflict flags for numeric differences (e.g. casualty figures 25 vs 30) and opposing claims (road open vs closed). Includes human triage status logging.
5. **📈 Priority Analytics**: Full score distribution histograms, review queue prioritization, breakdown by urgency indicator phrases, and category-level triage queuing.
6. **📦 Resource Allocation**: Current baseline greedy allocation review, unmet shortage charts, and an **Interactive Scenario Simulator** allowing real-time inventory and request adjustments with instant recalculation.
7. **🛡️ Pipeline Health**: Diagnostics matrix verifying file existence, row counts, schema compliance, runtime versions, and module execution instructions.

---

## 📂 Project Directory Structure

```
RESCUEGRID/
├── allocator.py                     # Transparent greedy resource allocation engine
├── app.py                           # Upgraded Streamlit Emergency Intelligence Dashboard
├── conflicts.py                     # Rule-based conflict & discrepancy detection
├── extract.py                       # Report cleaning, category mapping & indicator tagging
├── group.py                         # TF-IDF char n-gram nearest-neighbor similarity
├── inspect_data.py                  # Dataset inspection & audit utility
├── inspect_pairs.py                 # Similarity pair text joiner & review generator
├── pipeline_utils.py                # Centralized cached loaders, parsers & simulation helpers
├── prepare_humaid.py                # HumAID dataset merger and deduplicator
├── priority.py                      # Rule-based explainable emergency priority scoring
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation and architecture guide
├── data/                            # Processed TSV/CSV datasets & raw HumAID files
│   ├── allocation_results.csv
│   ├── demo_incidents.csv
│   ├── humaid_merge_audit.txt
│   ├── humaid_reports_merged.tsv
│   ├── potential_conflicts.tsv
│   ├── priority_reports.tsv
│   ├── processed_reports.tsv
│   ├── report_groups.tsv
│   ├── resource_inventory.csv
│   ├── review_similar_pairs.tsv
│   ├── similar_report_pairs.tsv
│   └── unmet_needs.csv
└── tests/
    └── test_rescuegrid.py           # Pytest test suite
```
