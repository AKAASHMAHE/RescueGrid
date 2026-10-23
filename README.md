# RESCUEGRID — 3D Emergency Command Center & Optimization Engine

**RESCUEGRID** is an operations-research-driven emergency intelligence and resource allocation platform with a real-time, interactive **3D Digital-Twin Command Center**. The system reconciles conflicting disaster reports, scores operational urgency, and solves constrained emergency resource allocation problems using **Mixed-Integer Linear Programming (MILP)** benchmarked against greedy heuristics.

---

## 🛰️ 3D Digital Twin Command Center

Inspired by tactical operations digital twins, the central command center provides:
- **WebGL / Three.js 3D Response Grid:** Interactive 3D visualization with glowing coordinate plane, central logistics depot hub, dynamic incident pillars elevated by priority score, and curved allocation conduits.
- **Plotly 3D Dual-Engine:** 100% offline-compatible 3D scatter and vector conduit view.
- **Real-Time Scenario Sandbox:** Dynamic inventory sliders and policy toggles that solve and update the 3D scene instantaneously.
- **Tactical Dossier Panel:** Live inspection of selected incident sites, evidence text, assigned units, unmet shortages, and solver trade-off rationale.

---

## 🧮 Optimization Engine & Mathematical Formulation

The optimization engine is located in `optimization/` and executes completely independently of Streamlit.

### 1. Decision Variables
- $x[i, r] \in \mathbb{Z}_{\ge 0}$ : Integer quantity of resource type $r$ allocated to incident $i$.
- $u[i, r] \in \mathbb{Z}_{\ge 0}$ : Integer quantity of unmet demand of resource $r$ for incident $i$.

### 2. Constraints
1. **Demand Conservation:**
   $$\forall i \in I, r \in R: \quad x[i, r] + u[i, r] = \text{demand}[i, r]$$
2. **Resource Depot Capacity:**
   $$\forall r \in R: \quad \sum_{i \in I} x[i, r] \le \text{available}[r]$$
3. **Integrality & Non-Negativity:**
   $$x[i, r], u[i, r] \in \{0, 1, 2, \dots\}$$
4. **Optional Minimum Service Guarantee (Anti-Starvation Policy):**
   $$\forall i \in I \text{ with } \text{demand} > 0: \quad \sum_{r \in R} x[i, r] \ge \text{min\_units} \quad (\text{when physically feasible})$$

### 3. Objective Function
Minimize cumulative priority-weighted unmet demand:
$$\min \sum_{i \in I} \sum_{r \in R} w[i, r] \cdot u[i, r]$$

Where:
$$w[i, r] = \max(0.01, \frac{\text{priority}[i]}{100.0}) \times \text{resource\_urgency}[r]$$
- Critical life-saving resources have higher weights (Ambulance: 1.5, Rescue Team: 1.3, Water Kit: 1.0).
- Configurable policies support standard linear weighting, minimum service anti-starvation guarantees, and bundle-aware critical escalation.

---

## 🔬 Optimization vs. Greedy Comparison Findings

The platform implements both `scipy.optimize.milp` (using the HiGHS MILP solver) and a strict greedy priority-first baseline operating on identical data structures:

| Scenario | Demands & Conditions | Greedy Baseline Result | MILP Optimizer Result | Key Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Sufficient Stock** | Surplus regional stockpile across all resource types. | 100% fulfillment. | 100% fulfillment. | Both algorithms achieve identical optimal solutions under surplus. |
| **Severe Shortage** | Heavy supply deficit (~30% stock available). | Priority queue takes stock until depleted. | Global minimum penalty achieved. | High-priority incidents prioritized; shortages transparently logged. |
| **Greedy Starvation vs. Equitable Guarantee** | High-demand site ($P=95$) competing with lower sites ($P=76, 64$). | $P=95$ site consumes 100% of stock; lower sites get **0 units** (completely starved). | All sites receive at least 1 unit under Minimum Service Guarantee. | **MILP eliminates starvation**, preventing regional abandonment while still prioritizing critical sites. |
| **Zero Inventory** | Complete stockout (0 units available). | 0% allocated, all unmet. | 0% allocated, all unmet. | Clean solver termination without divide-by-zero or crashes. |
| **Zero Demand** | Informational reports only (0 demand). | 100% fulfilled, 0 allocated. | 100% fulfilled, 0 allocated. | Zero demand edge case handled cleanly in 0.0ms. |

---

## 🚀 Quickstart & Setup

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests (21 Tests)
```powershell
pytest -v
```

### 3. Run the Allocator from Command Line
```powershell
# Mixed-Integer Linear Programming solver
python allocator.py --solver milp

# Greedy Priority-First baseline
python allocator.py --solver greedy
```

### 4. Launch the 3D Command Center Dashboard
```powershell
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📂 Project Architecture

```
RESCUEGRID/
├── optimization/                    # Modular Optimization Engine
│   ├── __init__.py                  # Package exports
│   ├── models.py                    # Incident, Resource, AllocationPlan, EvaluationMetrics
│   ├── optimizer.py                 # Mixed-Integer Linear Programming (MILP) solver
│   ├── baseline.py                  # Greedy priority-first baseline solver
│   ├── metrics.py                   # Comparative benchmarking & statistical metrics
│   └── scenarios.py                 # 7 reproducible simulation scenarios
├── visualization_3d.py              # Three.js WebGL & Plotly 3D visual engines
├── allocator.py                     # CLI entrypoint for MILP and greedy solvers
├── app.py                           # 3D Digital Twin Command Center dashboard
├── pipeline_utils.py                # Cached data loaders, conflict parsers, decision persistence
├── conflicts.py                     # Rule-based conflict & discrepancy detection
├── priority.py                      # Rule-based report priority scoring
├── extract.py                       # Text cleaning & humanitarian category mapping
├── prepare_humaid.py                # HumAID dataset merging and deduplication
├── group.py                         # TF-IDF candidate pair generation
├── inspect_pairs.py                 # Pair review joiner with Windows encoding fix
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Documentation & verification guide
├── data/                            # 43,409 HumAID records, conflicts, inventory, reviews
└── tests/
    └── test_rescuegrid.py           # 21 automated unit tests
```

---

## 🛡️ Operational Safeguards & Evidence Attribution

1. **Historical vs. Synthetic Data:**
   - **Data-Observed:** 43,409 real HumAID disaster social media reports, disaster events, and humanitarian classification labels.
   - **Rule-Based Heuristic:** Conflict discrepancy signals and priority review scores [0–100].
   - **Synthetic Demo:** 3D coordinates, depot inventory levels, and incident resource demands.
2. **Human Authorization:**
   - All optimizer recommendations are decision-support aids for incident commanders. **Real-world dispatch is never triggered.**
