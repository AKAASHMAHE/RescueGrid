"""
RESCUEGRID - Simulation Scenarios & Benchmark Datasets
=====================================================
Provides reproducible emergency scenarios for evaluating optimization and greedy
allocation policies under various disaster conditions (shortage, bottleneck, equity).
Supports deterministic scaling from Small (25), Medium (100), Large (250) to Stress Test (500).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple
from optimization.models import Incident, Resource, Scenario


# -----------------------------------------------------------------------------
# 1. RESOURCE DEFINITIONS CATALOG (Zero Emoji)
# -----------------------------------------------------------------------------
ALL_RESOURCE_CATALOG: List[Resource] = [
    Resource("ambulance", available_quantity=0, urgency_weight=1.5, color_hex="#F04452", icon=""),
    Resource("rescue_team", available_quantity=0, urgency_weight=1.3, color_hex="#F5A623", icon=""),
    Resource("water_kit", available_quantity=0, urgency_weight=1.0, color_hex="#00D5E8", icon=""),
    Resource("medical_kit", available_quantity=0, urgency_weight=1.4, color_hex="#E056FD", icon=""),
    Resource("food_ration", available_quantity=0, urgency_weight=0.9, color_hex="#2ECC71", icon=""),
    Resource("shelter_pack", available_quantity=0, urgency_weight=1.1, color_hex="#E67E22", icon=""),
    Resource("power_generator", available_quantity=0, urgency_weight=1.2, color_hex="#F1C40F", icon=""),
    Resource("hazmat_unit", available_quantity=0, urgency_weight=1.6, color_hex="#9B59B6", icon=""),
    Resource("communication_node", available_quantity=0, urgency_weight=1.1, color_hex="#3498DB", icon=""),
    Resource("boat_rescue", available_quantity=0, urgency_weight=1.4, color_hex="#1ABC9C", icon=""),
]

SCALE_CONFIGS: Dict[str, Dict[str, int]] = {
    "Small": {"incidents": 25, "resources": 5},
    "Medium": {"incidents": 100, "resources": 7},
    "Large": {"incidents": 250, "resources": 9},
    "Stress Test": {"incidents": 500, "resources": 10},
}

CATEGORIES = [
    "urgent_needs", "casualties", "infrastructure_damage",
    "evacuation", "rescue_support", "shelter_needs", "general_information"
]

EVENT_NAMES = [
    "hurricane_harvey_2017", "ecuador_earthquake_2016", "greece_wildfires_2018",
    "cyclone_idai_2019", "puebla_mexico_earthquake_2017", "hurricane_maria_2017"
]


# -----------------------------------------------------------------------------
# 2. LARGE-SCALE SYNTHETIC SCENARIO GENERATOR
# -----------------------------------------------------------------------------
def generate_scaled_scenario(
    preset_type: str,
    scale_name: str = "Small",
    seed: int = 42,
) -> Scenario:
    """
    Generate reproducible, scalable simulation scenarios with deterministic random seeds.

    Parameters:
    -----------
    preset_type : str
        One of 'ADEQUATE_STOCK', 'SEVERE_SHORTAGE', 'COMPETING_CRITICAL',
        'AMBULANCE_BOTTLENECK', 'RESCUE_BOTTLENECK', 'WATER_SHORTAGE',
        'MIXED_PRIORITIES', 'GREEDY_STARVATION', 'ZERO_INVENTORY', 'ZERO_DEMAND'
    scale_name : str
        'Small' (25), 'Medium' (100), 'Large' (250), 'Stress Test' (500)
    seed : int
        Deterministic random seed for reproducibility across Streamlit reruns.
    """
    cfg = SCALE_CONFIGS.get(scale_name, SCALE_CONFIGS["Small"])
    n_incidents = cfg["incidents"]
    n_resources = cfg["resources"]

    rng = random.Random(seed + hash(preset_type) % 10000)

    active_res_catalog = ALL_RESOURCE_CATALOG[:n_resources]
    res_names = [r.resource_type for r in active_res_catalog]

    incidents: List[Incident] = []
    total_demands_per_res = {r: 0 for r in res_names}

    # Generate incidents with deterministic coordinates on tactical grid
    for idx in range(n_incidents):
        inc_id = f"INC-{idx + 1:04d}"

        # Deterministic spatial grid coordinates centered around depot (0, 0, 0)
        grid_radius = 80.0 if n_incidents <= 100 else 140.0
        angle = (idx * 137.5) * (3.14159265 / 180.0)  # Golden angle distribution
        dist = (idx / n_incidents) ** 0.5 * grid_radius + 10.0
        x = dist * rng.uniform(0.85, 1.15) * 1.0 * (1 if idx % 2 == 0 else -1)
        y = dist * rng.uniform(0.85, 1.15) * 1.0 * (1 if (idx // 2) % 2 == 0 else -1)

        # Priority calculation based on preset
        if preset_type == "COMPETING_CRITICAL":
            priority = rng.uniform(82.0, 99.0) if rng.random() < 0.75 else rng.uniform(50.0, 80.0)
        elif preset_type == "ZERO_DEMAND":
            priority = rng.uniform(20.0, 45.0)
        elif preset_type == "GREEDY_STARVATION":
            # Site 1 has massive demand and highest priority
            priority = 96.0 if idx == 0 else rng.uniform(55.0, 78.0)
        else:
            priority = rng.uniform(30.0, 98.0)

        # Resource demand quantities
        demands: Dict[str, int] = {}
        for r in res_names:
            if preset_type == "ZERO_DEMAND":
                dem = 0
            elif preset_type == "GREEDY_STARVATION" and idx == 0:
                dem = rng.randint(8, 15)  # Massive single-site demand
            else:
                dem = rng.randint(0, 5) if rng.random() < 0.75 else rng.randint(4, 9)
            demands[r] = dem
            total_demands_per_res[r] += dem

        cat = CATEGORIES[idx % len(CATEGORIES)]
        ev = EVENT_NAMES[idx % len(EVENT_NAMES)]
        ev_summary = f"Simulated {cat.replace('_', ' ')} incident report at grid sector {idx + 1}."

        inc = Incident(
            incident_id=inc_id,
            priority_score=round(priority, 1),
            demands=demands,
            category=cat,
            event_name=ev,
            coordinates=(round(x, 1), round(y, 1), 0.0),
            evidence_summary=ev_summary,
            verified=(priority >= 75.0),
        )
        incidents.append(inc)

    # Independent Inventory Generation based on preset conditions
    inventory: List[Resource] = []
    for base_res in active_res_catalog:
        r_name = base_res.resource_type
        tot_dem = total_demands_per_res[r_name]

        if preset_type == "ZERO_INVENTORY":
            avail = 0
        elif preset_type == "ADEQUATE_STOCK":
            avail = int(tot_dem * 1.25) + 5
        elif preset_type == "SEVERE_SHORTAGE":
            avail = max(1, int(tot_dem * 0.32))
        elif preset_type == "AMBULANCE_BOTTLENECK":
            avail = max(1, int(tot_dem * 0.15)) if r_name == "ambulance" else int(tot_dem * 1.1)
        elif preset_type == "RESCUE_BOTTLENECK":
            avail = max(1, int(tot_dem * 0.15)) if r_name == "rescue_team" else int(tot_dem * 1.1)
        elif preset_type == "WATER_SHORTAGE":
            avail = max(1, int(tot_dem * 0.20)) if r_name == "water_kit" else int(tot_dem * 1.1)
        elif preset_type == "GREEDY_STARVATION":
            first_site_dem = incidents[0].demands.get(r_name, 0)
            avail = max(2, first_site_dem + 1)
        else:  # MIXED_PRIORITIES or default
            avail = max(1, int(tot_dem * rng.uniform(0.45, 0.75)))

        res = Resource(
            resource_type=r_name,
            available_quantity=avail,
            urgency_weight=base_res.urgency_weight,
            unit_cost=base_res.unit_cost,
            color_hex=base_res.color_hex,
            icon="",
        )
        inventory.append(res)

    preset_titles = {
        "ADEQUATE_STOCK": f"Adequate Inventory (Surplus, {scale_name}: {n_incidents} Sites)",
        "SEVERE_SHORTAGE": f"Severe Resource Shortage (~30% Stock, {scale_name}: {n_incidents} Sites)",
        "COMPETING_CRITICAL": f"Competing Critical Incidents ({scale_name}: {n_incidents} Sites)",
        "AMBULANCE_BOTTLENECK": f"Ambulance Resource Bottleneck ({scale_name}: {n_incidents} Sites)",
        "RESCUE_BOTTLENECK": f"Rescue Team Urban Deficit ({scale_name}: {n_incidents} Sites)",
        "WATER_SHORTAGE": f"Water Supply Shortage ({scale_name}: {n_incidents} Sites)",
        "MIXED_PRIORITIES": f"Mixed Priorities & Demands ({scale_name}: {n_incidents} Sites)",
        "GREEDY_STARVATION": f"Greedy Starvation vs. Equitable MILP ({scale_name}: {n_incidents} Sites)",
        "ZERO_INVENTORY": f"Complete Stockout (Zero Inventory, {scale_name}: {n_incidents} Sites)",
        "ZERO_DEMAND": f"Zero Demand Ingested ({scale_name}: {n_incidents} Sites)",
    }

    sc_id = f"SCENARIO_{preset_type}_{scale_name.upper().replace(' ', '_')}"
    name = preset_titles.get(preset_type, f"{preset_type} ({scale_name})")
    desc = f"Reproducible {scale_name} emergency response simulation with {n_incidents} incidents across {n_resources} resource categories."

    return Scenario(
        scenario_id=sc_id,
        name=name,
        description=desc,
        incidents=incidents,
        inventory=inventory,
        notes=f"Deterministic simulation generated under seed {seed}.",
    )


# -----------------------------------------------------------------------------
# 3. MASTER SCENARIO REGISTRY (Static Presets + Scaled Suite)
# -----------------------------------------------------------------------------
def get_all_scenarios(scale: str = "Small") -> Dict[str, Scenario]:
    """
    Return dictionary of all available simulation scenarios.
    Maintains backward compatibility with original scenario IDs while
    supporting dynamic scale expansion (Small: 25, Medium: 100, Large: 250, Stress Test: 500).
    """
    base_scenarios = [
        Scenario(
            scenario_id="SCENARIO_SUFFICIENT_STOCK",
            name="1. Sufficient Inventory (All Demands Met)",
            description="Abundant regional stockpile capable of 100% satisfying all incident requests.",
            incidents=[
                Incident("INC-101", 92.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 4}, "urgent_needs", "hurricane_harvey_2017", (-30.0, 15.0, 0.0), "Flooding; 4 residents trapped on roof.", True),
                Incident("INC-102", 78.0, {"ambulance": 1, "rescue_team": 2, "water_kit": 3}, "infrastructure_damage", "hurricane_harvey_2017", (25.0, 30.0, 0.0), "Bridge collapsed; medical transport needed.", True),
                Incident("INC-103", 65.0, {"ambulance": 1, "rescue_team": 1, "water_kit": 5}, "evacuation", "hurricane_harvey_2017", (40.0, -20.0, 0.0), "Community shelter requests drinking water.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=10, urgency_weight=1.5, color_hex="#F04452", icon=""),
                Resource("rescue_team", available_quantity=8, urgency_weight=1.3, color_hex="#F5A623", icon=""),
                Resource("water_kit", available_quantity=20, urgency_weight=1.0, color_hex="#00D5E8", icon=""),
            ],
            notes="Both Greedy and MILP achieve 100% fulfillment. Verifies that optimization does not degrade under surplus.",
        ),

        Scenario(
            scenario_id="SCENARIO_SEVERE_SHORTAGE",
            name="2. Severe Resource Shortage (~30% Stock)",
            description="Catastrophic earthquake aftermath with severe regional supply deficit across 5 competing sites.",
            incidents=[
                Incident("INC-201", 96.0, {"ambulance": 4, "rescue_team": 3, "water_kit": 10}, "casualties", "ecuador_earthquake_2016", (-45.0, 20.0, 0.0), "Major hospital damaged; severe casualties requiring ICU transit.", True),
                Incident("INC-202", 89.0, {"ambulance": 3, "rescue_team": 2, "water_kit": 8}, "urgent_needs", "ecuador_earthquake_2016", (10.0, 45.0, 0.0), "Multiple residential buildings collapsed; victims trapped.", True),
                Incident("INC-203", 79.0, {"ambulance": 2, "rescue_team": 2, "water_kit": 6}, "infrastructure_damage", "ecuador_earthquake_2016", (35.0, -15.0, 0.0), "Power and water grid severed; nursing home without cooling.", False),
                Incident("INC-204", 68.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 8}, "evacuation", "ecuador_earthquake_2016", (-20.0, -35.0, 0.0), "Landslide risk; preventive evacuation.", False),
                Incident("INC-205", 52.0, {"ambulance": 1, "rescue_team": 1, "water_kit": 12}, "general_information", "ecuador_earthquake_2016", (50.0, 25.0, 0.0), "Displaced families gathering in public square.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=4, urgency_weight=1.5, color_hex="#F04452", icon=""),
                Resource("rescue_team", available_quantity=3, urgency_weight=1.3, color_hex="#F5A623", icon=""),
                Resource("water_kit", available_quantity=12, urgency_weight=1.0, color_hex="#00D5E8", icon=""),
            ],
            notes="Tests triage trade-offs under extreme deficit. Highlights unmet shortages per incident.",
        ),

        Scenario(
            scenario_id="SCENARIO_COMPETING_CRITICAL",
            name="3. Competing High-Priority Incidents (Priority >= 85)",
            description="Four simultaneously critical incidents competing for life-saving assets.",
            incidents=[
                Incident("INC-301", 95.0, {"ambulance": 3, "rescue_team": 2, "water_kit": 2}, "casualties", "greece_wildfires_2018", (-35.0, 25.0, 0.0), "Wildfire cut off coastal village; severe burns reported.", True),
                Incident("INC-302", 92.0, {"ambulance": 3, "rescue_team": 2, "water_kit": 2}, "urgent_needs", "greece_wildfires_2018", (20.0, 40.0, 0.0), "Convoy trapped by flames on mountain pass.", True),
                Incident("INC-303", 88.0, {"ambulance": 2, "rescue_team": 2, "water_kit": 2}, "evacuation", "greece_wildfires_2018", (35.0, -25.0, 0.0), "Children summer camp under immediate evacuation order.", True),
                Incident("INC-304", 86.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 2}, "rescue_support", "greece_wildfires_2018", (-15.0, -40.0, 0.0), "Elderly care facility surrounded by smoke.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=5, urgency_weight=1.5, color_hex="#F04452", icon=""),
                Resource("rescue_team", available_quantity=4, urgency_weight=1.3, color_hex="#F5A623", icon=""),
                Resource("water_kit", available_quantity=8, urgency_weight=1.0, color_hex="#00D5E8", icon=""),
            ],
            notes="Evaluates tie-breaking and trade-off mechanics when all competing incidents are critical.",
        ),

        Scenario(
            scenario_id="SCENARIO_ZERO_INVENTORY",
            name="4. Complete Stockout (Zero Inventory Edge Case)",
            description="No resource units available in any depot.",
            incidents=[
                Incident("INC-401", 90.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 4}, "urgent_needs", "cyclone_idai_2019", (0.0, 30.0, 0.0), "Flash flooding cut off island; urgent aid requested.", True),
                Incident("INC-402", 70.0, {"ambulance": 1, "rescue_team": 2, "water_kit": 5}, "infrastructure_damage", "cyclone_idai_2019", (30.0, -10.0, 0.0), "Roof torn off clinic.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=0, urgency_weight=1.5, color_hex="#F04452", icon=""),
                Resource("rescue_team", available_quantity=0, urgency_weight=1.3, color_hex="#F5A623", icon=""),
                Resource("water_kit", available_quantity=0, urgency_weight=1.0, color_hex="#00D5E8", icon=""),
            ],
            notes="Validates solver safety: terminates with 0% fulfillment without divide-by-zero errors.",
        ),

        Scenario(
            scenario_id="SCENARIO_ZERO_DEMAND",
            name="5. Zero Demand Ingested (Edge Case)",
            description="Incidents report zero resource needs (informational reports only).",
            incidents=[
                Incident("INC-501", 40.0, {"ambulance": 0, "rescue_team": 0, "water_kit": 0}, "general_information", "hurricane_irma_2017", (-20.0, 20.0, 0.0), "Weather update; no immediate intervention needed.", False),
                Incident("INC-502", 30.0, {"ambulance": 0, "rescue_team": 0, "water_kit": 0}, "emotional_support", "hurricane_irma_2017", (20.0, -20.0, 0.0), "Community message.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=5, urgency_weight=1.5, color_hex="#F04452", icon=""),
                Resource("rescue_team", available_quantity=5, urgency_weight=1.3, color_hex="#F5A623", icon=""),
                Resource("water_kit", available_quantity=10, urgency_weight=1.0, color_hex="#00D5E8", icon=""),
            ],
            notes="Validates zero demand handling: displays N/A for fulfillment with 0 units allocated.",
        ),

        Scenario(
            scenario_id="SCENARIO_BOTTLENECK_RESOURCE",
            name="6. Specific Resource Bottleneck (Ambulance Deficit)",
            description="Abundant general relief supplies, but a critical shortage of medical transport.",
            incidents=[
                Incident("INC-601", 94.0, {"ambulance": 3, "rescue_team": 1, "water_kit": 5}, "casualties", "puebla_mexico_earthquake_2017", (-30.0, 30.0, 0.0), "Crush injuries from building facade collapse.", True),
                Incident("INC-602", 87.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 5}, "urgent_needs", "puebla_mexico_earthquake_2017", (20.0, 20.0, 0.0), "School evacuation triage.", True),
                Incident("INC-603", 75.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 10}, "infrastructure_damage", "puebla_mexico_earthquake_2017", (0.0, -35.0, 0.0), "Gas leak in residential block.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=2, urgency_weight=1.5, color_hex="#F04452", icon=""),
                Resource("rescue_team", available_quantity=10, urgency_weight=1.3, color_hex="#F5A623", icon=""),
                Resource("water_kit", available_quantity=50, urgency_weight=1.0, color_hex="#00D5E8", icon=""),
            ],
            notes="Highlights single-resource vulnerability: Ambulance fulfillment is low while Water Kits are 100% fulfilled.",
        ),

        Scenario(
            scenario_id="SCENARIO_GREEDY_VS_OPTIMIZER",
            name="7. Greedy Starvation vs. Equitable MILP (Divergence Demo)",
            description="A high-demand incident completely starves other sites under greedy triage, whereas MILP ensures equitable service coverage.",
            incidents=[
                Incident("INC-701", 95.0, {"ambulance": 4, "rescue_team": 2, "water_kit": 4}, "urgent_needs", "hurricane_maria_2017", (-40.0, 10.0, 0.0), "Large community hospital requesting all available units.", True),
                Incident("INC-702", 76.0, {"ambulance": 1, "rescue_team": 1, "water_kit": 2}, "casualties", "hurricane_maria_2017", (15.0, 40.0, 0.0), "Rural clinic isolated; critical patient stabilization.", False),
                Incident("INC-703", 64.0, {"ambulance": 1, "rescue_team": 1, "water_kit": 2}, "infrastructure_damage", "hurricane_maria_2017", (35.0, -30.0, 0.0), "Emergency shelter backup generator flooded.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=4, urgency_weight=1.5, color_hex="#F04452", icon=""),
                Resource("rescue_team", available_quantity=2, urgency_weight=1.3, color_hex="#F5A623", icon=""),
                Resource("water_kit", available_quantity=6, urgency_weight=1.0, color_hex="#00D5E8", icon=""),
            ],
            notes="Under Greedy, INC-701 consumes 100% of available ambulances and rescue teams, completely starving INC-702 and INC-703. Under MILP with Minimum Service Guarantee, every incident receives vital resources.",
        ),
    ]

    sc_dict = {s.scenario_id: s for s in base_scenarios}

    # 2. Add Scaled Scenarios for selected scale
    scaled_presets = [
        "ADEQUATE_STOCK",
        "SEVERE_SHORTAGE",
        "COMPETING_CRITICAL",
        "AMBULANCE_BOTTLENECK",
        "RESCUE_BOTTLENECK",
        "MIXED_PRIORITIES",
    ]
    for preset in scaled_presets:
        sc = generate_scaled_scenario(preset, scale_name=scale, seed=42)
        sc_dict[sc.scenario_id] = sc

    return sc_dict
