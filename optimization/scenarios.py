"""
RESCUEGRID - Simulation Scenarios & Benchmark Datasets
=====================================================
Provides reproducible emergency scenarios for evaluating optimization and greedy
allocation policies under various disaster conditions (shortage, bottleneck, equity).
"""

from __future__ import annotations

from typing import Dict, List, Optional
from optimization.models import Incident, Resource, Scenario


def get_all_scenarios() -> Dict[str, Scenario]:
    """Return dictionary of all available simulation scenarios."""
    scenarios = [
        # 1. Sufficient Stock Scenario
        Scenario(
            scenario_id="SCENARIO_SUFFICIENT_STOCK",
            name="1. Sufficient Inventory (All Demands Met)",
            description="Abundant regional stockpile capable of 100% satisfying all incident requests.",
            incidents=[
                Incident("INC-101", 92.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 4}, "urgent_needs", "hurricane_harvey_2017", (-30.0, 15.0, 0.0), "Life-threatening flooding; 4 residents trapped on roof.", True),
                Incident("INC-102", 78.0, {"ambulance": 1, "rescue_team": 2, "water_kit": 3}, "infrastructure_damage", "hurricane_harvey_2017", (25.0, 30.0, 0.0), "Bridge collapsed; medical transport needed.", True),
                Incident("INC-103", 65.0, {"ambulance": 1, "rescue_team": 1, "water_kit": 5}, "evacuation", "hurricane_harvey_2017", (40.0, -20.0, 0.0), "Community shelter requests drinking water and basic triage.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=10, urgency_weight=1.5, color_hex="#EF4444", icon="🚑"),
                Resource("rescue_team", available_quantity=8, urgency_weight=1.3, color_hex="#F59E0B", icon="🚒"),
                Resource("water_kit", available_quantity=20, urgency_weight=1.0, color_hex="#0284C7", icon="💧"),
            ],
            notes="Both Greedy and MILP achieve 100% fulfillment. Verifies that optimization does not degrade under surplus.",
        ),

        # 2. Severe Resource Shortage Scenario
        Scenario(
            scenario_id="SCENARIO_SEVERE_SHORTAGE",
            name="2. Severe Resource Shortage (~30% Stock)",
            description="Catastrophic earthquake aftermath with severe regional supply deficit across 5 competing sites.",
            incidents=[
                Incident("INC-201", 96.0, {"ambulance": 4, "rescue_team": 3, "water_kit": 10}, "casualties", "ecuador_earthquake_2016", (-45.0, 20.0, 0.0), "Major hospital damaged; severe casualties requiring urgent ICU transit.", True),
                Incident("INC-202", 89.0, {"ambulance": 3, "rescue_team": 2, "water_kit": 8}, "urgent_needs", "ecuador_earthquake_2016", (10.0, 45.0, 0.0), "Multiple residential buildings collapsed; victims trapped under rubble.", True),
                Incident("INC-203", 79.0, {"ambulance": 2, "rescue_team": 2, "water_kit": 6}, "infrastructure_damage", "ecuador_earthquake_2016", (35.0, -15.0, 0.0), "Power and water grid severed; nursing home without cooling.", False),
                Incident("INC-204", 68.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 8}, "evacuation", "ecuador_earthquake_2016", (-20.0, -35.0, 0.0), "Informal settlement landslide risk; preventive evacuation.", False),
                Incident("INC-205", 52.0, {"ambulance": 1, "rescue_team": 1, "water_kit": 12}, "general_information", "ecuador_earthquake_2016", (50.0, 25.0, 0.0), "Displaced families gathering in public square.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=4, urgency_weight=1.5, color_hex="#EF4444", icon="🚑"),
                Resource("rescue_team", available_quantity=3, urgency_weight=1.3, color_hex="#F59E0B", icon="🚒"),
                Resource("water_kit", available_quantity=12, urgency_weight=1.0, color_hex="#0284C7", icon="💧"),
            ],
            notes="Tests triage trade-offs under extreme deficit. Highlights unmet shortages per incident.",
        ),

        # 3. Multiple Competing Critical Incidents
        Scenario(
            scenario_id="SCENARIO_COMPETING_CRITICAL",
            name="3. Competing High-Priority Incidents (Priority >= 85)",
            description="Four simultaneously critical incidents competing for life-saving assets.",
            incidents=[
                Incident("INC-301", 95.0, {"ambulance": 3, "rescue_team": 2, "water_kit": 2}, "casualties", "greece_wildfires_2018", (-35.0, 25.0, 0.0), "Wildfire cut off coastal village; severe burns reported.", True),
                Incident("INC-302", 92.0, {"ambulance": 3, "rescue_team": 2, "water_kit": 2}, "urgent_needs", "greece_wildfires_2018", (20.0, 40.0, 0.0), "Convoy trapped by flames on mountain pass.", True),
                Incident("INC-303", 88.0, {"ambulance": 2, "rescue_team": 2, "water_kit": 2}, "evacuation", "greece_wildfires_2018", (35.0, -25.0, 0.0), "Children's summer camp under immediate evacuation order.", True),
                Incident("INC-304", 86.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 2}, "rescue_support", "greece_wildfires_2018", (-15.0, -40.0, 0.0), "Elderly care facility surrounded by smoke.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=5, urgency_weight=1.5, color_hex="#EF4444", icon="🚑"),
                Resource("rescue_team", available_quantity=4, urgency_weight=1.3, color_hex="#F59E0B", icon="🚒"),
                Resource("water_kit", available_quantity=8, urgency_weight=1.0, color_hex="#0284C7", icon="💧"),
            ],
            notes="Evaluates tie-breaking and trade-off mechanics when all competing incidents are critical.",
        ),

        # 4. Zero Inventory Scenario (Edge Case)
        Scenario(
            scenario_id="SCENARIO_ZERO_INVENTORY",
            name="4. Complete Stockout (Zero Inventory Edge Case)",
            description="No resource units available in any depot.",
            incidents=[
                Incident("INC-401", 90.0, {"ambulance": 2, "rescue_team": 1, "water_kit": 4}, "urgent_needs", "cyclone_idai_2019", (0.0, 30.0, 0.0), "Flash flooding cut off island; urgent aid requested.", True),
                Incident("INC-402", 70.0, {"ambulance": 1, "rescue_team": 2, "water_kit": 5}, "infrastructure_damage", "cyclone_idai_2019", (30.0, -10.0, 0.0), "Roof torn off clinic.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=0, urgency_weight=1.5, color_hex="#EF4444", icon="🚑"),
                Resource("rescue_team", available_quantity=0, urgency_weight=1.3, color_hex="#F59E0B", icon="🚒"),
                Resource("water_kit", available_quantity=0, urgency_weight=1.0, color_hex="#0284C7", icon="💧"),
            ],
            notes="Validates solver safety: must terminate with 0% fulfillment and zero division safety.",
        ),

        # 5. Zero Demand Scenario (Edge Case)
        Scenario(
            scenario_id="SCENARIO_ZERO_DEMAND",
            name="5. Zero Demand Ingested (Edge Case)",
            description="Incidents report zero resource needs (informational reports only).",
            incidents=[
                Incident("INC-501", 40.0, {"ambulance": 0, "rescue_team": 0, "water_kit": 0}, "general_information", "hurricane_irma_2017", (-20.0, 20.0, 0.0), "Weather update; no immediate intervention needed.", False),
                Incident("INC-502", 30.0, {"ambulance": 0, "rescue_team": 0, "water_kit": 0}, "emotional_support", "hurricane_irma_2017", (20.0, -20.0, 0.0), "Community prayer message.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=5, urgency_weight=1.5, color_hex="#EF4444", icon="🚑"),
                Resource("rescue_team", available_quantity=5, urgency_weight=1.3, color_hex="#F59E0B", icon="🚒"),
                Resource("water_kit", available_quantity=10, urgency_weight=1.0, color_hex="#0284C7", icon="💧"),
            ],
            notes="Validates zero demand handling: 100% satisfied with 0 units allocated.",
        ),

        # 6. Bottleneck Resource Deficit
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
                Resource("ambulance", available_quantity=2, urgency_weight=1.5, color_hex="#EF4444", icon="🚑"),  # Bottleneck: 2 available for 7 requested
                Resource("rescue_team", available_quantity=10, urgency_weight=1.3, color_hex="#F59E0B", icon="🚒"),
                Resource("water_kit", available_quantity=50, urgency_weight=1.0, color_hex="#0284C7", icon="💧"),
            ],
            notes="Highlights single-resource vulnerability: Ambulance fulfillment is low while Water Kits are 100% fulfilled.",
        ),

        # 7. Greedy Starvation vs. MILP Equitable Allocation (Divergence Showcase)
        Scenario(
            scenario_id="SCENARIO_GREEDY_VS_OPTIMIZER",
            name="7. Greedy Starvation vs. Equitable MILP (Divergence Demo)",
            description="A high-demand incident completely starves other sites under greedy triage, whereas MILP ensures equitable service coverage.",
            incidents=[
                Incident("INC-701", 95.0, {"ambulance": 4, "rescue_team": 2, "water_kit": 4}, "urgent_needs", "hurricane_maria_2017", (-40.0, 10.0, 0.0), "Large community hospital complex requesting all available regional units.", True),
                Incident("INC-702", 76.0, {"ambulance": 1, "rescue_team": 1, "water_kit": 2}, "casualties", "hurricane_maria_2017", (15.0, 40.0, 0.0), "Rural clinic isolated; critical patient stabilization.", False),
                Incident("INC-703", 64.0, {"ambulance": 1, "rescue_team": 1, "water_kit": 2}, "infrastructure_damage", "hurricane_maria_2017", (35.0, -30.0, 0.0), "Emergency shelter backup generator flooded.", False),
            ],
            inventory=[
                Resource("ambulance", available_quantity=4, urgency_weight=1.5, color_hex="#EF4444", icon="🚑"),
                Resource("rescue_team", available_quantity=2, urgency_weight=1.3, color_hex="#F59E0B", icon="🚒"),
                Resource("water_kit", available_quantity=6, urgency_weight=1.0, color_hex="#0284C7", icon="💧"),
            ],
            notes="Under Greedy, INC-701 consumes 100% of available ambulances and rescue teams, completely starving INC-702 and INC-703. Under MILP with Minimum Service Guarantee, every incident receives vital resources.",
        ),
    ]

    return {s.scenario_id: s for s in scenarios}
