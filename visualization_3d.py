"""
RESCUEGRID - 3D Emergency Command Center Visualization Engine
============================================================
Renders interactive 3D digital-twin command scenes representing disaster response
environments, incident towers, central resource depots, and allocation conduits.

Supports two rendering backends:
1. Three.js WebGL Digital Twin (Hardware accelerated, spatial response grid).
2. Plotly 3D Operations Scene (Offline-first fallback with interactive controls).
Strictly adheres to the tactical dark command center design system with zero emojis.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import plotly.graph_objects as go

from optimization.models import Incident, Resource, AllocationPlan
import theme


def generate_plotly_3d_figure(
    incidents: List[Incident],
    inventory: List[Resource],
    plan: AllocationPlan,
    selected_incident_id: Optional[str] = None,
    show_critical: bool = True,
    show_elevated: bool = True,
    show_standard: bool = True,
    show_conduits: bool = True,
    show_depot: bool = True,
) -> go.Figure:
    """
    Generate an offline-capable interactive Plotly 3D scene of the response area.

    Components:
    - Ground grid plane (dark slate with tactical coordinates)
    - Central Resource Logistics Depot (gold diamond node at origin)
    - Incident 3D markers (elevated by priority score, colored by urgency)
    - Allocation conduit vectors connecting Depot to Incidents
    """
    fig = go.Figure()
    n_incidents = len(incidents)

    # Dynamic marker sizing based on scenario scale
    if n_incidents <= 30:
        base_marker_size = 14
        show_all_text_labels = True
    elif n_incidents <= 120:
        base_marker_size = 9
        show_all_text_labels = False
    elif n_incidents <= 300:
        base_marker_size = 6
        show_all_text_labels = False
    else:  # 500 incidents stress test
        base_marker_size = 4
        show_all_text_labels = False

    # 1. Central Logistics Depot at (0, 0, 0)
    if show_depot:
        total_stock = sum(r.available_quantity for r in inventory)
        depot_hover = f"<b>LOGISTICS DEPOT</b><br>Available Stock: {total_stock} units<br>"
        for r in inventory:
            depot_hover += f"- {r.resource_type.title()}: {r.available_quantity}<br>"

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            marker=dict(
                size=base_marker_size + 4,
                color=theme.DEPOT_GOLD,
                symbol="diamond",
                line=dict(color="#FFFFFF", width=1.5),
                opacity=0.95,
            ),
            text=["DEPOT"],
            textposition="top center",
            textfont=dict(color=theme.DEPOT_GOLD, size=10, family="monospace"),
            name="Logistics Depot",
            hoverinfo="text",
            hovertext=[depot_hover],
        ))

    # 2. Categorized Incident Nodes by Severity Status
    groups = {
        "selected": {"x": [], "y": [], "z": [], "texts": [], "hovers": [], "color": theme.ACCENT_CYAN, "name": "Selected Site", "size": base_marker_size + 4, "show": True},
        "critical": {"x": [], "y": [], "z": [], "texts": [], "hovers": [], "color": theme.CRITICAL_RED, "name": "Critical Priority (>=80)", "size": base_marker_size + 1, "show": show_critical},
        "elevated": {"x": [], "y": [], "z": [], "texts": [], "hovers": [], "color": theme.WARNING_AMBER, "name": "High Priority (50-79)", "size": base_marker_size, "show": show_elevated},
        "normal":   {"x": [], "y": [], "z": [], "texts": [], "hovers": [], "color": theme.SUCCESS_GREEN, "name": "Standard Priority (<50)", "size": max(3, base_marker_size - 2), "show": show_standard},
    }

    line_x, line_y, line_z = [], [], []

    for inc in incidents:
        x, y, _ = inc.coordinates
        z = inc.priority_score

        alloc_map = plan.allocations.get(inc.incident_id, {})
        unmet_map = plan.unmet.get(inc.incident_id, {})
        tot_alloc = sum(alloc_map.values())
        tot_unmet = sum(unmet_map.values())
        tot_req = inc.total_demand

        hover = (
            f"<b>SITE: {inc.incident_id}</b><br>"
            f"Priority Score: <b>{inc.priority_score:.1f}</b>/100<br>"
            f"Event: {inc.event_name}<br>"
            f"Category: {inc.category}<br>"
            f"Fulfillment: <b>{tot_alloc}/{tot_req}</b> units<br>"
            f"Unmet Shortage: <b style='color: {theme.CRITICAL_RED if tot_unmet > 0 else theme.SUCCESS_GREEN};'>{tot_unmet}</b> units<br>"
            f"Status: {'CRITICAL' if inc.priority_score >= 80 else ('HIGH' if inc.priority_score >= 50 else 'STANDARD')}"
        )

        is_selected = (selected_incident_id == inc.incident_id)
        if is_selected:
            g_key = "selected"
        elif inc.priority_score >= 80:
            g_key = "critical"
        elif inc.priority_score >= 50:
            g_key = "elevated"
        else:
            g_key = "normal"

        if groups[g_key]["show"]:
            groups[g_key]["x"].append(x)
            groups[g_key]["y"].append(y)
            groups[g_key]["z"].append(z)
            # Display text label if selected or in small scenario
            label_text = inc.incident_id if (is_selected or show_all_text_labels) else ""
            groups[g_key]["texts"].append(label_text)
            groups[g_key]["hovers"].append(hover)

            # Allocation Conduits from Depot (0, 0, 0) to Incident (x, y, z)
            if show_conduits and tot_alloc > 0:
                line_x.extend([0, x, None])
                line_y.extend([0, y, None])
                line_z.extend([0, z, None])

    # Add Categorized Incident Traces
    for g_key, g_data in groups.items():
        if g_data["x"] and g_data["show"]:
            mode = "markers+text" if (g_key == "selected" or show_all_text_labels) else "markers"
            fig.add_trace(go.Scatter3d(
                x=g_data["x"], y=g_data["y"], z=g_data["z"],
                mode=mode,
                marker=dict(
                    size=g_data["size"],
                    color=g_data["color"],
                    symbol="diamond" if g_key == "selected" else "circle",
                    line=dict(color="#FFFFFF", width=1.5 if g_key == "selected" else 0.5),
                    opacity=0.92,
                ),
                text=g_data["texts"] if mode == "markers+text" else None,
                textposition="top center",
                textfont=dict(color=theme.TEXT_MAIN, size=9, family="monospace"),
                name=g_data["name"],
                hoverinfo="text",
                hovertext=g_data["hovers"],
            ))

    # Add Allocation Conduits (Lines from depot to incident sites)
    if show_conduits and line_x:
        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode="lines",
            line=dict(color=theme.ACCENT_CYAN, width=2, dash="solid"),
            name="Allocation Conduits",
            hoverinfo="none",
        ))

    # Tactical Scene Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=theme.APP_BG,
        scene=dict(
            bgcolor=theme.PANEL_BG,
            xaxis=dict(
                title=dict(text="Sector X (km)", font=dict(size=10, color=theme.TEXT_SECONDARY)),
                backgroundcolor=theme.PANEL_BG,
                gridcolor=theme.PANEL_BORDER,
                showbackground=True,
                zerolinecolor=theme.PANEL_BORDER,
                tickfont=dict(size=9, color=theme.TEXT_SECONDARY),
            ),
            yaxis=dict(
                title=dict(text="Sector Y (km)", font=dict(size=10, color=theme.TEXT_SECONDARY)),
                backgroundcolor=theme.PANEL_BG,
                gridcolor=theme.PANEL_BORDER,
                showbackground=True,
                zerolinecolor=theme.PANEL_BORDER,
                tickfont=dict(size=9, color=theme.TEXT_SECONDARY),
            ),
            zaxis=dict(
                title=dict(text="Priority Elevation (Z)", font=dict(size=10, color=theme.TEXT_SECONDARY)),
                backgroundcolor=theme.APP_BG,
                gridcolor=theme.PANEL_BORDER,
                showbackground=True,
                zerolinecolor=theme.PANEL_BORDER,
                range=[0, 105],
                tickfont=dict(size=9, color=theme.TEXT_SECONDARY),
            ),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
                center=dict(x=0, y=0, z=0.2),
            ),
            aspectratio=dict(x=1.2, y=1.2, z=0.8),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=540,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(16, 27, 45, 0.85)",
            bordercolor=theme.PANEL_BORDER,
            borderwidth=1,
            x=0.02,
            y=0.98,
            font=dict(size=10, color=theme.TEXT_SECONDARY),
        ),
    )

    return fig


def generate_threejs_html(
    incidents: List[Incident],
    inventory: List[Resource],
    plan: AllocationPlan,
    selected_incident_id: Optional[str] = None,
) -> str:
    """
    Generate an embedded Three.js WebGL digital-twin scene with glowing grid,
    pulsing tactical pillars, central logistics depot, and allocation arcs.
    Strictly zero emojis in HUD overlay and card inspects.
    """
    incidents_data = []
    for inc in incidents:
        x, y, _ = inc.coordinates
        alloc_map = plan.allocations.get(inc.incident_id, {})
        unmet_map = plan.unmet.get(inc.incident_id, {})
        incidents_data.append({
            "id": inc.incident_id,
            "x": float(x),
            "y": float(y),
            "priority": float(inc.priority_score),
            "category": inc.category,
            "total_demand": inc.total_demand,
            "total_allocated": sum(alloc_map.values()),
            "total_unmet": sum(unmet_map.values()),
            "selected": (inc.incident_id == selected_incident_id),
            "is_critical": (inc.priority_score >= 80.0),
        })

    inventory_data = [r.to_dict() for r in inventory]
    payload_json = json.dumps({
        "incidents": incidents_data,
        "inventory": inventory_data,
        "solver": plan.solver_type,
        "selected_id": selected_incident_id or "",
    })

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: {theme.APP_BG};
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            }}
            #canvas-container {{
                width: 100vw;
                height: 100vh;
                position: absolute;
            }}
            .hud-overlay {{
                position: absolute;
                top: 10px;
                left: 12px;
                color: {theme.ACCENT_CYAN};
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.08em;
                background: rgba(16, 27, 45, 0.9);
                padding: 8px 12px;
                border-radius: 4px;
                border: 1px solid {theme.PANEL_BORDER};
                border-left: 3px solid {theme.ACCENT_CYAN};
                pointer-events: none;
                z-index: 10;
            }}
            .hud-instructions {{
                position: absolute;
                bottom: 10px;
                right: 12px;
                color: {theme.TEXT_SECONDARY};
                font-size: 10px;
                background: rgba(16, 27, 45, 0.85);
                padding: 5px 10px;
                border-radius: 4px;
                border: 1px solid {theme.PANEL_BORDER};
                pointer-events: none;
            }}
            #inspect-card {{
                position: absolute;
                top: 10px;
                right: 12px;
                background: rgba(16, 27, 45, 0.95);
                border: 1px solid {theme.ACCENT_CYAN};
                border-radius: 4px;
                padding: 10px 14px;
                color: {theme.TEXT_MAIN};
                font-size: 11px;
                max-width: 230px;
                display: none;
                z-index: 20;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            }}
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div class="hud-overlay">
            <div>RESCUEGRID 3D TACTICAL RESPONSE GRID</div>
            <div style="color: {theme.TEXT_SECONDARY}; font-size: 9px; margin-top: 2px;">ENGINE: WEBGL HARDWARE ACCELERATED</div>
        </div>
        <div id="inspect-card"></div>
        <div class="hud-instructions">Left Drag: Orbit | Right Drag: Pan | Scroll: Zoom</div>
        <div id="canvas-container"></div>

        <script>
            const DATA = {payload_json};
            const container = document.getElementById('canvas-container');
            const inspectCard = document.getElementById('inspect-card');

            if (typeof THREE === 'undefined') {{
                container.innerHTML = `
                    <div style="color: {theme.TEXT_MAIN}; padding: 50px 20px; text-align: center;">
                        <h4 style="color: {theme.CRITICAL_RED}; margin-bottom: 8px;">WebGL Three.js CDN Unavailable</h4>
                        <p style="color: {theme.TEXT_SECONDARY}; font-size: 12px; max-width: 460px; margin: 0 auto 16px auto;">
                            External WebGL library could not be loaded from CDN. Switch to the <b>Plotly 3D</b> engine
                            using the 3D View toggle in the left simulation controls.
                        </p>
                    </div>
                `;
            }} else {{
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x080F1C);
                scene.fog = new THREE.FogExp2(0x080F1C, 0.004);

                const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
                camera.position.set(70, 80, 90);

                const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                renderer.shadowMap.enabled = true;
                container.appendChild(renderer.domElement);

                let controls;
                if (typeof THREE.OrbitControls !== 'undefined') {{
                    controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;
                    controls.maxPolarAngle = Math.PI / 2.05;
                    controls.target.set(0, 10, 0);
                }}

                const ambientLight = new THREE.AmbientLight(0x223344, 1.2);
                scene.add(ambientLight);

                const dirLight = new THREE.DirectionalLight(0x00D5E8, 1.5);
                dirLight.position.set(50, 100, 50);
                scene.add(dirLight);

                const pointLight = new THREE.PointLight(0xF5A623, 2, 80);
                pointLight.position.set(0, 20, 0);
                scene.add(pointLight);

                // Grid Floor
                const gridHelper = new THREE.GridHelper(160, 32, 0x00D5E8, 0x26374D);
                gridHelper.position.y = 0;
                scene.add(gridHelper);

                // Central Logistics Depot
                const depotGeo = new THREE.CylinderGeometry(6, 7, 4, 16);
                const depotMat = new THREE.MeshStandardMaterial({{
                    color: 0x142238,
                    emissive: 0x00D5E8,
                    emissiveIntensity: 0.3,
                    roughness: 0.3,
                    metalness: 0.8
                }});
                const depot = new THREE.Mesh(depotGeo, depotMat);
                depot.position.set(0, 2, 0);
                scene.add(depot);

                const ringGeo = new THREE.RingGeometry(8, 9, 32);
                const ringMat = new THREE.MeshBasicMaterial({{
                    color: 0x00D5E8,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.6
                }});
                const ring = new THREE.Mesh(ringGeo, ringMat);
                ring.rotation.x = Math.PI / 2;
                ring.position.y = 0.2;
                scene.add(ring);

                const interactiveObjects = [];

                DATA.incidents.forEach(inc => {{
                    const height = Math.max(6, inc.priority * 0.35);
                    const pillarGeo = new THREE.BoxGeometry(4, height, 4);

                    let colorHex = 0x2ECC71; // Green
                    if (inc.selected) colorHex = 0x00D5E8;
                    else if (inc.is_critical) colorHex = 0xF04452; // Red
                    else if (inc.priority >= 50) colorHex = 0xF5A623; // Amber

                    const pillarMat = new THREE.MeshStandardMaterial({{
                        color: colorHex,
                        emissive: colorHex,
                        emissiveIntensity: inc.selected ? 0.6 : 0.2,
                        roughness: 0.3,
                        metalness: 0.7
                    }});

                    const pillar = new THREE.Mesh(pillarGeo, pillarMat);
                    pillar.position.set(inc.x, height / 2, inc.y);
                    pillar.userData = inc;
                    scene.add(pillar);
                    interactiveObjects.push(pillar);

                    const incRingGeo = new THREE.RingGeometry(3, 4, 16);
                    const incRingMat = new THREE.MeshBasicMaterial({{
                        color: colorHex,
                        side: THREE.DoubleSide,
                        transparent: true,
                        opacity: 0.35
                    }});
                    const incRing = new THREE.Mesh(incRingGeo, incRingMat);
                    incRing.rotation.x = Math.PI / 2;
                    incRing.position.set(inc.x, 0.1, inc.y);
                    scene.add(incRing);

                    // Allocation Arc
                    if (inc.total_allocated > 0) {{
                        const curve = new THREE.QuadraticBezierCurve3(
                            new THREE.Vector3(0, 3, 0),
                            new THREE.Vector3(inc.x * 0.5, height + 15, inc.y * 0.5),
                            new THREE.Vector3(inc.x, height, inc.y)
                        );
                        const points = curve.getPoints(24);
                        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
                        const lineMat = new THREE.LineBasicMaterial({{
                            color: 0x00D5E8,
                            transparent: true,
                            opacity: 0.75,
                            linewidth: 2
                        }});
                        const arc = new THREE.Line(lineGeo, lineMat);
                        scene.add(arc);
                    }}
                }});

                const raycaster = new THREE.Raycaster();
                const mouse = new THREE.Vector2();

                function onMouseMove(event) {{
                    const rect = renderer.domElement.getBoundingClientRect();
                    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects(interactiveObjects);

                    if (intersects.length > 0) {{
                        const d = intersects[0].object.userData;
                        inspectCard.style.display = 'block';
                        inspectCard.innerHTML = `
                            <div style="font-weight: 700; color: {theme.ACCENT_CYAN};">#${{d.id}}</div>
                            <div style="font-size: 10px; color: {theme.TEXT_SECONDARY}; margin-bottom: 4px;">PRIORITY: ${{d.priority.toFixed(1)}}/100</div>
                            <div>Allocated: <b>${{d.total_allocated}} / ${{d.total_demand}}</b> units</div>
                            <div>Unmet Deficit: <b style="color: ${{d.total_unmet > 0 ? '{theme.CRITICAL_RED}' : '{theme.SUCCESS_GREEN}'}}">${{d.total_unmet}}</b> units</div>
                            <div style="font-size: 9px; color: {theme.ACCENT_CYAN}; margin-top: 4px;">Selected in command dossier</div>
                        `;
                        document.body.style.cursor = 'pointer';
                    }} else {{
                        inspectCard.style.display = 'none';
                        document.body.style.cursor = 'default';
                    }}
                }}

                window.addEventListener('mousemove', onMouseMove);

                let clock = new THREE.Clock();
                function animate() {{
                    requestAnimationFrame(animate);
                    const t = clock.getElapsedTime();
                    ring.rotation.z = t * 0.4;
                    if (controls) controls.update();
                    renderer.render(scene, camera);
                }}
                animate();

                window.addEventListener('resize', () => {{
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }});
            }}
        </script>
    </body>
    </html>
    """
    return html_code
