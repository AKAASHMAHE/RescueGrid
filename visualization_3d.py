"""
RESCUEGRID - 3D Emergency Command Center Visualization Engine
============================================================
Renders interactive 3D digital-twin command scenes representing disaster response
environments, incident towers, central resource depots, and allocation conduits.

Supports two rendering backends:
1. Three.js WebGL Digital Twin (Cyberpunk / Tactical Command aesthetic).
2. Plotly 3D Operations Scene (Offline-first fallback).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import plotly.graph_objects as go

from optimization.models import Incident, Resource, AllocationPlan


def generate_plotly_3d_figure(
    incidents: List[Incident],
    inventory: List[Resource],
    plan: AllocationPlan,
    selected_incident_id: Optional[str] = None,
) -> go.Figure:
    """
    Generate an offline-capable interactive Plotly 3D scene of the response area.

    Components:
    - Ground grid plane (dark slate with tactical coordinates)
    - Central Resource Logistics Depot (diamond node at origin)
    - Incident 3D markers (elevated by priority score, colored by urgency)
    - Allocation conduit vectors connecting Depot to Incidents
    """
    fig = go.Figure()

    # 1. Central Logistics Depot at (0, 0, 0)
    total_stock = sum(r.available_quantity for r in inventory)
    depot_hover = f"<b>CENTRAL LOGISTICS DEPOT</b><br>Total Stock: {total_stock} units<br>"
    for r in inventory:
        depot_hover += f"- {r.icon} {r.resource_type}: {r.available_quantity}<br>"

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="markers+text",
        marker=dict(
            size=16,
            color="#00F0FF",
            symbol="diamond",
            line=dict(color="#FFFFFF", width=2),
            opacity=0.95,
        ),
        text=["DEPOT"],
        textposition="top center",
        textfont=dict(color="#00F0FF", size=11, family="monospace"),
        name="Logistics Depot",
        hoverinfo="text",
        hovertext=[depot_hover],
    ))

    # 2. Incident Nodes
    inc_x, inc_y, inc_z = [], [], []
    inc_colors, inc_sizes, inc_texts, inc_hovers = [], [], [], []

    line_x, line_y, line_z = [], [], []

    for inc in incidents:
        x, y, _ = inc.coordinates
        z = inc.priority_score  # Height reflects priority severity

        inc_x.append(x)
        inc_y.append(y)
        inc_z.append(z)

        # Priority Color Palette
        is_selected = (selected_incident_id == inc.incident_id)
        if is_selected:
            c = "#00F0FF"  # Cyan highlight
            s = 18
        elif inc.priority_score >= 80:
            c = "#EF4444"  # Critical Red
            s = 14
        elif inc.priority_score >= 50:
            c = "#F59E0B"  # High Orange
        else:
            c = "#10B981"  # Moderate Green
            s = 10

        inc_colors.append(c)
        inc_sizes.append(s)
        inc_texts.append(inc.incident_id)

        # Detailed hovercard
        alloc_map = plan.allocations.get(inc.incident_id, {})
        unmet_map = plan.unmet.get(inc.incident_id, {})
        tot_alloc = sum(alloc_map.values())
        tot_unmet = sum(unmet_map.values())
        tot_req = inc.total_demand

        hover = (
            f"<b>INCIDENT: {inc.incident_id}</b><br>"
            f"Priority Score: <b>{inc.priority_score:.1f}</b>/100<br>"
            f"Event: {inc.event_name}<br>"
            f"Category: {inc.category}<br>"
            f"Fulfillment: {tot_alloc}/{tot_req} units<br>"
            f"Unmet Shortage: {tot_unmet} units<br>"
            f"Status: {'CRITICAL' if inc.priority_score >= 80 else 'ELEVATED'}"
        )
        inc_hovers.append(hover)

        # Allocation Conduits from Depot (0, 0, 0) to Incident (x, y, z)
        if tot_alloc > 0:
            line_x.extend([0, x, None])
            line_y.extend([0, y, None])
            line_z.extend([0, z, None])

    # Add Incident Nodes Trace
    fig.add_trace(go.Scatter3d(
        x=inc_x, y=inc_y, z=inc_z,
        mode="markers+text",
        marker=dict(
            size=inc_sizes,
            color=inc_colors,
            symbol="circle",
            line=dict(color="#FFFFFF", width=1),
            opacity=0.9,
        ),
        text=inc_texts,
        textposition="top center",
        textfont=dict(color="#F8FAFC", size=10, family="sans-serif"),
        name="Emergency Incidents",
        hoverinfo="text",
        hovertext=inc_hovers,
    ))

    # Add Allocation Conduits (Laser / Arcs)
    if line_x:
        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode="lines",
            line=dict(color="#00F0FF", width=3, dash="solid"),
            name="Resource Allocation Conduits",
            hoverinfo="none",
        ))

    # Tactical Scene Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1120",
        scene=dict(
            bgcolor="#0F172A",
            xaxis=dict(
                title="Sector X (km)",
                backgroundcolor="#0F172A",
                gridcolor="#1E293B",
                showbackground=True,
                zerolinecolor="#334155",
            ),
            yaxis=dict(
                title="Sector Y (km)",
                backgroundcolor="#0F172A",
                gridcolor="#1E293B",
                showbackground=True,
                zerolinecolor="#334155",
            ),
            zaxis=dict(
                title="Priority Elevation (Z)",
                backgroundcolor="#0B1120",
                gridcolor="#1E293B",
                showbackground=True,
                zerolinecolor="#334155",
                range=[0, 105],
            ),
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
                center=dict(x=0, y=0, z=0.2),
            ),
            aspectratio=dict(x=1.2, y=1.2, z=0.8),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=520,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
            x=0.02,
            y=0.98,
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
    pulsing tactical pillars, central logistics depot, and animated particle arcs.
    """
    # Prepare serializable payload
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
                background-color: #070B14;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            }}
            #canvas-container {{
                width: 100vw;
                height: 100vh;
                position: absolute;
            }}
            .hud-overlay {{
                position: absolute;
                top: 12px;
                left: 14px;
                color: #00F0FF;
                font-size: 11px;
                letter-spacing: 0.08em;
                background: rgba(11, 17, 32, 0.85);
                padding: 8px 12px;
                border-radius: 4px;
                border: 1px solid rgba(0, 240, 255, 0.3);
                pointer-events: none;
                z-index: 10;
            }}
            .hud-instructions {{
                position: absolute;
                bottom: 12px;
                right: 14px;
                color: #94A3B8;
                font-size: 10px;
                background: rgba(11, 17, 32, 0.8);
                padding: 6px 10px;
                border-radius: 4px;
                border: 1px solid #1E293B;
                pointer-events: none;
            }}
            #inspect-card {{
                position: absolute;
                top: 12px;
                right: 14px;
                background: rgba(15, 23, 42, 0.95);
                border: 1px solid #00F0FF;
                border-radius: 6px;
                padding: 10px 14px;
                color: #F8FAFC;
                font-size: 11px;
                max-width: 220px;
                display: none;
                z-index: 20;
                box-shadow: 0 0 15px rgba(0, 240, 255, 0.2);
            }}
        </style>
        <!-- Load Three.js and OrbitControls from CDN with fallback -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div class="hud-overlay">
            <div>🛰️ RESCUEGRID 3D TACTICAL TWIN</div>
            <div style="color: #64748B; font-size: 9px; margin-top: 2px;">ENGINE: WEBGL • REAL-TIME ALLOCATION MESH</div>
        </div>
        <div id="inspect-card"></div>
        <div class="hud-instructions">Left Drag: Orbit • Right Drag: Pan • Scroll: Zoom</div>
        <div id="canvas-container"></div>

        <script>
            const DATA = {payload_json};
            const container = document.getElementById('canvas-container');
            const inspectCard = document.getElementById('inspect-card');

            // Scene, Camera, Renderer
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x070B14);
            scene.fog = new THREE.FogExp2(0x070B14, 0.005);

            const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
            camera.position.set(70, 80, 90);

            const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            // Controls
            let controls;
            if (typeof THREE.OrbitControls !== 'undefined') {{
                controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.maxPolarAngle = Math.PI / 2.05; // Prevent camera going under floor
                controls.target.set(0, 10, 0);
            }}

            // Lights
            const ambientLight = new THREE.AmbientLight(0x223344, 1.2);
            scene.add(ambientLight);

            const dirLight = new THREE.DirectionalLight(0x00F0FF, 1.5);
            dirLight.position.set(50, 100, 50);
            scene.add(dirLight);

            const pointLight = new THREE.PointLight(0xFF4444, 2, 80);
            pointLight.position.set(0, 20, 0);
            scene.add(pointLight);

            // Tactical Grid Floor
            const gridHelper = new THREE.GridHelper(160, 32, 0x00F0FF, 0x1E293B);
            gridHelper.position.y = 0;
            scene.add(gridHelper);

            // Central Logistics Depot Hub
            const depotGeo = new THREE.CylinderGeometry(6, 7, 4, 16);
            const depotMat = new THREE.MeshStandardMaterial({{
                color: 0x0284C7,
                emissive: 0x0369A1,
                roughness: 0.3,
                metalness: 0.8
            }});
            const depot = new THREE.Mesh(depotGeo, depotMat);
            depot.position.set(0, 2, 0);
            scene.add(depot);

            // Glowing Depot Ring
            const ringGeo = new THREE.RingGeometry(8, 9, 32);
            const ringMat = new THREE.MeshBasicMaterial({{
                color: 0x00F0FF,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.6
            }});
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = Math.PI / 2;
            ring.position.y = 0.2;
            scene.add(ring);

            // Incident Meshes and Conduit Arcs
            const interactiveObjects = [];

            DATA.incidents.forEach(inc => {{
                const height = Math.max(6, inc.priority * 0.35);
                const pillarGeo = new THREE.BoxGeometry(4, height, 4);

                let colorHex = 0x10B981; // Green
                if (inc.selected) colorHex = 0x00F0FF;
                else if (inc.is_critical) colorHex = 0xEF4444; // Red
                else if (inc.priority >= 50) colorHex = 0xF59E0B; // Orange

                const pillarMat = new THREE.MeshStandardMaterial({{
                    color: colorHex,
                    emissive: colorHex,
                    emissiveIntensity: inc.selected ? 0.6 : 0.25,
                    roughness: 0.2,
                    metalness: 0.7
                }});

                const pillar = new THREE.Mesh(pillarGeo, pillarMat);
                pillar.position.set(inc.x, height / 2, inc.y);
                pillar.userData = inc;
                scene.add(pillar);
                interactiveObjects.push(pillar);

                // Base Status Ring around each incident
                const incRingGeo = new THREE.RingGeometry(3, 4, 16);
                const incRingMat = new THREE.MeshBasicMaterial({{
                    color: colorHex,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.4
                }});
                const incRing = new THREE.Mesh(incRingGeo, incRingMat);
                incRing.rotation.x = Math.PI / 2;
                incRing.position.set(inc.x, 0.1, inc.y);
                scene.add(incRing);

                // Curved Allocation Arc from Depot to Incident
                if (inc.total_allocated > 0) {{
                    const curve = new THREE.QuadraticBezierCurve3(
                        new THREE.Vector3(0, 3, 0),
                        new THREE.Vector3(inc.x * 0.5, height + 15, inc.y * 0.5),
                        new THREE.Vector3(inc.x, height, inc.y)
                    );
                    const points = curve.getPoints(24);
                    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
                    const lineMat = new THREE.LineBasicMaterial({{
                        color: 0x00F0FF,
                        transparent: true,
                        opacity: 0.75,
                        linewidth: 2
                    }});
                    const arc = new THREE.Line(lineGeo, lineMat);
                    scene.add(arc);
                }}
            }});

            // Raycaster for Hover & Selection
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
                        <div style="font-weight: 700; color: #00F0FF;">#${{d.id}}</div>
                        <div style="font-size: 10px; color: #94A3B8; margin-bottom: 4px;">PRIORITY: ${{d.priority.toFixed(1)}}</div>
                        <div>Allocated: <b>${{d.total_allocated}} / ${{d.total_demand}}</b> units</div>
                        <div>Unmet Deficit: <b style="color: ${{d.total_unmet > 0 ? '#EF4444' : '#10B981'}}">${{d.total_unmet}}</b> units</div>
                        <div style="font-size: 9px; color: #38BDF8; margin-top: 4px;">Click to inspect details panel</div>
                    `;
                    document.body.style.cursor = 'pointer';
                }} else {{
                    inspectCard.style.display = 'none';
                    document.body.style.cursor = 'default';
                }}
            }}

            window.addEventListener('mousemove', onMouseMove);

            // Animation Loop
            let clock = new THREE.Clock();
            function animate() {{
                requestAnimationFrame(animate);
                const t = clock.getElapsedTime();

                // Rotate Depot Beacon
                ring.rotation.z = t * 0.4;

                if (controls) controls.update();
                renderer.render(scene, camera);
            }}
            animate();

            // Resize Handler
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        </script>
    </body>
    </html>
    """
    return html_code
