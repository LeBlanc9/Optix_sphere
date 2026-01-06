"""
Visualize integrating sphere geometry and photon batch positions.
Simple script to check if photons are positioned correctly relative to the sphere.
"""

import numpy as np
import plotly.graph_objects as go
import trimesh
import optix_sphere._core as osg
from pathlib import Path


def visualize_sphere_and_photons(mesh_path, photon_batch, num_display=1000):
    """
    Visualize sphere mesh and photon positions/directions.

    Args:
        mesh_path: Path to sphere .obj file
        photon_batch: PhotonBatch on GPU
        num_display: Max number of photons to display (for performance)
    """
    # Load mesh
    print(f"Loading mesh: {Path(mesh_path).name}")
    mesh = trimesh.load(mesh_path, force='mesh')

    # Get photon data from GPU
    print(f"Transferring {photon_batch.size()} photons from GPU...")
    host_batch = photon_batch.to_host()

    # host_batch.positions/directions are already numpy arrays (N, 3)
    positions = host_batch.positions  # shape: (N, 3)
    directions = host_batch.directions  # shape: (N, 3)
    weights = host_batch.weights  # shape: (N,)

    # Subsample if too many
    if len(positions) > num_display:
        indices = np.random.choice(len(positions), num_display, replace=False)
        positions = positions[indices]
        directions = directions[indices]
        weights = weights[indices]

    print(f"\nPhoton stats:")
    print(f"  Count: {photon_batch.size():,} (showing {len(positions):,})")
    print(f"  Position range: x[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
          f"y[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}], "
          f"z[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
    print(f"  Total weight: {photon_batch.total_weight():.6f}")

    # Create figure
    fig = go.Figure()

    # Add sphere mesh (semi-transparent)
    fig.add_trace(go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color='lightblue',
        opacity=0.3,
        name='Sphere',
    ))

    # Add photon positions (colored by weight)
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers',
        marker=dict(size=3, color=weights, colorscale='Viridis', showscale=True),
        name='Photons',
    ))

    # Add some direction arrows (subsample for clarity)
    num_arrows = min(50, len(positions))
    arrow_idx = np.linspace(0, len(positions)-1, num_arrows, dtype=int)
    for idx in arrow_idx:
        p = positions[idx]
        d = directions[idx] * 0.5  # Arrow length
        fig.add_trace(go.Scatter3d(
            x=[p[0], p[0] + d[0]],
            y=[p[1], p[1] + d[1]],
            z=[p[2], p[2] + d[2]],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Layout
    fig.update_layout(
        title=f'Sphere + Photons ({len(positions):,} shown, weight={photon_batch.total_weight():.3f})',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
        ),
        width=1200,
        height=900,
    )

    return fig


if __name__ == "__main__":
    print("="*70)
    print("Sphere and Photon Visualization")
    print("="*70)

    # 1. Generate photons
    print("\n1. Generating photons...")
    source = osg.CollimatedBeamSource()
    source.position = osg.float3(0.0, 0.0, -0.1)
    source.direction = osg.float3(0.0, 0.0, 1.0)
    source.weight = 1.0

    photon_batch = osg.generate_photons(source, 10000)
    print(f"   Generated {photon_batch.size()} photons")

    # 2. Transform to sphere coordinates
    print("\n2. Transforming to sphere port...")

    photon_batch = osg.translate_photons(photon_batch, osg.float3(0, 0, 80))

    # 3. Load sphere mesh
    mesh_path = (
        Path(__file__).parent.parent /
        "assets/validations/port_thickness/integrating_sphere_25.4_10.obj"
    )

    # 4. Visualize
    print("\n3. Creating visualization...")
    fig = visualize_sphere_and_photons(str(mesh_path), photon_batch, num_display=1000)

    print("\n✅ Opening in browser (drag to rotate, scroll to zoom)...")
    fig.show()
