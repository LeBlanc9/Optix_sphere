"""
Visualization utilities for integrating sphere and photon batch analysis.

Example:
    >>> import optix_sphere as osg
    >>> from optix_sphere.visualization import Viewer
    >>>
    >>> scene = osg.Scene.from_obj("sphere.obj")
    >>> photon_batch = ...
    >>>
    >>> viewer = Viewer()
    >>> viewer.add(scene)
    >>> viewer.add(photon_batch)
    >>> viewer.show()  # or viewer.save("result.html")
"""

import numpy as np

import plotly.graph_objects as go

class Viewer:
    """
    Interactive 3D viewer for scenes and photon batches.

    Attributes:
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        num_photons: Maximum number of photons to display
        num_arrows: Number of direction arrows to show
        arrow_length: Length of direction arrows (mm)
        mesh_opacity: Opacity of mesh surfaces (0-1)
        photon_size: Size of photon markers
        show_sphere_bounds: Whether to show bounding sphere info

    Example:
        >>> viewer = Viewer(title="My Simulation")
        >>> viewer.num_photons = 2000
        >>> viewer.add(scene)
        >>> viewer.add(photon_batch)
        >>> viewer.show()
    """

    def __init__(self, title="OptiX Sphere Visualization"):
        # Display settings
        self.title = title
        self.width = 1400
        self.height = 1000
        self.num_photons = 1000
        self.num_arrows = 30
        self.arrow_length = 5.0
        self.mesh_opacity = 0.3
        self.photon_size = 4
        self.show_sphere_bounds = True

        # Internal state
        self._items = []
        self._sphere_center = None
        self._sphere_radius = None

    def add(self, item):
        """
        Add a scene or photon batch to the viewer.

        Args:
            item: Scene or PhotonBatch object

        Returns:
            self (for method chaining)
        """
        self._items.append(item)
        return self

    def clear(self):
        """Clear all items from the viewer."""
        self._items.clear()
        self._sphere_center = None
        self._sphere_radius = None
        return self

    def _extract_mesh(self, scene):
        """Extract vertices and faces from scene."""
        mesh = scene.get_mesh()
        vertices = np.array([[v.x, v.y, v.z] for v in mesh.vertices])
        faces = np.array([[tri.x, tri.y, tri.z] for tri in mesh.indices])
        return vertices, faces

    def _extract_photons(self, photon_batch):
        """Extract photon data from batch."""
        host_batch = photon_batch.to_host()
        positions = np.array([[p.x, p.y, p.z] for p in host_batch.positions])
        directions = np.array([[d.x, d.y, d.z] for d in host_batch.directions])
        weights = np.array(host_batch.weights)

        # Subsample if needed
        if len(positions) > self.num_photons:
            indices = np.random.choice(len(positions), self.num_photons, replace=False)
            positions = positions[indices]
            directions = directions[indices]
            weights = weights[indices]

        return positions, directions, weights

    def _compute_sphere_bounds(self, scene):
        """Compute sphere center and radius from scene bounds."""
        if self._sphere_center is None:
            min_b, max_b = scene.get_bounds()
            self._sphere_center = np.array([
                (min_b.x + max_b.x) / 2,
                (min_b.y + max_b.y) / 2,
                (min_b.z + max_b.z) / 2
            ])
            self._sphere_radius = max(
                max_b.x - min_b.x,
                max_b.y - min_b.y,
                max_b.z - min_b.z
            ) / 2

    def _check_containment(self, positions):
        """Check which photons are inside the sphere."""
        if self._sphere_center is None:
            return None, None

        distances = np.linalg.norm(positions - self._sphere_center, axis=1)
        inside_mask = distances < self._sphere_radius

        if self.show_sphere_bounds:
            num_inside = np.sum(inside_mask)
            num_total = len(positions)
            print(f"\n📊 Containment Check:")
            print(f"   Inside: {num_inside}/{num_total} ({100*num_inside/num_total:.1f}%)")
            print(f"   Distance range: [{distances.min():.2f}, {distances.max():.2f}] mm")

        return inside_mask, distances

    def build(self):
        """
        Build the plotly figure from added items.

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        photon_data = []  # Store photon data for later processing

        # Process all items
        for item in self._items:
            item_type = type(item).__name__

            if item_type == 'Scene':
                # Add mesh
                vertices, faces = self._extract_mesh(item)
                self._compute_sphere_bounds(item)

                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color='lightblue',
                    opacity=self.mesh_opacity,
                    name='Scene Mesh',
                ))

            elif item_type == 'PhotonBatch':
                # Store for later (need to check containment after sphere bounds known)
                positions, directions, weights = self._extract_photons(item)
                photon_data.append((positions, directions, weights, item))

        # Add photons (now that we know sphere bounds)
        for positions, directions, weights, batch in photon_data:
            inside_mask, _ = self._check_containment(positions)

            if inside_mask is not None:
                # Add photons with containment info
                if np.any(inside_mask):
                    fig.add_trace(go.Scatter3d(
                        x=positions[inside_mask, 0],
                        y=positions[inside_mask, 1],
                        z=positions[inside_mask, 2],
                        mode='markers',
                        marker=dict(size=self.photon_size, color='green', opacity=0.7),
                        name=f'Inside ({np.sum(inside_mask)})',
                    ))

                if np.any(~inside_mask):
                    fig.add_trace(go.Scatter3d(
                        x=positions[~inside_mask, 0],
                        y=positions[~inside_mask, 1],
                        z=positions[~inside_mask, 2],
                        mode='markers',
                        marker=dict(size=self.photon_size, color='red', opacity=0.7),
                        name=f'Outside ({np.sum(~inside_mask)})',
                    ))
            else:
                # No sphere bounds, just show all photons
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='markers',
                    marker=dict(size=self.photon_size, color='blue', opacity=0.7),
                    name=f'Photons ({len(positions)})',
                ))

            # Add direction arrows
            num_arrows = min(self.num_arrows, len(positions))
            arrow_idx = np.linspace(0, len(positions)-1, num_arrows, dtype=int)
            for idx in arrow_idx:
                p = positions[idx]
                d = directions[idx] * self.arrow_length

                if inside_mask is not None:
                    color = 'darkgreen' if inside_mask[idx] else 'darkred'
                else:
                    color = 'darkblue'

                fig.add_trace(go.Scatter3d(
                    x=[p[0], p[0] + d[0]],
                    y=[p[1], p[1] + d[1]],
                    z=[p[2], p[2] + d[2]],
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))

        # Update layout
        fig.update_layout(
            title=self.title,
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
            ),
            width=self.width,
            height=self.height,
        )

        return fig

    def show(self):
        """Build and display the visualization in browser."""
        fig = self.build()
        fig.show()
        return self

    def save(self, filename):
        """
        Build and save the visualization to HTML file.

        Args:
            filename: Output HTML filename
        """
        fig = self.build()
        fig.write_html(filename)
        print(f"✅ Saved visualization to: {filename}")
        return self


# Utility functions (kept for backwards compatibility)

def check_photons_inside_sphere(photon_positions, sphere_center, sphere_radius):
    """
    Check which photons are inside a sphere.

    Args:
        photon_positions: (N, 3) array of photon positions
        sphere_center: (3,) array of sphere center
        sphere_radius: float, sphere radius in mm

    Returns:
        inside_mask: (N,) boolean array, True if inside
        distances: (N,) array of distances from center
    """
    distances = np.linalg.norm(photon_positions - sphere_center, axis=1)
    inside_mask = distances < sphere_radius
    return inside_mask, distances


def compute_centering_offset(photon_positions, sphere_center):
    """
    Compute offset needed to center photon batch at sphere center.

    Args:
        photon_positions: (N, 3) array of photon positions
        sphere_center: (3,) array of desired center

    Returns:
        offset: (3,) array of offset to apply to photons
    """
    photon_center = np.mean(photon_positions, axis=0)
    offset = sphere_center - photon_center
    return offset


__all__ = [
    'Viewer',
    'check_photons_inside_sphere',
    'compute_centering_offset',
]
