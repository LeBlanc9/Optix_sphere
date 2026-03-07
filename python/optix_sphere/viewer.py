"""
Visualization utilities for mesh and photon batch analysis.

Example:
    >>> import optix_sphere as osg
    >>> from optix_sphere.viewer import Viewer
    >>>
    >>> mesh = osg.Mesh.from_obj("sphere.obj")
    >>> photon_batch = ...
    >>>
    >>> viewer = Viewer()
    >>> viewer.add(mesh)
    >>> viewer.add(photon_batch)
    >>> viewer.show()  # or viewer.save("result.html")
"""

import numpy as np

import plotly.graph_objects as go
import optix_sphere as osg

class Viewer:
    """
    Interactive 3D viewer for meshes and photon batches.

    Attributes:
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        num_photons: Maximum number of photons to display
        num_arrows: Number of direction arrows to show
        arrow_length: Length of direction arrows (mm)
        mesh_opacity: Opacity of mesh surfaces (0-1)
        photon_size: Size of photon markers
        material_colors: Dictionary mapping material names to colors

    Example:
        >>> import optix_sphere as osg
        >>> from optix_sphere.viewer import Viewer
        >>>
        >>> # Load mesh
        >>> mesh = osg.Mesh.from_obj("sphere.obj")
        >>>
        >>> # Create viewer with custom material colors
        >>> viewer = Viewer()
        >>> viewer.material_colors = {
        ...     'wall': 'lightgray',
        ...     'detector': 'red',
        ...     'port': 'blue'
        ... }
        >>>
        >>> # Add mesh and show
        >>> viewer.add(mesh)
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

        # Material coloring
        # Users can override colors by setting: viewer.material_colors = { 'wall': 'lightgray', 'detector': 'red' }
        self.material_colors = {}
        self._default_palette = [
            'lightblue', 'lightgreen', 'salmon', 'gold', 'violet', 'orange',
            'teal', 'pink', 'gray', 'khaki', 'tomato', 'plum', 'turquoise', 'wheat'
        ]

        # Internal state
        self._items = []

    def add(self, item):
        """
        Add a mesh or photon batch to the viewer.

        Args:
            item: osg.Mesh or osg.PhotonBatch object

        Returns:
            self (for method chaining)

        Example:
            >>> import optix_sphere as osg
            >>> viewer = osg.viewer.Viewer()
            >>> mesh = osg.Mesh.from_obj("sphere.obj")
            >>> viewer.add(mesh).show()
        """
        self._items.append(item)
        return self

    def clear(self):
        """Clear all items from the viewer."""
        self._items.clear()
        return self

    def _extract_mesh_data(self, mesh):
        """Extract vertices and faces from mesh."""
        vertices = np.array([[v.x, v.y, v.z] for v in mesh.vertices])
        faces = np.array([[tri.x, tri.y, tri.z] for tri in mesh.indices])
        return vertices, faces

    def _get_material_face_colors(self, mesh):
        """
        Get color for each face based on material.

        Returns:
            List of color strings, one per triangle
        """
        if not mesh.triangle_materials or not mesh.material_names:
            return None  # No material info, use default single color

        # Build material name -> color mapping
        color_map = {}
        for i, mat_name in enumerate(mesh.material_names):
            if mat_name in self.material_colors:
                color_map[i] = self.material_colors[mat_name]
            else:
                # Use default palette (cycle if needed)
                palette_idx = i % len(self._default_palette)
                color_map[i] = self._default_palette[palette_idx]

        # Map each triangle to its color
        face_colors = [color_map[mat_idx] for mat_idx in mesh.triangle_materials]
        return face_colors

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

    def build(self):
        """
        Build the plotly figure from added items.

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Process all items
        for item in self._items:
            item_type = type(item).__name__

            if item_type == 'Mesh':
                # Add mesh
                vertices, faces = self._extract_mesh_data(item)
                face_colors = self._get_material_face_colors(item)

                if face_colors:
                    # Use per-face colors based on materials
                    fig.add_trace(go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        facecolor=face_colors,
                        opacity=self.mesh_opacity,
                        name='Mesh',
                        showlegend=False,
                    ))

                    # Add legend entries for each material
                    for i, mat_name in enumerate(item.material_names):
                        tri_count = item.get_triangle_count_by_material(mat_name)

                        # Get color for this material
                        if mat_name in self.material_colors:
                            color = self.material_colors[mat_name]
                        else:
                            palette_idx = i % len(self._default_palette)
                            color = self._default_palette[palette_idx]

                        # Add invisible scatter point for legend
                        fig.add_trace(go.Scatter3d(
                            x=[None], y=[None], z=[None],
                            mode='markers',
                            marker=dict(size=10, color=color),
                            name=f'{mat_name} ({tri_count} tris)',
                            showlegend=True,
                        ))
                else:
                    # Fallback to single color
                    fig.add_trace(go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=faces[:, 0],
                        j=faces[:, 1],
                        k=faces[:, 2],
                        color='lightblue',
                        opacity=self.mesh_opacity,
                        name='Mesh',
                    ))

            elif item_type == 'PhotonBatch':
                # Extract and add photons
                positions, directions, weights = self._extract_photons(item)

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

                    fig.add_trace(go.Scatter3d(
                        x=[p[0], p[0] + d[0]],
                        y=[p[1], p[1] + d[1]],
                        z=[p[2], p[2] + d[2]],
                        mode='lines',
                        line=dict(color='darkblue', width=3),
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


__all__ = [
    'Viewer',
]
