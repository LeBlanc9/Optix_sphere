"""
OptiX Sphere - Monte Carlo simulation for integrating spheres

This package provides GPU-accelerated Monte Carlo simulation tools for
integrating sphere optical measurements, including support for layered media
and photon batch processing.

Main Components:
    - Scene: 3D scene representation with transformation support
    - Simulator: OptiX-based Monte Carlo ray tracer
    - PhotonBatch: GPU-resident photon batch management
    - LayeredMediaSimulator: Multi-layer media simulation
    - VoxelSimulator: Voxel-based media simulation
    - visualization: Tools for visualizing scenes and photon batches

Example:
    >>> import optix_sphere as osg
    >>>
    >>> # Load scene
    >>> scene = osg.Scene.from_obj("sphere.obj")
    >>>
    >>> # Create simulator
    >>> sim = osg.Simulator()
    >>> materials = {
    ...     "wall": osg.material.lambertian(0.98),
    ...     "detector": osg.material.detector()
    ... }
    >>> sim.build_scene(scene, materials)
    >>>
    >>> # Run simulation
    >>> source = osg.CollimatedBeamSource()
    >>> config = osg.SimConfig()
    >>> result = sim.run(source, config)
    >>> print(f"Detected flux: {result.detected_flux}")
    >>>
    >>> # Visualize (requires plotly)
    >>> from optix_sphere.visualization import visualize_scene_and_photons
    >>> fig = visualize_scene_and_photons(scene, photon_batch)
    >>> fig.show()
"""

# Import the C++ extension module
from . import _core

# Import Python submodules
from . import viewer

# ============================================================================
# Core Classes
# ============================================================================

# Scene and Mesh
Scene = _core.Scene
Mesh = _core.Mesh

# Simulation
Simulator = _core.Simulator
SimConfig = _core.SimConfig
SimulationResult = _core.SimulationResult

# Photon System
PhotonBatch = _core.PhotonBatch
HostPhotonBatch = _core.HostPhotonBatch

# ============================================================================
# Photon Sources
# ============================================================================

CollimatedBeamSource = _core.CollimatedBeamSource
SpotSource = _core.SpotSource
IsotropicPointSource = _core.IsotropicPointSource
GaussianBeamSource = _core.GaussianBeamSource
FocusedSpotSource = _core.FocusedSpotSource

# Photon generation
generate_photons = _core.generate_photons
translate_photons = _core.translate_photons
translate_photons_inplace = _core.translate_photons_inplace

# ============================================================================
# Media and Voxel
# ============================================================================

# Media namespace (layered media simulation)
media = _core.media

# Voxel namespace (voxel-based simulation)
voxel = _core.voxel

# ============================================================================
# Types
# ============================================================================

float3 = _core.float3
int3 = _core.int3
uint3 = _core.uint3

# ============================================================================
# Material System
# ============================================================================

# Material namespace (contains factory functions)
material = _core.material

# ============================================================================
# Theory/Validation
# ============================================================================

TheoryResult = _core.TheoryResult

# ============================================================================
# Logging
# ============================================================================

set_log_level = _core.set_log_level
LogLevel = _core.LogLevel

# ============================================================================
# Metadata
# ============================================================================

__version__ = _core.__version__

# ============================================================================
# Auto-import all public symbols from _core
# ============================================================================

def _import_all_from_core():
    """
    Automatically import all public symbols from _core module.

    This function inspects the _core module and imports all symbols that:
    - Don't start with underscore (public API)
    - Aren't already explicitly imported above

    This ensures that new additions to the C++ bindings are automatically
    available without manually updating this file.
    """
    import sys
    current_module = sys.modules[__name__]

    # Get all public symbols from _core
    for name in dir(_core):
        # Skip private symbols
        if name.startswith('_'):
            continue

        # Skip if already imported
        if hasattr(current_module, name):
            continue

        # Import the symbol
        setattr(current_module, name, getattr(_core, name))

# Execute auto-import
_import_all_from_core()

# Build __all__ dynamically
__all__ = [
    # Explicitly listed core classes
    "Scene",
    "Mesh",
    "Simulator",
    "SimConfig",
    "SimulationResult",
    "PhotonBatch",
    "HostPhotonBatch",
    "generate_photons",
    "translate_photons",
    "translate_photons_inplace",
    "CollimatedBeamSource",
    "SpotSource",
    "IsotropicPointSource",
    "GaussianBeamSource",
    "FocusedSpotSource",
    "media",
    "voxel",
    "float3",
    "int3",
    "uint3",
    "material",
    "TheoryResult",
    "set_log_level",
    "LogLevel",

    # Python submodules
    "viewer",
]

# Add all auto-imported symbols to __all__
__all__.extend([
    name for name in dir(_core)
    if not name.startswith('_') and name not in __all__
])

