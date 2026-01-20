import numpy as np
import sys
import optix_sphere._core as osg

def test_voxel_numpy():
    print("=== Testing Voxel Simulator with NumPy Arrays ===\n")

    num_photons = int(1e7)

    # Create a 3-layer voxel grid (100x100x3)
    grid = np.zeros((100, 100, 3), dtype=np.uint8)

    grid[:, :, 0] = 1  # Layer 1 -> material ID 1
    grid[:, :, 1] = 2  # Layer 2 -> material ID 2
    grid[:, :, 2] = 3  # Layer 3 -> material ID 3

    print(f"   Grid shape: {grid.shape}")
    print(f"   Grid dtype: {grid.dtype}")
    print(f"   Unique material IDs: {np.unique(grid)}")

    # Create light source
    source = osg.CollimatedBeamSource()
    source.position = osg.float3(50.0, 50.0, -0.1)
    source.direction = osg.float3(0.0, 0.0, 1.0)
    source.weight = 1.0

    # Create simulation configuration
    config = osg.voxel.SimConfig()
    config.grid = grid
    config.voxel_size = (1.0, 1.0, 1.0)
    config.materials = np.array([
        [1.0,  0.0,   1e-6, 0.0],   # Material 0 (ambient - air)
        [1.42, 0.01,  20.0, 0.7],   # Material 1
        [1.00, 0.1,   90.0, 0.7],   # Material 2
        [1.42, 0.3,   80.0, 0.7]    # Material 3
    ], dtype=np.float32)
    config.source = source

    # Configure boundary collection
    # Collect photons from -Z (reflected) and +Z (transmitted) faces
    boundary_config = osg.voxel.BoundaryCollectionConfig()
    boundary_config.enable_z_faces()  # Enable both -Z and +Z faces
    boundary_config.use_grid_center = True  # Auto-center the collection

    # Optional: Set collection radius (e.g., 50mm detector)
    # boundary_config.set_negative_radius(50.0)  # For -Z face
    # boundary_config.set_positive_radius(50.0)  # For +Z face

    config.boundary_collection = boundary_config
    config.enable_specular = True  # Enable specular reflection calculation

    print(f"   Config valid: {config.is_valid()}")
    print(f"   Boundary collection: {boundary_config.to_string()}")

    # Create simulator and run
    simulator = osg.voxel.Simulator(config)
    result = simulator.run(num_photons)

    # Copy results to host
    result.to_host()

    # Calculate reflectance and transmittance
    specular_weight = result.specular_batch.total_weight()
    reflected_weight = result.negative_boundary_batch.total_weight()  # Changed from reflected_batch
    transmitted_weight = result.positive_boundary_batch.total_weight()  # Changed from transmitted_batch

    R_specular = specular_weight / num_photons
    R_diffuse = reflected_weight / num_photons
    R = R_specular + R_diffuse
    T = transmitted_weight / num_photons
    A = 1.0 - R - T

    print(f"\n   === Results ===")
    print(f"   Specular reflectance:  {R_specular:.6f}")
    print(f"   Diffuse reflectance:   {R_diffuse:.6f}")
    print(f"   Total reflectance (R): {R:.6f}")
    print(f"   Transmittance (T):     {T:.6f}")
    print(f"   Absorption (A):        {A:.6f}")
    print(f"   R + T + A = {R + T + A:.6f}")

    print(f"\n   === Photon Counts ===")
    print(f"   Specular count:     {result.specular_batch.size():,}")
    print(f"   Reflected count:    {result.negative_boundary_batch.size():,}")
    print(f"   Transmitted count:  {result.positive_boundary_batch.size():,}")


if __name__ == "__main__":
    test_voxel_numpy()
