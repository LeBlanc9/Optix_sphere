import numpy as np
import sys
import optix_sphere._core as osg

def test_voxel_numpy():
    print("=== Testing Voxel Simulator with NumPy Arrays ===\n")

    num_photons = int(1e7)

    grid = np.zeros((100, 100, 3), dtype=np.uint8)

    grid[:, :, 0] = 1  # Layer 1 -> material ID 1
    grid[:, :, 1] = 2  # Layer 2 -> material ID 2
    grid[:, :, 2] = 3  # Layer 3 -> material ID 3

    print(f"   Grid shape: {grid.shape}")
    print(f"   Grid dtype: {grid.dtype}")
    print(f"   Unique material IDs: {np.unique(grid)}")

    # Step 2: Define materials as numpy array
    print("\n2. Defining materials...")
    materials = np.array([
        [1.0,  0.0,   1e-6, 0.0],   # Material 0 (ambient)
        [1.42, 0.01,  20.0, 0.7],   # Material 1
        [1.00, 0.1,   90.0, 0.7],   # Material 2
        [1.42, 0.3,   80.0, 0.7]    # Material 3
    ], dtype=np.float32)
    print(f"   Materials shape: {materials.shape}")
    print(f"   Format: [n, mua, mus, g]")

    # Step 3: Create collimated beam source
    print("\n3. Creating collimated beam source...")
    source = osg.CollimatedBeamSource()
    source.position = osg.float3(50.0, 50.0, -0.1)
    source.direction = osg.float3(0.0, 0.0, 1.0)
    source.weight = 1.0

    # Step 4: Create SimConfig with numpy array
    config = osg.voxel.SimConfig()
    config.set_grid(grid)
    config.set_materials(materials)
    config.set_source(source)
    config.set_exit_boundaries(0.0, 3.0)

    print(f"   Config valid: {config.is_valid()}")

    simulator = osg.voxel.Simulator(config)
    result = simulator.run(num_photons)

    specular_weight = result.specular_batch.total_weight()
    reflected_weight = result.reflected_batch.total_weight()
    transmitted_weight = result.transmitted_batch.total_weight()

    R_specular = specular_weight / num_photons
    R_diffuse = reflected_weight / num_photons
    R = R_specular + R_diffuse
    T = transmitted_weight / num_photons
    A = 1.0 - R - T

    print(f"   Specular reflectance:  {R_specular:.6f}")
    print(f"   Diffuse reflectance:   {R_diffuse:.6f}")
    print(f"   Total reflectance (R): {R:.6f}")
    print(f"   Transmittance (T):     {T:.6f}")
    print(f"   Absorption (A):        {A:.6f}")
    print(f"   R + T + A = {R + T + A:.6f}")

    print(f"\n   Specular count:     {result.specular_batch.size():,}")
    print(f"   Reflected count:    {result.reflected_batch.size():,}")
    print(f"   Transmitted count:  {result.transmitted_batch.size():,}")


if __name__ == "__main__":
    test_voxel_numpy()
