"""
Example: Fast Material Parameter Updates
=========================================

Demonstrates how to quickly update material parameters (like reflectance)
without rebuilding the expensive BVH geometry structure using the new 
Property-based and Name-based APIs.
"""

import optix_sphere as osg
from pathlib import Path
import time

# Setup paths
mesh_path = (Path(__file__).parent.parent.parent / "assets/R_25.4_1mm.obj")
osg._core.set_log_level(osg._core.LogLevel.WARN)

def main():
    print("Material Update Example (New Name-Based API)")

    # 1. Create simulator and setup scene
    print("\n[1] Building scene (slow, one-time operation)...")
    sim = osg.Simulator()

    start_build = time.time()
    scene = osg.Scene.from_obj(str(mesh_path))
    
    # NEW: Direct mapping from names to material objects
    # No need to manage indices manually!
    materials = {
        "wall_material": osg.material.lambertian(0.99),
        "detector_material": osg.material.detector(),
        "sample_material": osg.material.lambertian(0.0)
    }

    sim.build_scene(scene, materials)

    build_time = time.time() - start_build
    print(f"✅ Scene built in {build_time:.3f}s")

    # 2. Setup simulation config
    sim.config.num_rays = 1000000
    sim.config.max_bounces = 500
    sim.config.use_nee = True

    source = osg.IsotropicPointSource()
    source.position = (0, 0, 0)
    source.weight = 1.0

    print("\n[2] Testing different wall reflectances (Property-based updates)...")
    print("-" * 60)

    # NEW: Get the material object directly by name
    wall = sim.get_material("wall_material")
    
    reflectances = [0.90, 0.95, 0.98, 0.99]

    for rho in reflectances:
        # NEW: Just change the property!
        start_update = time.time()
        wall.reflectance = rho
        sim.sync_to_gpu() # Sync to GPU is fast
        update_time = time.time() - start_update

        # Run simulation
        start_sim = time.time()
        result = sim.run(source)
        sim_time = time.time() - start_sim

        print(f"Reflectance = {rho:.2f}:")
        print(f"  Property update: {update_time*1000:.2f}ms (instant)")
        print(f"  Simulation:      {sim_time:.3f}s")
        print(f"  Irradiance:      {result.irradiance:.6f} W/mm²")
        print()

    # 3. Test switching material types (Lambertian -> Mixed)
    print("\n[3] Switching material type (Name-based set_material)...")
    print("-" * 60)

    # NEW: Use set_material to replace the entire material object by name
    start_update = time.time()
    sim.set_material("wall_material", osg.material.mixed(0.7, 0.3, 0.98), sync_to_gpu=True)
    update_time = time.time() - start_update

    result = sim.run(source)
    print(f"Material type: Mixed (70% diffuse, 30% specular)")
    print(f"Update + Sync:  {update_time*1000:.2f}ms")
    print(f"Irradiance:      {result.irradiance:.6f} W/mm²")
    print()

    # 4. Updating multiple materials
    print("\n[4] Updating multiple materials simultaneously...")
    print("-" * 60)

    start_update = time.time()
    # Change properties of different materials
    sim.get_material("wall_material").reflectance = 0.95
    sim.get_material("sample_material").reflectance = 0.5
    # Sync once for all changes
    sim.sync_to_gpu()
    update_time = time.time() - start_update

    result = sim.run(source)
    print(f"Updated wall and sample materials")
    print(f"Batch Sync:      {update_time*1000:.2f}ms")
    print(f"Irradiance:      {result.irradiance:.6f} W/mm²")

    print("\n✅ New API Benefits:")
    print("   - wall.reflect = 0.99 (Natural property access)")
    print("   - sim.set_material('name', mat) (No more index management)")
    print("   - sim.sync_to_gpu() (Clear intent: move data to hardware)")

if __name__ == "__main__":
    main()
