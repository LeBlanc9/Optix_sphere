"""
Example: Fast Material Parameter Updates
=========================================

Demonstrates how to quickly update material parameters (like reflectance)
without rebuilding the expensive BVH geometry structure.
"""

import optix_sphere._core as osg
from pathlib import Path
import time

# Setup paths
mesh_path = (Path(__file__).parent.parent / "assets/R_25.4_1mm.obj")
osg.set_log_level(osg.LogLevel.WARN)

def main():
    print("Material Update Example")

    # 1. Create simulator and build scene
    print("\n[1] Building scene (slow, one-time operation)...")
    sim = osg.Simulator()

    start_build = time.time()
    scene = osg.Scene.from_obj(str(mesh_path))
    print(scene.get_material_names())
    materials = {
        "wall_material": osg.material.lambertian(0.99),
        "detector_material": osg.material.detector(),
        "sample_material": osg.material.lambertian(80.0)
    }
    sim.build_scene(scene, materials)

    build_time = time.time() - start_build
    print(f"✅ Scene built in {build_time:.3f}s")

    # 2. Setup simulation config
    sim_config = osg.SimConfig()
    sim_config.num_rays = 100000
    sim_config.max_bounces = 100
    sim_config.use_nee = True

    # Setup photon source (collimated beam through port)
    source = osg.CollimatedBeamSource()
    source.position = osg.float3(0, 0, -15)
    source.direction = osg.float3(0, 0, 1)
    source.weight = 1.0

    print("\n[2] Testing different wall reflectances...")
    print("-" * 60)

    # Test multiple reflectance values (fast material updates!)
    reflectances = [0.90, 0.95, 0.98, 0.99]
    results = []

    for rho in reflectances:
        # Update wall material reflectance (FAST - no geometry rebuild!)
        start_update = time.time()
        sim.update_material("wall_material", osg.material.lambertian(rho))
        update_time = time.time() - start_update

        # Run simulation with updated material
        start_sim = time.time()
        result = sim.run(source, sim_config)
        sim_time = time.time() - start_sim

        results.append(result)

        print(f"Reflectance = {rho:.2f}:")
        print(f"  Material update: {update_time*1000:.2f}ms (fast!)")
        print(f"  Simulation:      {sim_time:.3f}s")
        print(f"  Irradiance:      {result.irradiance:.6f} W/mm²")
        print(f"  Detected flux:   {result.detected_flux:.6f} W")
        print(f"  Avg bounces:     {result.avg_bounces:.2f}")
        print()

    # 3. Test switching material types (Lambertian -> Mixed)
    print("\n[3] Switching from Lambertian to Mixed material...")
    print("-" * 60)

    # Update to mixed material (70% diffuse, 30% specular)
    start_update = time.time()
    sim.update_material("wall_material", osg.material.mixed(0.7, 0.3, 0.98))
    update_time = time.time() - start_update

    result = sim.run(source, sim_config)

    print(f"Material type: Mixed (70% diffuse, 30% specular)")
    print(f"Material update: {update_time*1000:.2f}ms")
    print(f"Irradiance:      {result.irradiance:.6f} W/mm²")
    print(f"Detected flux:   {result.detected_flux:.6f} W")
    print()

    # 4. Compare with Lambertian at same reflectance
    sim.update_material("wall_material", osg.material.lambertian(0.98))
    result_lamb = sim.run(source, sim_config)

    print(f"Material type: Lambertian (purely diffuse)")
    print(f"Irradiance:      {result_lamb.irradiance:.6f} W/mm²")
    print(f"Detected flux:   {result_lamb.detected_flux:.6f} W")
    print()

    print(f"Difference: {abs(result.irradiance - result_lamb.irradiance)/result_lamb.irradiance*100:.2f}%")
    print("(Mixed material has different scattering behavior)")

    # 5. Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Initial scene build: {build_time:.3f}s (slow, one-time)")
    print(f"Material updates:    ~{update_time*1000:.2f}ms (fast, reusable)")
    print(f"Speedup:             ~{build_time/update_time:.0f}x faster!")
    print()
    print("✅ Material updates are MUCH faster than rebuilding scene!")
    print("   Use this for parameter sweeps, optimization, etc.")

if __name__ == "__main__":
    main()
