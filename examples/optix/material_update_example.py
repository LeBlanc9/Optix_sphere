"""
Example: Fast Material Parameter Updates
=========================================

Demonstrates how to quickly update material parameters (like reflectance)
without rebuilding the expensive BVH geometry structure using the material pool API.
"""

import optix_sphere as osg
from pathlib import Path
import time

# Setup paths
mesh_path = (Path(__file__).parent.parent.parent / "assets/R_25.4_1mm.obj")
osg._core.set_log_level(osg._core.LogLevel.WARN)

def main():
    print("Material Update Example (Material Pool API)")

    # 1. Create simulator and setup material pool
    print("\n[1] Building scene (slow, one-time operation)...")
    sim = osg.Simulator()

    start_build = time.time()
    scene = osg.Scene.from_obj(str(mesh_path))
    print(f"Scene material names: {scene.get_material_names()}")

    # Setup material pool with initial values
    # Pool index 0: wall material (starts at 0.99 reflectance)
    # Pool index 1: detector material
    # Pool index 2: sample material
    idx_wall = sim.add_material(osg.material.lambertian(0.99))
    idx_detector = sim.add_material(osg.material.detector())
    idx_sample = sim.add_material(osg.material.lambertian(0.0))

    # Map mesh material names to pool indices
    material_mapping = {
        "wall_material": idx_wall,
        "detector_material": idx_detector,
        "sample_material": idx_sample
    }

    sim.build_scene(scene, material_mapping)

    build_time = time.time() - start_build
    print(f"✅ Scene built in {build_time:.3f}s")

    # 2. Setup simulation config
    sim.config.num_rays = 1000000
    sim.config.max_bounces = 500
    sim.config.use_nee = True

    # Setup photon source (isotropic point source at center)
    source = osg.IsotropicPointSource()
    source.position = (0, 0, 0)
    source.weight = 1.0

    print("\n[2] Testing different wall reflectances...")
    print("-" * 60)

    # Test multiple reflectance values (fast material updates!)
    reflectances = [0.90, 0.95, 0.98, 0.99]
    results = []

    for rho in reflectances:
        # Update wall material reflectance (FAST - no geometry rebuild!)
        # Just modify the material pool and call update_materials()
        start_update = time.time()
        sim.set_material(idx_wall, osg.material.lambertian(rho))
        sim.update_materials()
        update_time = time.time() - start_update

        # Run simulation with updated material
        start_sim = time.time()
        result = sim.run(source)
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
    sim.set_material(idx_wall, osg.material.mixed(0.7, 0.3, 0.98))
    sim.update_materials()
    update_time = time.time() - start_update

    result = sim.run(source)

    print(f"Material type: Mixed (70% diffuse, 30% specular)")
    print(f"Material update: {update_time*1000:.2f}ms")
    print(f"Irradiance:      {result.irradiance:.6f} W/mm²")
    print(f"Detected flux:   {result.detected_flux:.6f} W")
    print()

    # 4. Compare with Lambertian at same reflectance
    sim.set_material(idx_wall, osg.material.lambertian(0.98))
    sim.update_materials()
    result_lamb = sim.run(source)

    print(f"Material type: Lambertian (purely diffuse)")
    print(f"Irradiance:      {result_lamb.irradiance:.6f} W/mm²")
    print(f"Detected flux:   {result_lamb.detected_flux:.6f} W")
    print()

    if result_lamb.irradiance > 0:
        print(f"Difference: {abs(result.irradiance - result_lamb.irradiance)/result_lamb.irradiance*100:.2f}%")
        print("(Mixed material has different scattering behavior)")
    else:
        print("Note: Results are zero - check light source configuration")

    # 5. Demonstrate updating multiple materials at once
    print("\n[4] Updating multiple materials simultaneously...")
    print("-" * 60)

    start_update = time.time()
    # Change both wall and sample materials
    sim.set_material(idx_wall, osg.material.lambertian(0.95))
    sim.set_material(idx_sample, osg.material.lambertian(0.5))
    sim.update_materials()
    update_time = time.time() - start_update

    result = sim.run(source)
    print(f"Updated wall and sample materials")
    print(f"Material update: {update_time*1000:.2f}ms")
    print(f"Irradiance:      {result.irradiance:.6f} W/mm²")
    print()

    # 6. Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Initial scene build: {build_time:.3f}s (slow, one-time)")
    print(f"Material updates:    ~{update_time*1000:.2f}ms (fast, reusable)")
    print(f"Speedup:             ~{build_time/update_time:.0f}x faster!")
    print()
    print("✅ Material pool API benefits:")
    print("   - Use add_material() to add materials and get indices")
    print("   - Use set_material(index, material) to update specific materials")
    print("   - Call update_materials() to sync changes to GPU")
    print("   - Update multiple materials before calling update_materials()")
    print("   - Share material instances across multiple mesh materials")
    print("   - Much faster than rebuilding scene!")

if __name__ == "__main__":
    main()
