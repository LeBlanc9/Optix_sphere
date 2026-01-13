import optix_sphere._core as osg
from pathlib import Path
import time
import os

def main():
    print("--- Python Mesh-Based Simulation ---")

    osg.set_log_level(osg.LogLevel.WARN)


    # --- Configuration (Hardcoded as per user's request) ---
    sim_config = osg.SimConfig()
    sim_config.num_rays = 2_000_000
    sim_config.max_bounces = 500
    sim_config.use_nee = True

    source = osg.IsotropicPointSource()
    source.position = osg.float3(0, 0, 0)
    source.weight = 1.0

    materials = {}
    # materials["wall_material"] = osg.material.mixed(0.7, 0.3, 0.99)
    materials["wall_material"] = osg.material.lambertian(0.99)
    materials["detector_material"] = osg.material.detector()

    # --- Asset Path ---
    mesh_path = (Path(__file__).parent.parent / "assets/validations/port_thickness/integrating_sphere_25.4_0.01.obj")
    scene = osg.Scene.from_obj(str(mesh_path))
        
    # --- Simulation ---
    simulator = osg.Simulator()

    start_build = time.time()
    simulator.build_scene(scene, materials)
    end_build = time.time()
    print(f"   ✅ Scene built in {end_build - start_build:.3f} seconds.")

    start_run = time.time()
    result = simulator.run(source, sim_config)
    end_run = time.time() 
    print(f"   ✅ Simulation finished in {end_run - start_run:.3f} seconds.")

    # --- Results ---
    print("\n--- Simulation Results ---")
    print(f"  Detected Flux:   {result.detected_flux/ result.total_rays:.6f} W")
    print(f"  Irradiance:      {result.irradiance/ result.total_rays:.6f} W/mm²")
    print(f"  Detected Rays:   {result.detected_rays} / {result.total_rays:,}")
    print(f"  Average Bounces: {result.avg_bounces:.2f}")


if __name__ == "__main__":
    main()