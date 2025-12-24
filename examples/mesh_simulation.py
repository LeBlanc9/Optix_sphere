import optix_sphere._core as osg
import time
import os

def main():
    print("--- Python Mesh-Based Simulation ---")

    osg.set_log_level("warn")


    # --- Configuration (Hardcoded as per user's request) ---
    sim_config = osg.SimConfig()
    sim_config.num_rays = 1_000_000
    sim_config.max_bounces = 500
    sim_config.use_nee = True

    source = osg.IsotropicPointSource()
    source.position = osg.float3(0, 0, 0)
    source.weight = 1.0

    mesh_config = osg.MeshSceneConfig()
    mesh_config.default_reflectance = 0.98

    # --- Asset Path ---
    mesh_path = os.path.join("./assets", "integrating_sphere_0.3.obj")
        
    # --- Simulation ---
    simulator = osg.Simulator()

    start_build = time.time()
    simulator.build_scene_from_file(mesh_path, mesh_config)
    end_build = time.time()
    print(f"   ✅ Scene built in {end_build - start_build:.3f} seconds.")

    start_run = time.time()
    result = simulator.run(source, sim_config)
    end_run = time.time() 
    print(f"   ✅ Simulation finished in {end_run - start_run:.3f} seconds.")

    # --- Results ---
    print("\n--- Simulation Results ---")
    print(f"  Detected Flux:   {result.detected_flux:.6f} W")
    print(f"  Irradiance:      {result.irradiance:.6f} W/mm²")
    print(f"  Detected Rays:   {result.detected_rays} / {result.total_rays:,}")
    print(f"  Average Bounces: {result.avg_bounces:.2f}")


if __name__ == "__main__":
    main()