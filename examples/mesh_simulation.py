import optix_sphere._core as osg
import time
import os
# Removed argparse as per user's request

def main():
    """
    """
    print("--- Python Mesh-Based Simulation ---")

    # --- Configuration (Hardcoded as per user's request) ---
    sim_config = osg.SimConfig()
    sim_config.num_rays = 1_000_000
    sim_config.max_bounces = 500
    sim_config.use_nee = True # Enable Next Event Estimation (NEE)

    source = osg.IsotropicPointSource()
    source.position = osg.float3(0, 0, 0)
    source.weight = 1.0

    mesh_config = osg.MeshSceneConfig()
    mesh_config.default_reflectance = 0.98

    # --- Asset Path ---
    # Construct the absolute path to the mesh file relative to this script's location.
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        mesh_path = os.path.join(project_root, "assets", "integrating_sphere_0.3.obj")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Asset not found at '{mesh_path}'. Make sure the path is correct.")
    except Exception as e:
        print(f"Error resolving asset path: {e}")
        return
        
    # --- Simulation ---
    print("\n1. Initializing Simulator...")
    simulator = osg.Simulator()

    print("2. Building Scene from file...")
    start_build = time.time()
    simulator.build_scene_from_file(mesh_path, mesh_config)
    end_build = time.time()
    print(f"   ✅ Scene built in {end_build - start_build:.3f} seconds.")

    print("\n3. Running Simulation...")
    start_run = time.time()
    result = simulator.run(source, sim_config) # Argument order matches C++ signature
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