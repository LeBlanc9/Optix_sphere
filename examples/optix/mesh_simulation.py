import optix_sphere as osg
from pathlib import Path
import time

def main():
    print("--- Python Mesh-Based Simulation ---")

    osg._core.set_log_level(osg._core.LogLevel.WARN)

    # Create photon source
    source = osg.IsotropicPointSource()
    source.position = (0, 0, 0)

    # --- Load Scene ---
    mesh_path = (Path(__file__).parent.parent.parent /
                 "assets/validations/port_thickness/integrating_sphere_25.4_0.01.obj")

    scene = osg.Scene.from_obj(str(mesh_path))

    # --- Setup Simulator ---
    simulator = osg.Simulator()
    simulator.config.num_rays = 2_000_000
    simulator.config.max_bounces = 500
    simulator.config.use_nee = True

    # NEW API: Define materials directly as a name->material dictionary
    materials = {
        "detector_material": osg.material.detector(),
        "wall_material": osg.material.lambertian(0.99)
    }

    # --- Build Scene ---
    start_build = time.time()
    # No more manual pool indexing! Just pass the dictionary.
    simulator.build_scene(scene, materials)
    end_build = time.time()
    print(f"   ✅ Scene built in {end_build - start_build:.3f} seconds.")

    # --- Run Simulation ---
    start_run = time.time()
    result = simulator.run(source)
    end_run = time.time()
    print(f"   ✅ Simulation finished in {end_run - start_run:.3f} seconds.")

    # --- Results ---
    print("\n--- Simulation Results ---")
    print(f"  Detected Flux:   {result.detected_flux:.6f} W")
    print(f"  Irradiance:      {result.irradiance:.6f} W/mm²")
    print(f"  Detected Rays:   {result.detected_rays:,} / {result.total_rays:,}")
    print(f"  Average Bounces: {result.avg_bounces:.2f}")


if __name__ == "__main__":
    main()
