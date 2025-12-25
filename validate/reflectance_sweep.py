import optix_sphere._core as osg
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time

# --- Simulation Parameters ---
NUM_PHOTONS = 3_000_000
MAX_BOUNCES = 1000
SPHERE_RADIUS_MM = 154.0 / 2
INCIDENT_POWER_W = 1.0
NUM_REPETITIONS = 5 # Number of times to run the simulation for each reflectance value

# --- Helper function to run a single simulation ---
def run_simulation(simulator, reflectance, sim_config, source, mesh_path):
    """
    Builds the scene with a specific reflectance and runs the simulation.
    """
    mesh_config = osg.MeshSceneConfig()
    mesh_config.default_reflectance = reflectance
    
    simulator.build_scene_from_file(mesh_path, mesh_config)
    
    sim_config.use_nee = True
    result = simulator.run(source, sim_config)
    
    return result

# --- Helper function to run a theoretical calculation ---
def run_theory(reflectance, sphere_radius, detector_area, incident_power):
    """
    Calculates the theoretical detected flux.
    """
    theoretical_sphere = osg.TheoreticalIntegratingSphere(sphere_radius, reflectance)
    detector_radius = np.sqrt(detector_area / osg.PI)

    theoretical_sphere.add_port(detector_radius, 0.0)
    theoretical_sphere.add_port(25.4/2, 0.0)
    theoretical_sphere.add_port(25.4/2, 0.0)

    theory_result = osg.TheoryCalculator.calculate(theoretical_sphere, incident_power)
    theoretical_flux = theory_result.avg_irradiance * detector_area
    return theoretical_flux

def run_wall_reflectance_sweep():
    """
    Runs the validation over a range of wall reflectances, with repetitions,
    and saves statistical results.
    """
    print("--- Running Reflectance Sweep Validation (with repetitions) ---")
    
    # --- Base Configurations ---
    sim_config = osg.SimConfig()
    sim_config.num_rays = NUM_PHOTONS
    sim_config.max_bounces = MAX_BOUNCES
    
    source = osg.IsotropicPointSource()
    source.position = osg.float3(0, 0, 0)
    source.weight = INCIDENT_POWER_W

    osg.set_log_level(osg.LogLevel.WARN)

    # --- Asset Path ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        # mesh_path = os.path.join(project_root, "assets", "integrating_sphere_0.3.obj")
        # mesh_path = os.path.join(project_root, "assets", "integrating_sphere_25.4_1.obj")
        # mesh_path = os.path.join(project_root, "assets", "integrating_sphere_25.4_5.obj")
        mesh_path = os.path.join(project_root, "assets", "integrating_sphere_25.4_ideal.obj")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Asset not found at '{mesh_path}'.")
    except Exception as e:
        print(f"Error resolving asset path: {e}")
        return

    # --- Initialize Simulator and get detector area ---
    simulator = osg.Simulator()
    print("Performing initial build to get detector geometry...")
    initial_mesh_config = osg.MeshSceneConfig()
    initial_mesh_config.default_reflectance = 0.98
    simulator.build_scene_from_file(mesh_path, initial_mesh_config)
    detector_area = simulator.get_detector_total_area()
    print(f"  -> Detector Area from mesh: {detector_area:.4f} mm²")

    # --- Main Sweep Loop ---
    wall_reflectances = np.linspace(0.95, 0.99, 40)
    results = []
    
    print(f"\nRunning validation sweep for wall reflectance ({NUM_REPETITIONS} repetitions each)...")
    for r_w in tqdm(wall_reflectances, desc="Reflectance Sweep"):
        
        repetition_fluxes = []
        for i in range(NUM_REPETITIONS):
            # Give each run a unique random seed for statistical independence
            # Ensure the seed fits into a 32-bit unsigned int
            sim_config.random_seed = (int(time.time() * 1000) + i) % (2**32)
            
            sim_result = run_simulation(simulator, r_w, sim_config, source, mesh_path)
            repetition_fluxes.append(sim_result.detected_flux)

        # Calculate statistics
        flux_mean = np.mean(repetition_fluxes)
        flux_std = np.std(repetition_fluxes)

        # Normalize the simulation results
        # Each photon starts with a weight equal to INCIDENT_POWER_W (e.g., 1.0W)
        # So, the sum of detected weights (flux_mean) must be scaled back by NUM_PHOTONS.
        normalized_flux_mean = (flux_mean / NUM_PHOTONS) * INCIDENT_POWER_W
        normalized_flux_std = (flux_std / NUM_PHOTONS) * INCIDENT_POWER_W # Std dev also needs scaling

        # Run Analytical Calculation
        classical_flux = run_theory(r_w, SPHERE_RADIUS_MM, detector_area, INCIDENT_POWER_W)
        
        results.append({
            "wall_reflectance": r_w,
            "simulation_flux_mean": normalized_flux_mean,
            "simulation_flux_std": normalized_flux_std,
            "theory_flux": classical_flux,
        })

    # --- Save results to CSV ---
    df = pd.DataFrame(results)
    df['relative_error_%'] = 100 * (df['simulation_flux_mean'] - df['theory_flux']) / df['theory_flux']

    output_filename = "sweep_reflectance.csv"
    df.to_csv(output_filename, index=False)
    
    print("\n--- Sweep Complete ---")
    print(df.to_string())
    print(f"\nResults with statistics saved to {output_filename}")

if __name__ == "__main__":
    run_wall_reflectance_sweep()