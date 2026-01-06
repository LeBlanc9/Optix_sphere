import numpy as np
import optix_sphere._core as osg
import time
import os


def setup_layered_medium():
    medium = osg.media.LayeredMedium(ambient_n=1.0, width=100.0)

    medium.add_layer(n=1.52, mua=0.001, mus=0.1, g=0.5, d=1.0)
    medium.add_layer(n=1.40, mua=0.05, mus=10.0, g=0.9, d=2.0)
    medium.add_layer(n=1.52, mua=0.001, mus=0.1, g=0.5, d=1.0)

    return medium


def setup_collimated_source():
    source = osg.CollimatedBeamSource()
    source.position = osg.float3(0.0, 0.0, -0.1)
    source.direction = osg.float3(0.0, 0.0, 1.0)
    source.weight = 1.0
    return source


def run_layered_media_simulation(medium, source, num_photons=int(1e6)):
    """
    Run Monte Carlo simulation through layered medium.

    Returns:
        device_result: MediaSimulationResult containing reflected and transmitted photon batches
    """
    print("\n" + "="*70)
    print("STEP 1: Layered Media Monte Carlo Simulation")
    print("="*70)

    print(f"\nMedium configuration:")
    print(f"  Number of layers: {medium.num_layers}")
    print(f"  Total thickness: {medium.total_thickness:.2f} mm")
    print(f"  Width: {medium.width:.2f} mm")

    # Configure simulation
    media_config = osg.media.MediaSimConfig()
    media_config.medium = medium
    media_config.source = source
    media_config.gpu_id = 0

    # Run simulation
    media_sim = osg.media.MediaSimulator(media_config)

    print(f"\nSimulating {num_photons:,} photons through layered medium...")
    start = time.time()
    device_result = media_sim.run(num_photons)
    elapsed = time.time() - start

    print(f"✅ Layered media simulation completed in {elapsed:.3f} seconds")

    # Get host copy for statistics
    host_result = device_result.to_host()

    # Calculate statistics
    R_diffuse = np.sum(host_result.reflected_batch.weights) / num_photons
    R_specular = host_result.specular_reflection_weight / num_photons
    R_total = R_diffuse + R_specular
    T_total = np.sum(host_result.transmitted_batch.weights) / num_photons
    A_total = 1.0 - R_total - T_total

    print(f"\n📊 Layered Media Results:")
    print(f"  Total Reflectance:    {R_total:.6f}")
    print(f"    - Specular:         {R_specular:.6f}")
    print(f"    - Diffuse:          {R_diffuse:.6f}")
    print(f"  Total Transmittance:  {T_total:.6f}")
    print(f"  Total Absorptance:    {A_total:.6f}")
    print(f"  Transmitted photons:  {host_result.transmitted_batch.size():,}")

    return device_result


def setup_integrating_sphere_geometry():
   # Choose sphere geometry
    # 25.4 mm diameter sphere with 0.01 mm port thickness
    mesh_path = os.path.join(
        "E:/workspace/Optix_sphere/assets/port_thickness",
        "integrating_sphere_25.4_0.01.obj"
    )

    # Define materials
    materials = {
        "wall_material": osg.material.lambertian(0.99),  # High reflectance coating
        "detector_material": osg.material.detector()
    }

    # Sphere geometric information
    sphere_info = {
        "diameter": 25.4,  # mm
        "radius": 12.7,    # mm
        "port_thickness": 0.01  # mm
    }

    return mesh_path, materials, sphere_info


def transform_photons_to_sphere_port(transmitted_batch, sphere_info):
    """
    Transform transmitted photons from layered medium coordinate system
    to integrating sphere coordinate system.

    Coordinate systems:
    - Layered medium: Sample at z=0 to z=total_thickness, beam along +z
    - Sphere: Typically centered at origin, port at specific location

    For sample placed at sphere port (z-axis port):
    - Port opening at z = -radius (bottom of sphere)
    - Transmitted photons need to be shifted to this location

    Args:
        transmitted_batch: PhotonBatch from media simulation
        sphere_info: Dictionary with sphere geometry

    Returns:
        transformed_batch: PhotonBatch positioned at sphere port
    """
    # For this example, assume:
    # - Sphere center at (0, 0, 0)
    # - Port is on bottom of sphere (z = -radius)
    # - Layered medium output at z = total_thickness
    # - Need to shift photons to enter sphere through bottom port

    # Port location (bottom of sphere, entering upward into +z)
    port_z = -sphere_info["radius"] + sphere_info["port_thickness"] + 0.01

    # Create offset to move photons to port location
    offset = osg.float3(0.0, 0.0, port_z)

    print(f"\n🔧 Photon coordinate transformation:")
    print(f"  Sphere port location: z = {port_z:.3f} mm")
    print(f"  Translation offset: (0, 0, {port_z:.3f}) mm")

    # Transform photon positions
    transformed_batch = osg.translate_photons(transmitted_batch, offset)

    return transformed_batch


def run_integrating_sphere_simulation(photon_batch, mesh_path, materials):
    """
    Run Monte Carlo simulation in integrating sphere using photon batch as source.

    Args:
        photon_batch: PhotonBatch to use as light source
        mesh_path: Path to sphere OBJ file
        materials: Material dictionary

    Returns:
        result: SimulationResult with detected flux and other metrics
    """
    print("\n" + "="*70)
    print("STEP 2: Integrating Sphere Monte Carlo Simulation")
    print("="*70)

    # Configure simulation
    sim_config = osg.SimConfig()
    sim_config.num_rays = photon_batch.size()  # Use all transmitted photons
    sim_config.max_bounces = 500
    sim_config.use_nee = True
    sim_config.random_seed = 42

    mesh_config = osg.MeshSceneConfig()

    # Build sphere scene
    simulator = osg.Simulator()

    print(f"\n🔨 Building integrating sphere scene...")
    print(f"  Mesh file: {os.path.basename(mesh_path)}")
    start = time.time()
    simulator.build_scene_from_file(mesh_path, materials, mesh_config)
    elapsed = time.time() - start
    print(f"✅ Scene built in {elapsed:.3f} seconds")

    # Run simulation with photon batch as source
    print(f"\n🚀 Tracing {photon_batch.size():,} photons through integrating sphere...")
    print(f"  Max bounces: {sim_config.max_bounces}")
    print(f"  Next Event Estimation: {'Enabled' if sim_config.use_nee else 'Disabled'}")

    start = time.time()
    result = simulator.run(photon_batch, sim_config)
    elapsed = time.time() - start

    print(f"✅ Sphere simulation completed in {elapsed:.3f} seconds")

    # Print results
    detector_area = simulator.get_detector_total_area()

    print(f"\n📊 Integrating Sphere Results:")
    print(f"  Detected Flux:        {result.detected_flux:.6f} W")
    print(f"  Detector Irradiance:  {result.irradiance:.6f} W/mm²")
    print(f"  Detector Area:        {detector_area:.6f} mm²")
    print(f"  Detected Rays:        {result.detected_rays:,} / {result.total_rays:,}")
    print(f"  Detection Efficiency: {result.detected_rays/result.total_rays*100:.2f}%")
    print(f"  Average Bounces:      {result.avg_bounces:.2f}")

    return result


def main():
    """
    Main function demonstrating end-to-end simulation workflow.
    """
    print("="*70)
    print("  End-to-End Simulation: Layered Media → Integrating Sphere")
    print("="*70)
    print("\nThis simulation mimics a realistic optical measurement:")
    print("  1. Laser beam → Tissue sample (layered media MC)")
    print("  2. Transmitted light → Integrating sphere (ray tracing MC)")
    print("  3. Detector measures total flux")

    # Set logging level
    # osg.set_log_level(osg.LogLevel.WARN)
    osg.set_log_level(osg.LogLevel.WARN)

    # ============================================================
    # STEP 1: Layered Media Simulation
    # ============================================================

    medium = setup_layered_medium()
    source = setup_collimated_source()

    num_photons = int(1e7)  # Increase for better statistics
    media_result = run_layered_media_simulation(medium, source, num_photons)

    mesh_path, materials, sphere_info = setup_integrating_sphere_geometry()
    transmitted_batch_transformed = transform_photons_to_sphere_port(
        media_result.transmitted_batch,
        sphere_info
    )

    # ============================================================
    # STEP 3: Integrating Sphere Simulation
    # ============================================================

    sphere_result = run_integrating_sphere_simulation(
        transmitted_batch_transformed,
        mesh_path,
        materials
    )

    # ============================================================
    # STEP 4: Combined Analysis
    # ============================================================

    print("\n" + "="*70)
    print("FINAL RESULTS: End-to-End Measurement")
    print("="*70)

    # Calculate transmittance based on sphere measurement
    # The detected flux represents the transmitted light
    host_media = media_result.to_host()
    T_from_media = np.sum(host_media.transmitted_batch.weights) / num_photons

    print(f"\n📈 Comparison:")
    print(f"  Transmittance (from layered media):  {T_from_media:.8f}")
    print(f"  Detected flux (from sphere):         {sphere_result.detected_flux/num_photons:.8f}")
    print(f"  \n  Note: These values represent different quantities:")
    print(f"    - T is the fraction of photons transmitted")
    print(f"    - Detected flux is the integrated signal at detector")

if __name__ == "__main__":
    main()
