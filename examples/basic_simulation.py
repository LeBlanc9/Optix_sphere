"""
Basic integrating sphere simulation example

This script demonstrates how to use the optix_sphere package
to simulate a simple integrating sphere.
"""

import optix_sphere as opt
import time

def main():
    print("=== OptiX Sphere Simulation ===\n")

    # 1. Create sphere geometry
    sphere = opt.Sphere()
    sphere.radius = 50.0  # mm
    sphere.reflectance = 0.98
    print(f"Sphere: radius={sphere.radius}mm, reflectance={sphere.reflectance}")

    # 2. Create light source
    light = opt.LightSource()
    light.power = 1.0  # W
    print(f"Light: power={light.power}W")

    # 3. Create detector with chord surface geometry
    detector = opt.Detector()
    port_hole_radius = 0.564  # mm (area ≈ 1mm²)
    opt.configure_detector_chord(detector, sphere, port_hole_radius)
    print(f"Detector: radius={detector.radius}mm, position=({detector.position.x:.2f}, {detector.position.y:.2f}, {detector.position.z:.2f})")

    # 4. Configure simulation
    config = opt.SimConfig()
    config.num_rays = 1_000_000  # 1M rays
    config.max_bounces = 500
    config.use_nee = False
    config.random_seed = int(time.time())
    print(f"Config: {config.num_rays} rays, max {config.max_bounces} bounces\n")

    # 5. Create simulator and setup scene
    print("Initializing OptiX...")
    sim = opt.Simulator()
    sim.setup_scene(sphere, detector)

    # 6. Run simulation
    print("Running simulation...")
    start_time = time.time()
    result = sim.run(config, light)
    elapsed = time.time() - start_time

    # 7. Calculate theory
    port_area = opt.PI * port_hole_radius ** 2
    theory = opt.calculate_theory(
        sphere.radius,
        sphere.reflectance,
        light.power,
        port_area
    )

    # 8. Print results
    print(f"\nSimulation completed in {elapsed:.2f}s\n")
    print("=" * 70)
    print(f"{'Metric':<30} {'Simulation':<20} {'Theory':<20}")
    print("=" * 70)
    print(f"{'Irradiance (W/mm²)':<30} {result.irradiance:<20.6e} {theory.avg_irradiance:<20.6e}")
    print(f"{'Detected flux (W)':<30} {result.detected_flux:<20.6e} {theory.detected_flux:<20.6e}")
    print(f"{'Detected rays':<30} {result.detected_rays:<20} {'N/A':<20}")
    print(f"{'Average bounces':<30} {result.avg_bounces:<20.2f} {'N/A':<20}")
    print("=" * 70)

    # Calculate error
    error = abs(result.irradiance - theory.avg_irradiance) / theory.avg_irradiance * 100
    print(f"\nRelative error: {error:.3f}%")

if __name__ == "__main__":
    main()
