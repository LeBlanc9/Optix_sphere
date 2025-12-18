"""
Parameter sweep example

This script demonstrates how to use optix_sphere to perform
parameter sweeps for integrating sphere analysis.
"""

import optix_sphere as opt
import numpy as np
import time

def sweep_reflectance():
    """Sweep reflectance values and compare with theory"""
    print("=== Reflectance Sweep ===\n")

    # Setup
    reflectances = np.linspace(0.90, 0.99, 10)
    results = []

    # Create components
    sphere = opt.Sphere()
    sphere.radius = 50.0

    light = opt.LightSource()
    light.power = 1.0

    detector = opt.Detector()
    opt.configure_detector_chord(detector, sphere, 0.564)

    config = opt.SimConfig()
    config.num_rays = 500_000  # Faster for sweeps
    config.max_bounces = 500
    config.random_seed = 12345  # Fixed seed for reproducibility

    # Create simulator
    sim = opt.Simulator()

    port_area = opt.PI * 0.564 ** 2

    for rho in reflectances:
        sphere.reflectance = rho

        # Re-setup scene with new reflectance
        sim.setup_scene(sphere, detector)

        # Run simulation
        print(f"Reflectance {rho:.3f}...", end=" ", flush=True)
        start = time.time()
        result = sim.run(config, light)
        elapsed = time.time() - start

        # Theory
        theory = opt.calculate_theory(sphere.radius, rho, light.power, port_area)

        # Calculate error
        error = abs(result.irradiance - theory.avg_irradiance) / theory.avg_irradiance * 100

        results.append({
            'reflectance': rho,
            'sim_irradiance': result.irradiance,
            'theory_irradiance': theory.avg_irradiance,
            'error_percent': error,
            'time': elapsed
        })

        print(f"Error: {error:.2f}%  Time: {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Reflectance':<15} {'Sim (W/mm²)':<15} {'Theory (W/mm²)':<15} {'Error (%)':<15}")
    print("=" * 80)
    for r in results:
        print(f"{r['reflectance']:<15.3f} {r['sim_irradiance']:<15.6e} "
              f"{r['theory_irradiance']:<15.6e} {r['error_percent']:<15.3f}")
    print("=" * 80)

    avg_error = np.mean([r['error_percent'] for r in results])
    print(f"\nAverage error: {avg_error:.3f}%")
    print(f"Total time: {sum(r['time'] for r in results):.1f}s")

if __name__ == "__main__":
    sweep_reflectance()
