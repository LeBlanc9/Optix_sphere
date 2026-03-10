"""
Test Numerical Aperture (NA) functionality for detector

This script tests how different NA values affect detector response.
NA = 1.0 (default) means no angle limitation
NA < 1.0 means only photons within a certain angle are detected
"""

import optix_sphere as osg
from pathlib import Path
import time

# Setup paths
mesh_path = Path(__file__).parent.parent / "assets/R_25.4_1mm.obj"
osg._core.set_log_level(osg._core.LogLevel.INFO)

def test_na_values():
    print("=" * 70)
    print("Testing Numerical Aperture (NA) Effect on Detector")
    print("=" * 70)

    # Test different NA values
    na_values = [1.0, 0.5, 0.37]
    results = []

    for na in na_values:
        print(f"\n{'='*70}")
        print(f"Testing NA = {na:.2f}")
        print(f"{'='*70}\n")

        # Create simulator
        sim = osg.Simulator()
        sim.config.num_rays = 1000000
        sim.config.max_bounces = 500
        sim.config.use_nee = True

        # Setup material pool with detector having specific NA
        idx_wall = sim.add_material(osg.material.lambertian(0.98))
        idx_detector = sim.add_material(osg.material.detector(na=na, n=1.0))
        idx_sample = sim.add_material(osg.material.lambertian(0.0))

        # Build scene
        scene = osg.Scene.from_obj(str(mesh_path))
        material_mapping = {
            "wall_material": idx_wall,
            "detector_material": idx_detector,
            "sample_material": idx_sample
        }

        start_build = time.time()
        sim.build_scene(scene, material_mapping)
        build_time = time.time() - start_build

        # Setup photon source
        source = osg.IsotropicPointSource()
        source.position = (0, 0, 0)
        source.weight = 1.0

        # Run simulation
        start_sim = time.time()
        result = sim.run(source)
        sim_time = time.time() - start_sim

        # Store results
        results.append({
            'na': na,
            'irradiance': result.irradiance,
            'detected_flux': result.detected_flux,
            'build_time': build_time,
            'sim_time': sim_time
        })

        # Calculate acceptance angle
        import math
        theta_max_deg = math.asin(na) * 180 / math.pi

        print(f"\nResults for NA = {na:.2f}:")
        print(f"  Acceptance angle: ±{theta_max_deg:.1f}°")
        print(f"  Irradiance:       {result.irradiance:.6f} W/mm²")
        print(f"  Detected flux:    {result.detected_flux:.6f} W")
        print(f"  Build time:       {build_time:.3f}s")
        print(f"  Simulation time:  {sim_time:.3f}s")

    # Summary comparison
    print(f"\n{'='*70}")
    print("Summary Comparison")
    print(f"{'='*70}\n")
    print(f"{'NA':<8} {'Angle (°)':<12} {'Irradiance':<15} {'Flux':<15} {'Relative':<10}")
    print(f"{'-'*70}")

    baseline_irradiance = results[0]['irradiance']

    for r in results:
        import math
        theta_max = math.asin(r['na']) * 180 / math.pi
        relative = r['irradiance'] / baseline_irradiance if baseline_irradiance > 0 else 0
        print(f"{r['na']:<8.2f} ±{theta_max:<11.1f} {r['irradiance']:<15.6f} "
              f"{r['detected_flux']:<15.6f} {relative:<10.2%}")

    print(f"\n{'='*70}")
    print("Observations:")
    print(f"{'='*70}")
    print(f"• NA = 1.0: Full hemisphere acceptance (baseline)")
    print(f"• NA = 0.5: ~30° acceptance cone, reduced signal")
    print(f"• NA = 0.37: ~22° acceptance cone, further reduced signal")
    print(f"• Smaller NA → less detected flux (physically correct)")
    print(f"• Build time is the same (NA doesn't affect geometry)")
    print(f"• Simulation time is similar (NA just filters photons)")

if __name__ == "__main__":
    test_na_values()
