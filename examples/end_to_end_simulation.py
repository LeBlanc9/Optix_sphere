import optix_sphere as osg
import time
import random
from pathlib import Path


def main():
    # osg.set_log_level(osg.LogLevel.WARN)

    num_photons = int(1e7)

    # 1. layered media simulation
    medium = osg.media.LayeredMedium(ambient_n=1.0, width=100.0)
    medium.add_layer(n=1.42, mua=0.001, mus=0.1, g=0.5, d=1.25)
    medium.add_layer(n=1.33, mua=0.05, mus=10.0, g=0.9, d=1)
    medium.add_layer(n=1.42, mua=0.001, mus=0.1, g=0.5, d=1.25)


    source = osg.CollimatedBeamSource()
    source.position = osg.float3(0.0, 0.0, -0.1)
    source.direction = osg.float3(0.0, 0.0, 1.0)
    source.weight = 1.0

    media_config = osg.media.MediaSimConfig()
    media_config.medium = medium
    media_config.source = source
    media_config.gpu_id = 0
    media_config.reflected_radius = 25.4/2
    media_config.transmitted_radius = 25.4/2

    media_sim = osg.media.MediaSimulator(media_config)

    print(f"\nSimulating {num_photons:,} photons through layered medium...")
    start = time.time()
    device_result = media_sim.run(num_photons)
    elapsed = time.time() - start
    print(f"✅ Layered media simulation completed in {elapsed:.3f} seconds")

    R_total = device_result.reflected_batch.total_weight() / num_photons
    T_total = device_result.transmitted_batch.total_weight() / num_photons
    print(f"\n📊 Layered Media Results:")
    print(f"  Total Reflectance:    {R_total:.6f}")
    print(f"  Total Transmittance:  {T_total:.6f}")


    offset = osg.float3(0.0, 0.0, (152.4 / 2))
    osg.translate_photons_inplace(device_result.reflected_batch, offset)

    # 2. integrating sphere simulation
    sim_config = osg.SimConfig()
    sim_config.num_rays = device_result.reflected_batch.size()
    sim_config.max_bounces = 500
    sim_config.use_nee = True
    sim_config.random_seed = random.randint(1, 1000000)


    mesh_path = (Path(__file__).parent.parent / "assets/R_25.4_1mm.obj")
    
    # NEW RECOMMENDED API: Define materials directly in a dictionary
    materials = {
        "wall_material": osg.material.lambertian(0.99),  # High reflectance coating
        "detector_material": osg.material.detector(),
        "sample_material": osg.material.lambertian(R_total)  # Sample with reflectance from layered media
    }


    simulator = osg.Simulator()
    simulator.config = sim_config

    print(f"\n🔨 Building integrating sphere scene...")
    start = time.time()
    scene = osg.Scene.from_obj(str(mesh_path))
    # Direct name-based build
    simulator.build_scene(scene, materials)
    elapsed = time.time() - start
    print(f"✅ Scene built in {elapsed:.3f} seconds")

    start = time.time()
    result = simulator.run(device_result.reflected_batch)
    elapsed = time.time() - start

    print(f"✅ Sphere simulation completed in {elapsed:.3f} seconds")

    print(f"  Detected Flux:        {result.detected_flux:.6f} W")
    print(f"  Detector Irradiance:  {result.irradiance:.6f} W/mm²")
    print(f"  Detected Rays:        {result.detected_rays:,} / {result.total_rays:,}")
    print(f"  Average Bounces:      {result.avg_bounces:.2f}")

if __name__ == "__main__":
    main()
