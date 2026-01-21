import numpy as np
from optix_sphere import _core

def main():
    # 1. Create a layered medium with three distinct layers.
    medium = _core.media.LayeredMedium(ambient_n=1.0, width=100.0)
    # medium.add_layer(n=1.42, mua=0.01, mus=20.0, g=0.7, d=1.0)  # Layer 1
    # medium.add_layer(n=1.00, mua=0.1,  mus=90.0, g=0.7, d=1.0)  # Layer 2 (fixed: was 1.32)
    # medium.add_layer(n=1.42, mua=0.3,  mus=80.0, g=0.7, d=1.0)  # Layer 3

    n_glass = 1.458  # 样品容器折射率
    mua_glass = 0.001
    mus_glass = 0.001
    def create_medium(mua, mus, thickness, n=1.338, g=0.9):
        medium = _core.media.LayeredMedium(ambient_n=1.0, width=33.0)
        medium.add_layer(n_glass, mua_glass, mus_glass, 1.0, 1.25)  # 上玻璃
        medium.add_layer(n, mua, mus, g, thickness)                 # 样品
        medium.add_layer(n_glass, mua_glass, mus_glass, 1.0, 1.25)  # 下玻璃
        return medium

    medium = create_medium(mua=0.01, mus=10.0, thickness=1.0)


    print("Medium configuration:")
    print(f"  Num layers: {medium.num_layers}")
    print(f"  Total thickness: {medium.total_thickness:.2f} mm")
    print(f"  Width: {medium.width} mm")

    # 2. Define a collimated beam source pointing along the z-axis.
    source_params = _core.CollimatedBeamSource()
    source_params.position = _core.float3(0.0, 0.0, -0.1)
    source_params.direction = _core.float3(0.0, 0.0, 1.0)
    source_params.weight = 1.0

    print("Source configuration:")
    print("  Type: Collimated Beam source")
    print(f"  Position: ({source_params.position.x}, {source_params.position.y}, {source_params.position.z}) mm")
    print(f"  Direction: ({source_params.direction.x}, {source_params.direction.y}, {source_params.direction.z})")

    # 3. Set up the simulation configuration.
    media_config = _core.media.MediaSimConfig()
    media_config.medium = medium
    media_config.source = source_params
    media_config.gpu_id = 0
    media_config.reflected_radius = 25.4 / 2
    media_config.transmitted_radius = 25.4 / 2

    # 4. Initialize the simulator and run the Monte Carlo simulation.
    media_sim = _core.media.MediaSimulator(media_config)
    num_photons_to_simulate = int(1e7)

    print(f"Running simulation with {num_photons_to_simulate} photons...")
    device_result = media_sim.run(num_photons_to_simulate)



    # 5. Print a summary of the results.
    reflected_weight_sum = device_result.reflected_batch.total_weight()
    transmitted_weight_sum = device_result.transmitted_batch.total_weight()


    print(f"  Reflected photons: {device_result.reflected_batch.size()}")
    print(f"  Transmitted photons: {device_result.transmitted_batch.size()}")

    print("Normalized weights (per incident photon):")
    total_reflected = reflected_weight_sum / num_photons_to_simulate
    total_transmitted = transmitted_weight_sum / num_photons_to_simulate
    
    print(f"  Total Reflected:    {total_reflected:.6f}")
    print(f"  Total Transmitted:  {total_transmitted:.6f}")

if __name__ == "__main__":
    main()