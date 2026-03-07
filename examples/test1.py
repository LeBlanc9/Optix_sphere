
import numpy as np
from optix_sphere import _core

medium = _core.media.LayeredMedium(ambient_n=1.0, width=33.0)
medium.add_layer(1.37, 0.2, 10, 0.9, 0.02) 

source_params = _core.CollimatedBeamSource()
source_params.position = _core.float3(0.0, 0.0, -0.1)
source_params.direction = _core.float3(0.0, 0.0, 1.0)
source_params.weight = 1.0

media_config = _core.media.MediaSimConfig()
media_config.medium = medium
media_config.source = source_params
media_config.gpu_id = 0
# media_config.reflected_radius = 25.4 / 2
# media_config.transmitted_radius = 25.4 / 2

media_sim = _core.media.MediaSimulator(media_config)
num_photons_to_simulate = int(1e7)

print(f"Running simulation with {num_photons_to_simulate} photons...")
device_result = media_sim.run(num_photons_to_simulate)

reflected_weight_sum = device_result.reflected_batch.total_weight()
transmitted_weight_sum = device_result.transmitted_batch.total_weight()


total_reflected = reflected_weight_sum / num_photons_to_simulate
total_transmitted = transmitted_weight_sum / num_photons_to_simulate
print(total_reflected, total_transmitted)