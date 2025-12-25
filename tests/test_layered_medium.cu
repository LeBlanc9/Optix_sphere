#include <iostream>
#include <numeric>
#include <cuda_runtime.h>
#include "layered_media/layered_medium.cuh"
#include "layered_media/media_simulator.cuh" // This now includes the necessary photon headers

using namespace phonder;

int main() {
    std::cout << "=== Layered Medium Monte Carlo Test ===" << std::endl;
    std::cout << std::endl;

    // 1. Create layered medium (3 layers)
    LayeredMedium medium = LayeredMedium(1.0f, 100.0f)  // ambient_n=1.0, width=100.0
        .add_layer(1.42f, 0.01f, 20.0f, 0.7f, 1.0f)  // Layer 1: n=1.42, mua=0.01, mus=20.0, g=0.7, d=1.0mm
        .add_layer(1.00f, 0.1f,  90.0f, 0.7f, 1.0f)  // Layer 2: n=1.00, mua=0.1,  mus=90.0, g=0.7, d=1.0mm
        .add_layer(1.42f, 0.3f,  80.0f, 0.7f, 1.0f); // Layer 3: n=1.42, mua=0.3,  mus=80.0, g=0.7, d=1.0mm

    std::cout << "Medium configuration:" << std::endl;
    std::cout << "  Num layers: " << medium.num_layers << std::endl;
    std::cout << "  Total thickness: " << medium.total_thickness << " mm" << std::endl;
    std::cout << "  Width: " << medium.width << " mm" << std::endl;
    std::cout << std::endl;

    // 2. Create point source (incident beam)
    phonder::CollimatedBeamSource source_params;
    source_params.position = make_float3(0.0f, 0.0f, -0.1f);
    source_params.direction = make_float3(0.0f, 0.0f, 1.0f);
    source_params.weight = 1.0;

    std::cout << "Source configuration:" << std::endl;
    std::cout << "  Type: Collimated Beam source" << std::endl;
    std::cout << "  Position: (0.0, 0.0, -0.1) mm" << std::endl;
    std::cout << "  Direction: (0.0, 0.0, 1.0)" << std::endl;
    std::cout << std::endl;

    // 3. Create MediaSimConfig
    MediaSimConfig media_config;
    media_config.medium = medium;
    media_config.source = source_params;
    media_config.gpu_id = 0;
    // media_config.reflected_radius = 1.0f;
    // media_config.transmitted_radius = 1.0f;

    // 4. Create and run the media simulator
    MediaSimulator media_sim(media_config);
    int num_photons_to_simulate = 1000000;  // 1M photons

    std::cout << "Running simulation with " << num_photons_to_simulate << " photons..." << std::endl;
    MediaSimulationResult result = media_sim.run_and_copy_to_cpu(num_photons_to_simulate);

    // 5. Analyze results
    double reflected_weight_sum = result.reflected_weight();
    double transmitted_weight_sum = result.transmitted_weight();
    
    double specular_reflection = result.specular_reflection_weight;
    double diffuse_reflection = reflected_weight_sum - specular_reflection;

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "  Reflected photons: " << result.reflected_batch.size() << std::endl;
    std::cout << "  Transmitted photons: " << result.transmitted_batch.size() << std::endl;
    std::cout << std::endl;

    std::cout << "Normalized weights (per incident photon):" << std::endl;
    std::cout << "  Total Reflected:    " << reflected_weight_sum / num_photons_to_simulate << std::endl;
    std::cout << "    - Specular:       " << specular_reflection / num_photons_to_simulate << std::endl;
    std::cout << "    - Diffuse:        " << diffuse_reflection / num_photons_to_simulate << std::endl;
    std::cout << "  Total Transmitted:  " << transmitted_weight_sum / num_photons_to_simulate << std::endl;
    std::cout << "  Total (R+T):        " << (reflected_weight_sum + transmitted_weight_sum) / num_photons_to_simulate << std::endl;
    std::cout << "  Absorbed:           " << 1.0 - (reflected_weight_sum + transmitted_weight_sum) / num_photons_to_simulate << std::endl;
    std::cout << std::endl;

    std::cout << "✅ Test completed successfully!" << std::endl;

    return 0;
}
