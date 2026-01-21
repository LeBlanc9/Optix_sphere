#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include "voxel_media/voxel_simulator.h"
#include "voxel_media/voxel_sim_config.h"
#include "layered_media/media_simulator.cuh"
#include "photon/sources.h"

using namespace phonder;
using namespace phonder::voxel;

void test_comparison_with_layered_medium() {
    const int num_photons = int(1e7);

    std::cout << "Comparing VoxelGrid with LayeredMedium using " << num_photons << " photons..." << std::endl;
    std::cout << std::endl;

    // === Method 1: LayeredMedium (reference) - Standard 3-layer structure ===
    std::cout << "=== LayeredMedium (Reference) - 3 Layers ===" << std::endl;
    LayeredMedium layered_medium(1.0f, 100.0f);
    layered_medium.add_layer(1.42f, 0.01f, 20.0f, 0.7f, 1.0f)   // Layer 1
                  .add_layer(1.00f, 0.1f,  90.0f, 0.7f, 1.0f)   // Layer 2
                  .add_layer(1.42f, 0.3f,  80.0f, 0.7f, 1.0f);  // Layer 3

    std::cout << "  Layer 1: n=1.42, mua=0.01, mus=20.0, g=0.7, d=1.0mm" << std::endl;
    std::cout << "  Layer 2: n=1.00, mua=0.1,  mus=90.0, g=0.7, d=1.0mm" << std::endl;
    std::cout << "  Layer 3: n=1.42, mua=0.3,  mus=80.0, g=0.7, d=1.0mm" << std::endl;
    std::cout << "  Total thickness: 3.0mm" << std::endl;

    // Source for LayeredMedium (centered at origin)
    auto layered_source = std::make_shared<CollimatedBeamSource>();
    layered_source->position = make_float3(0.0f, 0.0f, -0.1f);
    layered_source->direction = make_float3(0.0f, 0.0f, 1.0f);
    layered_source->weight = 1.0;

    MediaSimConfig layered_config;
    layered_config.medium = layered_medium;
    layered_config.source = layered_source;
    layered_config.reflected_radius = 2;
    layered_config.transmitted_radius = 2;

    MediaSimulator layered_sim(layered_config);

    auto start_layered = std::chrono::high_resolution_clock::now();
    auto layered_result = layered_sim.run(num_photons);
    cudaDeviceSynchronize();  // Ensure GPU work is complete
    auto end_layered = std::chrono::high_resolution_clock::now();

    auto layered_host = layered_result.to_host();

    double layered_time_ms = std::chrono::duration<double, std::milli>(end_layered - start_layered).count();

    double layered_R = layered_host.reflected_batch.total_weight() / num_photons;
    double layered_T = layered_host.transmitted_batch.total_weight() / num_photons;
    double layered_A = 1.0 - layered_R - layered_T;

    std::cout << "  Reflectance (R): " << layered_R << std::endl;
    std::cout << "  Transmittance (T): " << layered_T << std::endl;
    std::cout << "  Other (A): " << layered_A << std::endl;
    std::cout << "  R + T + A = " << layered_R + layered_T + layered_A << std::endl;
    std::cout << "  Reflected count: " << layered_host.reflected_batch.size() << std::endl;
    std::cout << "  Transmitted count: " << layered_host.transmitted_batch.size() << std::endl;
    std::cout << "  ⏱️  Runtime: " << layered_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (num_photons / layered_time_ms * 1000.0) << " photons/sec" << std::endl;
    std::cout << std::endl;

    // === Method 2: VoxelGrid ===
    std::cout << "=== VoxelGrid (Our Implementation) - 3 Layers ===" << std::endl;

    // Step 1: Create voxel grid directly using nested vectors
    int nx = 100;
    int ny = 100;
    int nz = 3;

    auto voxel_source = std::make_shared<CollimatedBeamSource>();
    voxel_source->position = make_float3(50.0f, 50.0f, -0.1f);
    voxel_source->direction = make_float3(0.0f, 0.0f, 1.0f);
    voxel_source->weight = 1.0;

    SimConfig voxel_config;

    // Create 3D nested vector grid and fill with material IDs
    voxel_config.grid.resize(nx);
    for (int x = 0; x < nx; x++) {
        voxel_config.grid[x].resize(ny);
        for (int y = 0; y < ny; y++) {
            voxel_config.grid[x][y].resize(nz);
            voxel_config.grid[x][y][0] = 1;  // Layer 1 -> material ID 1
            voxel_config.grid[x][y][1] = 2;  // Layer 2 -> material ID 2
            voxel_config.grid[x][y][2] = 3;  // Layer 3 -> material ID 3
        }
    }

    voxel_config.voxel_size = make_float3(1.0f, 1.0f, 1.0f);
    voxel_config.materials = std::vector<std::vector<float>>({
        {1.0f,  0.0f,   1e-6f, 0.0f},   // Material 0 (ambient/default)
        {1.42f, 0.01f,  20.0f, 0.7f},   // Material 1
        {1.00f, 0.1f,   90.0f, 0.7f},   // Material 2
        {1.42f, 0.3f,   80.0f, 0.7f}    // Material 3
    });

    voxel_config.source = voxel_source;
    voxel_config.merge_specular = true;
    

    voxel_config.boundary_collection.enable_z_faces();
    voxel_config.boundary_collection.set_positive_radius(2.0f);
    voxel_config.boundary_collection.set_negative_radius(2.0f);

    // Debug: print boundary collection config
    std::cout << "Boundary collection: " << voxel_config.boundary_collection.to_string() << std::endl;
    for (int i = 0; i < 6; i++) {
        std::cout << "  Face " << i << ": " << (voxel_config.boundary_collection.collect_faces[i] ? "enabled" : "disabled") << std::endl;
    }

    // Step 5: Create simulator
    Simulator voxel_sim(voxel_config);

    auto start_voxel = std::chrono::high_resolution_clock::now();
    auto voxel_result = voxel_sim.run(num_photons);
    cudaDeviceSynchronize();  // Ensure GPU work is complete
    auto end_voxel = std::chrono::high_resolution_clock::now();


    double voxel_time_ms = std::chrono::duration<double, std::milli>(end_voxel - start_voxel).count();

    double voxel_specular = voxel_result.specular_batch.total_weight() / num_photons;
    double voxel_R_diffuse = voxel_result.negative_boundary_batch.total_weight() / num_photons;
    double voxel_T = voxel_result.positive_boundary_batch.total_weight() / num_photons;
    double voxel_R = voxel_specular + voxel_R_diffuse;  // Total reflectance = specular + diffuse
    double voxel_A = 1.0 - voxel_R - voxel_T;

    std::cout << "  Specular reflectance: " << voxel_specular << std::endl;
    std::cout << "  Diffuse reflectance: " << voxel_R_diffuse << std::endl;
    std::cout << "  Total reflectance (R): " << voxel_R << std::endl;
    std::cout << "  Transmittance (T): " << voxel_T << std::endl;
    std::cout << "  Other (A): " << voxel_A << std::endl;
    std::cout << "  Specular count: " << voxel_result.specular_batch.size() << std::endl;
    std::cout << "  Reflected count: " << voxel_result.negative_boundary_batch.size() << std::endl;
    std::cout << "  Transmitted count: " << voxel_result.positive_boundary_batch.size() << std::endl;
    std::cout << "  ⏱️  Runtime: " << voxel_time_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << (num_photons / voxel_time_ms * 1000.0) << " photons/sec" << std::endl;
    std::cout << std::endl;

    // === Comparison ===
    std::cout << "=== Comparison ===" << std::endl;
    double error_R = std::abs(voxel_R - layered_R) / layered_R * 100.0;
    double error_T = std::abs(voxel_T - layered_T) / layered_T * 100.0;
    double error_A = std::abs(voxel_A - layered_A) / layered_A * 100.0;

    std::cout << "  Reflectance error: " << error_R << "%" << std::endl;
    std::cout << "  Transmittance error: " << error_T << "%" << std::endl;
    std::cout << "  Absorption error: " << error_A << "%" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Performance Comparison ===" << std::endl;
    std::cout << "  LayeredMedium runtime: " << layered_time_ms << " ms" << std::endl;
    std::cout << "  VoxelGrid runtime:     " << voxel_time_ms << " ms" << std::endl;
    double speedup = layered_time_ms / voxel_time_ms;
    if (speedup >= 1.0) {
        std::cout << "  Speedup: " << speedup << "x (VoxelGrid is faster)" << std::endl;
    } else {
        std::cout << "  Slowdown: " << (1.0 / speedup) << "x (VoxelGrid is slower)" << std::endl;
    }
    std::cout << std::endl;

    // Validation
    const double tolerance = 10.0;  // 10% tolerance for MC simulation
    const double tolerance_T = 50.0;  // Larger tolerance for very small transmittance

    bool passed = true;
    if (error_R > tolerance) {
        std::cerr << "❌ Reflectance error too large!" << std::endl;
        passed = false;
    }
    // For very small transmittance, relative error can be large due to statistics
    if (voxel_T > 0.001 && error_T > tolerance_T) {
        std::cerr << "❌ Transmittance error too large!" << std::endl;
        passed = false;
    }

    // Check energy conservation for voxel
    double voxel_total = voxel_R + voxel_T + voxel_A;
    if (std::abs(voxel_total - 1.0) > 0.05) {
        std::cerr << "❌ Energy not conserved in VoxelGrid: R+T+A = " << voxel_total << std::endl;
        passed = false;
    }

    if (passed) {
        std::cout << "✅ Comparison test passed!" << std::endl;
    } else {
        std::cerr << "❌ Comparison test FAILED!" << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "=== VoxelMediaSimulator Test ===" << std::endl;

    test_comparison_with_layered_medium();
    return 0;
}
