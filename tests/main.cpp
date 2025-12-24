#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <spdlog/spdlog.h>
#include "simulator.h" // <-- The new unified high-level API
#include "photon/sources.h"
#include "theory/theory.h" // <-- New theory API
#include "constants.h"

namespace fs = std::filesystem;

int main() {
    try {
        // --- Simulation Configuration ---
        SimConfig config;
        config.num_rays = 1'000'000;
        config.max_bounces = 500;
        config.random_seed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );

        // --- Create Photon Source ---
        phonder::IsotropicPointSource source_params;
        source_params.position = {0.0f, 0.0f, 0.0f};
        source_params.weight = 1.0; // Represents 1W total power
        phonder::PhotonSource light_source = source_params;

        spdlog::info("=== Test Configuration ===");
        spdlog::info("  Source Type: Isotropic Point Source");
        spdlog::info("  Total Rays: {}", config.num_rays);
        spdlog::info("  Max Bounces: {}", config.max_bounces);
        spdlog::info("  Random Seed: {}", config.random_seed);

        // --- High-level API Setup ---
        // 1. Create the simulator
        Simulator simulator;

        // Configure and build the scene from a file
        fs::path mesh_path = fs::path("E:/workspace/Optix_sphere/assets") / "integrating_sphere_0.3.obj";
        MeshSceneConfig scene_config;
        scene_config.default_reflectance = 0.98f;
        
        simulator.build_scene_from_file(mesh_path.string(), scene_config);

        // --- Theoretical Calculation ---
        // New object-oriented approach for theoretical sphere
        float sphere_radius_for_theory = 50.0f; // Known from the mesh model's parameters
        float wall_reflectance_for_theory = scene_config.default_reflectance;
        float incident_power_for_theory = phonder::get_source_power(light_source);

        // Create a theoretical sphere model
        theory::TheoreticalIntegratingSphere theoretical_sphere_model(
            sphere_radius_for_theory,
            wall_reflectance_for_theory
        );

        // Add the detector port to the theoretical model
        float detector_area_for_theory = simulator.get_detector_total_area();
        float detector_radius_for_theory = std::sqrt(detector_area_for_theory / PI);
        theoretical_sphere_model.add_port(detector_radius_for_theory, 0.0f); // Assume 0 reflectance for detector port in theory

        // Perform the theoretical calculation
        TheoryResult theory_result = theory::TheoryCalculator::calculate(
            theoretical_sphere_model,
            incident_power_for_theory
        );

        // Manually calculate theoretical detected flux for comparison display
        double theoretical_detected_flux = theory_result.avg_irradiance * detector_area_for_theory;


        // --- Run Simulations using the high-level API ---
        SimulationResult mesh_non_nee_result, mesh_nee_result;

        // Run Non-NEE
        spdlog::info("\n🔹 Running Mesh Non-NEE...");
        config.use_nee = false;
        auto start_time_non_nee = std::chrono::high_resolution_clock::now();
        mesh_non_nee_result = simulator.run(light_source, config);
        auto end_time_non_nee = std::chrono::high_resolution_clock::now();
        auto duration_non_nee = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_non_nee - start_time_non_nee);
        spdlog::info("  ✅ Mesh Non-NEE took: {} ms", duration_non_nee.count());

        // Run NEE
        spdlog::info("\n🔹 Running Mesh NEE...");
        config.use_nee = true;
        auto start_time_nee = std::chrono::high_resolution_clock::now();
        mesh_nee_result = simulator.run(light_source, config);
        auto end_time_nee = std::chrono::high_resolution_clock::now();
        auto duration_nee = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_nee - start_time_nee);
        spdlog::info("  ✅ Mesh NEE took: {} ms", duration_nee.count());

        // --- Print Results (adjusted for new TheoryResult) ---
        std::cout << "\n\n=== SIMULATION RESULTS & COMPARISON ===\n";
        std::cout << std::fixed << std::setprecision(6);

        std::cout << "\nTheory (Simplified Model):\n";
        std::cout << "  Irradiance:     " << theory_result.avg_irradiance << " W/mm²\n";
        std::cout << "  Detected flux:  " << theoretical_detected_flux << " W\n"; // Use manually calculated flux

        double mesh_non_nee_error = std::abs(mesh_non_nee_result.irradiance - theory_result.avg_irradiance)
                                    / theory_result.avg_irradiance * 100.0;
        double mesh_nee_error = std::abs(mesh_nee_result.irradiance - theory_result.avg_irradiance)
                                / theory_result.avg_irradiance * 100.0;

        std::cout << std::setprecision(6);
        std::cout << "\nMesh Geometry (Non-NEE):\n";
        std::cout << "  Irradiance:     " << mesh_non_nee_result.irradiance << " W/mm²";
        std::cout << "  (error: " << std::setprecision(3) << mesh_non_nee_error << " %)\n";
        std::cout << std::setprecision(6);
        std::cout << "  Detected flux:  " << mesh_non_nee_result.detected_flux << " W\n";
        std::cout << "  Detected rays:  " << mesh_non_nee_result.detected_rays << " / " << config.num_rays;
        std::cout << "  (avg bounces: " << std::setprecision(2) << mesh_non_nee_result.avg_bounces << ")\n";

        std::cout << std::setprecision(6);
        std::cout << "\nMesh Geometry (NEE):\n";
        std::cout << "  Irradiance:     " << mesh_nee_result.irradiance << " W/mm²";
        std::cout << "  (error: " << std::setprecision(3) << mesh_nee_error << " %)\n";
        std::cout << std::setprecision(6);
        std::cout << "  Detected flux:  " << mesh_nee_result.detected_flux << " W\n";
        std::cout << "  Detected rays:  " << mesh_nee_result.detected_rays << " / " << config.num_rays;
        std::cout << "  (avg bounces: " << std::setprecision(2) << mesh_nee_result.avg_bounces << ")\n";

        std::cout << "\n======================================\n" << std::endl;

    } catch (const std::exception& e) {
        spdlog::error("Fatal error in main: {}", e.what());
        return 1;
    }

    return 0;
}
