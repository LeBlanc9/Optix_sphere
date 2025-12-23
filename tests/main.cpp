#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include "core/optix_context.h"
#include "scene/scene.h"
#include "simulation/path_tracer.h"
#include "theory/theory.h"
#include "scene/scene_types.h"
#include "constants.h"
#include "embedded_ptx.h"  // 嵌入的 PTX 代码

namespace fs = std::filesystem;

int main() {
    try {
        OptixContext context;

        Sphere sphere_geom;
        sphere_geom.radius = 50.0f;              // 50 mm radius
        sphere_geom.reflectance = 0.98f;
        sphere_geom.center = {0.0f, 0.0f, 0.0f};

        LightSource light;
        light.position = {0.0f, 0.0f, 0.0f};     // center of sphere
        light.power = 1.0f;                       // 1 W

        // 配置探测器 - 使用弦面几何（在球面上开孔）
        Detector detector;
        float port_hole_radius = 0.3f;  // 开孔半径 (mm)
        configure_detector_chord(detector, sphere_geom, port_hole_radius);

        float port_hole_area = PI * port_hole_radius * port_hole_radius;  // 计算面积供理论使用
        std::cout << "  Port hole radius: " << port_hole_radius << " mm" << std::endl;
        std::cout << "  Port hole area: " << port_hole_area << " mm²" << std::endl;
        std::cout << "  Detector position: (" << detector.position.x << ", "
                  << detector.position.y << ", " << detector.position.z << ") mm" << std::endl;
        std::cout << "  Detector radius: " << detector.radius << " mm" << std::endl;
        std::cout << "  Inset depth: " << (sphere_geom.radius - detector.position.x) << " mm" << std::endl;

        SimConfig config;
        config.num_rays = 1'000'000;
        config.max_bounces = 500;

        // 随机数种子：0=固定（可重复），或设置为随机值
        config.random_seed = static_cast<unsigned int>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
        );
        std::cout << "  Random seed: " << config.random_seed << std::endl;

        std::cout << "=== Configuration Summary ===" << std::endl;
        std::cout << "  Sphere Radius: " << sphere_geom.radius << " mm" << std::endl;
        std::cout << "  Reflectance: " << sphere_geom.reflectance << std::endl;
        std::cout << "  Light Power: " << light.power << " W" << std::endl;
        std::cout << "  Detector Radius: " << detector.radius << " mm (area: "
                  << PI * detector.radius * detector.radius << " mm^2)" << std::endl;
        std::cout << "  Rays: " << config.num_rays << std::endl;
        std::cout << "===========================" << std::endl;

        // Calculate theoretical solution
        TheoryResult theory_result = TheoryCalculator::calculateWithPorts(
            sphere_geom.radius,
            sphere_geom.reflectance,
            light.power,
            port_hole_area
        );

        // Build mesh-based scene (triangle geometry)
        fs::path mesh_path = fs::path("E:/workspace/Optix_sphere/assets") / "integrating_sphere_0.3.obj";
        spdlog::info("\nBuilding Scene from Mesh (Triangle Geometry)");
        spdlog::info("Mesh path: {}", mesh_path.string());

        SimulationResult mesh_non_nee_result, mesh_nee_result;
        bool mesh_available = false;

        try {
            // Build mesh scene
            Scene mesh_scene(context);
            mesh_scene.build_scene(mesh_path.string(), sphere_geom);

            // Setup PathTracer for mesh scene
            PathTracer mesh_tracer(context, mesh_scene, embedded::g_forward_tracer_ptx, true);

            // Run Non-NEE on mesh
            spdlog::info("\n🔹 Running Mesh Non-NEE...");
            config.use_nee = false;
            auto start_time = std::chrono::high_resolution_clock::now();
            mesh_non_nee_result = mesh_tracer.launch(config, light, detector);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            spdlog::info("  ✅ Mesh Non-NEE took: {} ms", duration.count());

            // Run NEE on mesh
            spdlog::info("\n🔹 Running Mesh NEE...");
            config.use_nee = true;
            auto start_time2 = std::chrono::high_resolution_clock::now();
            mesh_nee_result = mesh_tracer.launch(config, light, detector);
            auto end_time2 = std::chrono::high_resolution_clock::now();
            auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - start_time2);
            spdlog::info("  ✅ Mesh NEE took: {} ms", duration2.count());

            mesh_available = true;

        } catch (const std::exception& e) {
            spdlog::warn("⚠️  Mesh test skipped: {}", e.what());
            spdlog::warn("    (Mesh file not found or mesh loading failed)");
        }

        // Print unified comparison
        std::cout << "\n=== SIMULATION RESULTS ===\n";
        std::cout << std::fixed << std::setprecision(6);

        std::cout << "\nTheory (Goebel formula):\n";
        std::cout << "  Irradiance:     " << theory_result.avg_irradiance << " W/mm²\n";
        std::cout << "  Detected flux:  " << theory_result.detected_flux << " W\n";

        if (mesh_available) {
            double mesh_non_nee_error = std::abs(mesh_non_nee_result.irradiance - theory_result.avg_irradiance)
                                      / theory_result.avg_irradiance * 100.0;
            double mesh_nee_error = std::abs(mesh_nee_result.irradiance - theory_result.avg_irradiance)
                                  / theory_result.avg_irradiance * 100.0;

            std::cout << std::setprecision(6);
            std::cout << "\nMesh Geometry (Non-NEE):\n";
            std::cout << "  Irradiance:     " << mesh_non_nee_result.irradiance << " W/mm²";
            std::cout << "  (error: " << std::setprecision(3) << mesh_non_nee_error << "%)\n";
            std::cout << std::setprecision(6);
            std::cout << "  Detected flux:  " << mesh_non_nee_result.detected_flux << " W\n";
            std::cout << "  Detected rays:  " << mesh_non_nee_result.detected_rays << " / " << config.num_rays;
            std::cout << "  (avg bounces: " << std::setprecision(2) << mesh_non_nee_result.avg_bounces << ")\n";

            std::cout << std::setprecision(6);
            std::cout << "\nMesh Geometry (NEE):\n";
            std::cout << "  Irradiance:     " << mesh_nee_result.irradiance << " W/mm²";
            std::cout << "  (error: " << std::setprecision(3) << mesh_nee_error << "%)\n";
            std::cout << std::setprecision(6);
            std::cout << "  Detected flux:  " << mesh_nee_result.detected_flux << " W\n";
            std::cout << "  Detected rays:  " << mesh_nee_result.detected_rays << " / " << config.num_rays;
            std::cout << "  (avg bounces: " << std::setprecision(2) << mesh_nee_result.avg_bounces << ")\n";
        }

        std::cout << "\n==========================\n" << std::endl;

    } catch (const std::exception& e) {
        spdlog::error("Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
