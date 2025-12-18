#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <optix_function_table_definition.h>
#include "core/optix_context.h"
#include "scene/scene.h"
#include "simulation/path_tracer.h"
#include "theory/theory.h"
#include "scene_types.h"
#include "constants.h"

int main() {
    try {
        // 1. Initialize OptiX and CUDA
        OptixContext context;

        // 2. Define Scene and Simulation Parameters (单位: mm)
        Sphere sphere_geom;
        sphere_geom.radius = 50.0f;              // 50 mm radius
        sphere_geom.reflectance = 0.98f;
        sphere_geom.center = {0.0f, 0.0f, 0.0f};

        LightSource light;
        light.position = {0.0f, 0.0f, 0.0f};     // center of sphere
        light.power = 1.0f;                       // 1 W

        // 配置探测器 - 使用弦面几何（在球面上开孔）
        Detector detector;
        float port_hole_radius = 10.0f;  // 开孔半径 (mm)
        configure_detector_chord(detector, sphere_geom, port_hole_radius);

        float port_hole_area = PI * port_hole_radius * port_hole_radius;  // 计算面积供理论使用
        std::cout << "  Port hole radius: " << port_hole_radius << " mm" << std::endl;
        std::cout << "  Port hole area: " << port_hole_area << " mm²" << std::endl;
        std::cout << "  Detector position: (" << detector.position.x << ", "
                  << detector.position.y << ", " << detector.position.z << ") mm" << std::endl;
        std::cout << "  Detector radius: " << detector.radius << " mm" << std::endl;
        std::cout << "  Inset depth: " << (sphere_geom.radius - detector.position.x) << " mm" << std::endl;

        SimConfig config;
        config.num_rays = 5'000'000;             // 500万光线 - 平衡精度与速度
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

        // 3. Build Scene Geometry (sphere + detector)
        Scene scene(context);
        scene.build_scene(sphere_geom, detector);

        // 4. Setup PathTracer
        PathTracer tracer(context, scene, "forward_tracer.ptx");

        // 5. Run Non-NEE simulation
        std::cout << "\n🔹 Running Non-NEE (Standard Path Tracing)..." << std::endl;
        config.use_nee = false;
        auto start_time = std::chrono::high_resolution_clock::now();
        SimulationResult non_nee_result = tracer.launch(config, light, detector);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "  ✅ Non-NEE took: " << duration.count() << " ms" << std::endl;

        // 6. Run NEE simulation
        std::cout << "\n🔹 Running NEE (Variance Reduction)..." << std::endl;
        config.use_nee = true;
        start_time = std::chrono::high_resolution_clock::now();
        SimulationResult nee_result = tracer.launch(config, light, detector);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "  ✅ NEE took: " << duration.count() << " ms" << std::endl;

        // 7. Calculate theoretical solution
        // 注意：理论计算应使用实际开孔面积，而不是探测器圆盘面积
        // （探测器圆盘有1.2x安全系数）
        TheoryResult theory_result = TheoryCalculator::calculateWithPorts(
            sphere_geom.radius,
            sphere_geom.reflectance,
            light.power,
            port_hole_area  // 使用实际开孔面积
        );

        // 8. Print three-way comparison
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              Three-Way Comparison: Non-NEE vs NEE vs Theory       ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════════════╣\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "║ Metric                │ Non-NEE      │ NEE          │ Theory       ║\n";
        std::cout << "╟───────────────────────┼──────────────┼──────────────┼──────────────╢\n";
        std::cout << "║ Irradiance (W/mm²)    │ " << std::setw(12) << non_nee_result.irradiance
                  << " │ " << std::setw(12) << nee_result.irradiance
                  << " │ " << std::setw(12) << theory_result.avg_irradiance << " ║\n";
        std::cout << "║ Detected flux (W)     │ " << std::setw(12) << non_nee_result.detected_flux
                  << " │ " << std::setw(12) << nee_result.detected_flux
                  << " │ " << std::setw(12) << theory_result.detected_flux << " ║\n";
        std::cout << "╟───────────────────────┴──────────────┴──────────────┴──────────────╢\n";

        double non_nee_error = std::abs(non_nee_result.irradiance - theory_result.avg_irradiance)
                             / theory_result.avg_irradiance * 100.0;
        double nee_error = std::abs(nee_result.irradiance - theory_result.avg_irradiance)
                         / theory_result.avg_irradiance * 100.0;

        std::cout << std::setprecision(3);
        std::cout << "║ Relative Error (%)    │ " << std::setw(12) << non_nee_error
                  << " │ " << std::setw(12) << nee_error << " │ " << std::setw(12) << 0.0 << " ║\n";
        std::cout << "╟───────────────────────────────────────────────────────────────────╢\n";
        std::cout << "║ Statistics:                                                        ║\n";
        std::cout << "║   Non-NEE detected rays: " << std::setw(10) << non_nee_result.detected_rays
                  << "   Avg bounces: " << std::setprecision(2) << std::setw(6) << non_nee_result.avg_bounces << "       ║\n";
        std::cout << "║   NEE detected rays:     " << std::setw(10) << nee_result.detected_rays
                  << "   Avg bounces: " << std::setprecision(2) << std::setw(6) << nee_result.avg_bounces << "       ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════════╝\n";
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
