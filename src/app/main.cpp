#include <iostream>
#include <optix_function_table_definition.h>
#include "core/optix_context.h"
#include "scene/scene.h"
#include "simulation/path_tracer.h"
#include "theory/theory.h"
#include "scene_types.h"

int main() {
    try {
        // 1. Initialize OptiX and CUDA
        OptixContext context;

        // 2. Define Scene and Simulation Parameters
        Sphere sphere_geom;
        sphere_geom.radius = 0.05f;
        sphere_geom.reflectance = 0.99f;
        sphere_geom.center = {0.0f, 0.0f, 0.0f};

        LightSource light;
        light.position = {0.0f, 0.0f, 0.0f};
        light.power = 1.0f;

        Detector detector;
        detector.position = {0.049f, 0.0f, 0.0f};
        detector.normal = {-1.0f, 0.0f, 0.0f};
        detector.radius = 0.000564f; // Area of 1 mm^2

        SimConfig config;
        config.num_rays = 2'000'000;
        config.max_bounces = 50;

        std::cout << "=== Configuration Summary ===" << std::endl;
        std::cout << "  Sphere Radius: " << sphere_geom.radius << " m" << std::endl;
        std::cout << "  Reflectance: " << sphere_geom.reflectance << std::endl;
        std::cout << "  Light Power: " << light.power << " W" << std::endl;
        std::cout << "  Detector Radius: " << detector.radius << " m" << std::endl;
        std::cout << "  Rays: " << config.num_rays << std::endl;
        std::cout << "===========================" << std::endl;


        // 3. Build Scene Geometry
        Scene scene(context);
        scene.build_ideal_sphere(sphere_geom);

        // 4. Setup and run the simulation
        PathTracer tracer(context, scene, "forward_tracer.ptx");
        SimulationResult sim_result = tracer.launch(config, light, detector);

        // 5. Calculate theoretical solution and compare
        TheoryResult theory_result = TheoryCalculator::calculateIdealSphere(
            sphere_geom.radius,
            sphere_geom.reflectance,
            light.power
        );
        
        TheoryCalculator::printComparison(sim_result, theory_result);

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
