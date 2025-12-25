#include "simulator.h"
#include "simulation/optix_context.h"
#include "scene/scene.h"
#include "simulation/path_tracer.h" // Now includes launch_from_batch
#include "embedded_ptx.h"
#include "photon/launchers.h" // For phonder::generate_photons_on_device
#include "photon/batch.cuh" // Add this to define DevicePhotonBatch
#include <spdlog/spdlog.h>
#include <stdexcept>

// =================================================================
// PIMPL (Pointer to Implementation) Idiom
// =================================================================
//
// This class holds the actual implementation details and expensive-to-include
// headers, keeping the public `simulator.h` header clean and lightweight.
class Simulator::Pimpl {
public:
    OptixContext context_;
    std::unique_ptr<Scene> scene_;
    std::unique_ptr<PathTracer> tracer_;

    // A flag to check if a scene has been successfully built.
    bool scene_is_built_ = false;

    // A helper to create the path tracer once a scene is available.
    void create_tracer() {
        if (!scene_) {
            throw std::runtime_error("Cannot create tracer without a scene.");
        }
        // The last parameter `true` enables NEE related program groups.
        tracer_ = std::make_unique<PathTracer>(context_, *scene_, embedded::g_forward_tracer_ptx, true);
        scene_is_built_ = true;
        spdlog::info("✅ Path tracer created successfully.");
    }
};


// =================================================================
// Simulator Public API Implementation
// =================================================================

Simulator::Simulator() : pimpl_(std::make_unique<Pimpl>()) {
    spdlog::info("Simulator created. Ready to build a scene.");
}

Simulator::~Simulator() = default; // Default destructor is fine with unique_ptr<Pimpl>

void Simulator::build_scene_from_file(const std::string& file_path, const MeshSceneConfig& config) {
    spdlog::info("Attempting to build scene from file: {}", file_path);
    
    // Create the scene object.
    pimpl_->scene_ = std::make_unique<Scene>(pimpl_->context_);

    // In this simplified API, we create a temporary Sphere object to pass
    // the reflectance, as the underlying `build_scene` method requires it.
    Sphere sphere_geom;
    sphere_geom.reflectance = config.default_reflectance;
    // The radius and center will be ignored by this `build_scene` overload,
    // but we set them for completeness.
    sphere_geom.radius = -1.0f; // Indicates it's not an ideal sphere
    sphere_geom.center = {0.0f, 0.0f, 0.0f};

    // Build the scene from the mesh file.
    pimpl_->scene_->build_scene(file_path, sphere_geom);
    spdlog::info("✅ Scene built successfully from file.");

    // Now that the scene exists, create the path tracer.
    pimpl_->create_tracer();
}

// Overload 1: Takes an existing DevicePhotonBatch
SimulationResult Simulator::run(const phonder::DevicePhotonBatch& source_batch, const SimConfig& config) {
    if (!pimpl_->scene_is_built_ || !pimpl_->tracer_) {
        throw std::runtime_error("Simulation cannot be run before a scene is built. Call 'build_scene_from_file' first.");
    }
    spdlog::info("🚀 Launching simulation from existing DevicePhotonBatch...");
    spdlog::info("   Num rays in batch: {}", source_batch.size());
    spdlog::info("   Max bounces: {}", config.max_bounces);
    spdlog::info("   Use NEE: {}", config.use_nee ? "Enabled" : "Disabled");

    SimulationResult result = pimpl_->tracer_->launch_from_batch(config, source_batch);

    spdlog::info("✅ Simulation complete.");
    spdlog::info("   Irradiance: {} W/mm²", result.irradiance);
    spdlog::info("   Detected flux: {} W", result.detected_flux);
    return result;
}

// Overload 2: Takes a procedural PhotonSource (generates batch internally)
SimulationResult Simulator::run(const phonder::PhotonSource& procedural_source, const SimConfig& config) {
    if (!pimpl_->scene_is_built_ || !pimpl_->tracer_) {
        throw std::runtime_error("Simulation cannot be run before a scene is built. Call 'build_scene_from_file' first.");
    }
    spdlog::info("🚀 Launching simulation from procedural PhotonSource...");
    spdlog::info("   Num rays (to generate): {}", config.num_rays);
    spdlog::info("   Max bounces: {}", config.max_bounces);
    spdlog::info("   Use NEE: {}", config.use_nee ? "Enabled" : "Disabled");

    phonder::DevicePhotonBatch d_batch;
    phonder::generate_photons_on_device(procedural_source, d_batch, config.num_rays, config.random_seed);

    if (d_batch.empty()) {
        spdlog::warn("Photon generation resulted in 0 photons. Aborting procedural launch.");
        return {};
    }
    spdlog::info("   Generated {} photons on GPU from procedural source.", d_batch.size());

    // Now call the other run overload with the generated batch
    return run(d_batch, config);
}


float Simulator::get_detector_total_area() const {
    if (!pimpl_->scene_is_built_ || !pimpl_->scene_) {
        throw std::runtime_error("Cannot get detector area before a scene is built.");
    }
    return pimpl_->scene_->get_detector_total_area();
}