#include "simulator.h"
#include "simulation/optix_context.h"
#include "scene/device_scene.h"
#include "simulation/path_tracer.h" // Now includes launch_from_batch
#include "embedded_ptx.h"
#include "geometry/mesh_loader.h" // For MeshLoader::get_default_material_configs
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

// =================================================================
// PIMPL (Pointer to Implementation) Idiom
// =================================================================
//
// This class holds the actual implementation details and expensive-to-include
// headers, keeping the public `simulator.h` header clean and lightweight.
class Simulator::Pimpl {
public:
    explicit Pimpl(int device_id) : context_(device_id) {}

    OptixContext context_;
    std::unique_ptr<DeviceScene> scene_;
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

Simulator::Simulator(int device_id) : pimpl_(std::make_unique<Pimpl>(device_id)) {
    spdlog::info("Simulator created on device {}. Ready to build a scene.", device_id);
}

Simulator::~Simulator() = default; // Default destructor is fine with unique_ptr<Pimpl>

void Simulator::build_scene(
    const Scene& scene,
    const std::map<std::string, MaterialFactory>& material_factories
) {
    spdlog::info("Building GPU scene from Scene...");

    // Create the DeviceScene object
    pimpl_->scene_ = std::make_unique<DeviceScene>(pimpl_->context_);

    // Get mesh data
    const Mesh& mesh = scene.get_mesh();

    // Validate that all materials have factories
    for (const auto& name : mesh.material_names) {
        auto it = material_factories.find(name);
        if (it == material_factories.end()) {
            throw std::runtime_error("Material factory not provided for material: " + name);
        }
    }

    // Build GPU scene with mesh and material factories
    pimpl_->scene_->build(mesh, material_factories);
    spdlog::info("✅ GPU scene built successfully.");

    // Create the path tracer
    pimpl_->create_tracer();
}

// Overload 1: Takes an existing PhotonBatch
SimulationResult Simulator::run(const phonder::PhotonBatch& source_batch, const SimConfig& config) {
    if (!pimpl_->scene_is_built_ || !pimpl_->tracer_) {
        throw std::runtime_error("Simulation cannot be run before a scene is built. Call 'build_scene_from_file' first.");
    }
    spdlog::info("🚀 Launching simulation from existing PhotonBatch...");
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
SimulationResult Simulator::run(phonder::PhotonSource& procedural_source, const SimConfig& config) {
    if (!pimpl_->scene_is_built_ || !pimpl_->tracer_) {
        throw std::runtime_error("Simulation cannot be run before a scene is built. Call 'build_scene_from_file' first.");
    }
    spdlog::info("🚀 Launching simulation from procedural PhotonSource...");
    spdlog::info("   Num rays (to generate): {}", config.num_rays);
    spdlog::info("   Max bounces: {}", config.max_bounces);
    spdlog::info("   Use NEE: {}", config.use_nee ? "Enabled" : "Disabled");

    phonder::PhotonBatch d_batch;
    procedural_source.generate(d_batch, config.num_rays, config.random_seed);

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

void Simulator::update_material(const std::string& name, const MaterialFactory& factory) {
    if (!pimpl_->scene_is_built_ || !pimpl_->scene_) {
        throw std::runtime_error("Cannot update material before a scene is built. Call 'build_scene_from_file' first.");
    }

    spdlog::info("🔄 Updating material '{}' (fast path, no geometry rebuild)", name);

    // Update material in the scene
    // SBT will be automatically rebuilt on next run() with the new material
    // Sphere center (if needed) is already captured in the factory closure
    pimpl_->scene_->update_material(name, factory);

    spdlog::info("✅ Material '{}' updated. Changes will take effect on next run()", name);
}
