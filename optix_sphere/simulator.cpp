#include "simulator.h"
#include "simulation/optix_context.h"
#include "scene/device_scene.h"
#include "simulation/path_tracer.h"
#include "embedded_ptx.h"
#include <spdlog/spdlog.h>
#include <stdexcept>

class Simulator::Pimpl {
public:
    explicit Pimpl(int device_id) : context_(device_id) {}

    OptixContext context_;
    std::unique_ptr<DeviceScene> scene_;
    std::unique_ptr<PathTracer> tracer_;
    bool scene_is_built_ = false;

    void create_tracer() {
        if (!scene_) {
            throw std::runtime_error("Cannot create tracer without a scene.");
        }
        tracer_ = std::make_unique<PathTracer>(context_, *scene_, embedded::g_forward_tracer_ptx, true);
        scene_is_built_ = true;
        spdlog::info("✅ Path tracer created successfully.");
    }
};

Simulator::Simulator(int device_id) : pimpl_(std::make_unique<Pimpl>(device_id)) {
    spdlog::info("Simulator created on device {}. Ready to build a scene.", device_id);
}

Simulator::~Simulator() = default;

size_t Simulator::get_material_count() const {
    return material_pool_.size();
}

std::shared_ptr<Material> Simulator::get_material(const std::string& material_name) const {
    size_t index = get_material_index(material_name);
    return material_pool_[index];
}

void Simulator::clear_materials() {
    material_pool_.clear();
    material_name_to_index_.clear();
}

void Simulator::set_material(const std::string& material_name, std::shared_ptr<Material> material, bool sync_to_gpu) {
    size_t index = get_material_index(material_name); // This will throw if not found
    material_pool_[index] = material;
    if (sync_to_gpu) {
        this->sync_to_gpu();
    }
}

std::vector<std::string> Simulator::get_material_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : material_name_to_index_) {
        names.push_back(name);
    }
    return names;
}

void Simulator::build_scene(
    const Scene& scene,
    const std::map<std::string, size_t>& material_mapping,
    bool flip_detector_normal
) {
    // This private method is for low-level mapping. 
    // Usually called by the public build_scene(scene, materials_dict)
    spdlog::info("Building GPU scene from Scene with material mapping index...");

    if (material_pool_.empty()) {
        throw std::runtime_error("material_pool is empty. Please add materials before building scene.");
    }

    // Create DeviceScene
    pimpl_->scene_ = std::make_unique<DeviceScene>(pimpl_->context_);
    const Mesh& mesh = scene.get_mesh();

    // Verify mapping
    for (const auto& mat : mesh.materials) {
        if (material_mapping.find(mat.name) == material_mapping.end()) {
             throw std::runtime_error("Mesh material '" + mat.name + "' missing mapping");
        }
    }

    pimpl_->scene_->build(mesh, material_pool_, material_mapping);
    if (flip_detector_normal) pimpl_->scene_->flip_detector_normal();
    pimpl_->create_tracer();
}

void Simulator::build_scene(
    const Scene& scene,
    const std::map<std::string, std::shared_ptr<Material>>& materials,
    bool flip_detector_normal
) {
    spdlog::info("Building GPU scene with direct material mapping...");

    material_pool_.clear();
    material_name_to_index_.clear();
    std::map<std::string, size_t> material_mapping;

    const Mesh& mesh = scene.get_mesh();
    auto default_mats = material::get_default_materials();

    // 1. Process requested materials first to maintain user order
    for (const auto& [name, mat] : materials) {
        size_t idx = material_pool_.size();
        material_pool_.push_back(mat);
        material_mapping[name] = idx;
        material_name_to_index_[name] = idx;
        spdlog::info("Registered material '{}' -> pool[{}]", name, idx);
    }

    // 2. Validate against mesh requirement and fill missing ones
    for (const auto& mesh_mat : mesh.materials) {
        if (material_mapping.find(mesh_mat.name) == material_mapping.end()) {
            spdlog::warn("⚠️ Material '{}' used in mesh but not provided in mapping!", mesh_mat.name);
            
            std::shared_ptr<Material> fallback_mat;
            // Try to find a sensible default by name, otherwise use Absorber
            if (default_mats.count(mesh_mat.name)) {
                spdlog::info("   Using default factory for '{}'", mesh_mat.name);
                fallback_mat = default_mats[mesh_mat.name]();
            } else {
                spdlog::warn("   No default found for '{}', using black Absorber.", mesh_mat.name);
                fallback_mat = std::make_shared<AbsorberMaterial>();
            }

            size_t idx = material_pool_.size();
            material_pool_.push_back(fallback_mat);
            material_mapping[mesh_mat.name] = idx;
            material_name_to_index_[mesh_mat.name] = idx;
        }
    }

    // Call the base build implementation
    this->build_scene(scene, material_mapping, flip_detector_normal);
}

size_t Simulator::get_material_index(const std::string& material_name) const {
    auto it = material_name_to_index_.find(material_name);
    if (it == material_name_to_index_.end()) {
        throw std::runtime_error("Material '" + material_name + "' not found");
    }
    return it->second;
}

std::map<std::string, size_t> Simulator::get_material_indices() const {
    return material_name_to_index_;
}

void Simulator::sync_to_gpu() {
    if (!pimpl_->scene_is_built_ || !pimpl_->scene_) {
        throw std::runtime_error("Cannot sync materials before scene is built. Call 'build_scene' first.");
    }

    spdlog::info("🔄 Syncing materials to GPU (pool size: {})", material_pool_.size());

    // 同步 material_pool 到 DeviceScene
    pimpl_->scene_->update_from_pool(material_pool_);

    spdlog::info("✅ Materials synced successfully.");
}

SimulationResult Simulator::run(const phonder::PhotonBatch& source_batch) {
    if (!pimpl_->scene_is_built_ || !pimpl_->tracer_) {
        throw std::runtime_error("Simulation cannot be run before a scene is built. Call 'build_scene' first.");
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

SimulationResult Simulator::run(phonder::PhotonSource& procedural_source) {
    if (!pimpl_->scene_is_built_ || !pimpl_->tracer_) {
        throw std::runtime_error("Simulation cannot be run before a scene is built. Call 'build_scene' first.");
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

    return this->run(d_batch);
}

float Simulator::get_detector_total_area() const {
    if (!pimpl_->scene_is_built_ || !pimpl_->scene_) {
        throw std::runtime_error("Cannot get detector area before a scene is built.");
    }
    return pimpl_->scene_->get_detector_total_area();
}
