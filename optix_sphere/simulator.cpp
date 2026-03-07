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

size_t Simulator::add_material(std::shared_ptr<Material> material) {
    material_pool_.push_back(material);
    return material_pool_.size() - 1;
}

void Simulator::set_material(size_t index, std::shared_ptr<Material> material) {
    if (index >= material_pool_.size()) {
        throw std::runtime_error(
            "Material index " + std::to_string(index) +
            " out of range (pool size: " + std::to_string(material_pool_.size()) + ")"
        );
    }
    material_pool_[index] = material;
}

size_t Simulator::get_material_pool_size() const {
    return material_pool_.size();
}

std::shared_ptr<Material> Simulator::get_material(size_t index) const {
    if (index >= material_pool_.size()) {
        throw std::runtime_error(
            "Material index " + std::to_string(index) +
            " out of range (pool size: " + std::to_string(material_pool_.size()) + ")"
        );
    }
    return material_pool_[index];
}

void Simulator::clear_materials() {
    material_pool_.clear();
}

void Simulator::build_scene(
    const Scene& scene,
    const std::map<std::string, size_t>& material_mapping,
    bool flip_detector_normal
) {
    spdlog::info("Building GPU scene from Scene with material pool...");

    // 验证 material_pool 不为空
    if (material_pool_.empty()) {
        throw std::runtime_error("material_pool is empty. Please add materials before building scene.");
    }

    // 验证 material_mapping 中的所有索引都有效
    for (const auto& [name, idx] : material_mapping) {
        if (idx >= material_pool_.size()) {
            throw std::runtime_error(
                "Material mapping for '" + name + "' points to invalid pool index " +
                std::to_string(idx) + " (pool size: " + std::to_string(material_pool_.size()) + ")"
            );
        }
    }

    // Create DeviceScene
    pimpl_->scene_ = std::make_unique<DeviceScene>(pimpl_->context_);

    // Get mesh data
    const Mesh& mesh = scene.get_mesh();

    // 验证 mesh 中的所有材质名称都有映射
    for (const auto& mat_name : mesh.material_names) {
        if (material_mapping.find(mat_name) == material_mapping.end()) {
            throw std::runtime_error(
                "Mesh material '" + mat_name + "' not found in material_mapping"
            );
        }
    }

    // Build GPU scene with material pool and mapping
    pimpl_->scene_->build(mesh, material_pool_, material_mapping);

    // Flip detector normal if requested
    if (flip_detector_normal) {
        spdlog::info("Flipping detector normal direction");
        pimpl_->scene_->flip_detector_normal();
    }

    spdlog::info("✅ GPU scene built successfully.");

    // Create the path tracer
    pimpl_->create_tracer();
}

void Simulator::update_materials() {
    if (!pimpl_->scene_is_built_ || !pimpl_->scene_) {
        throw std::runtime_error("Cannot update materials before scene is built. Call 'build_scene' first.");
    }

    spdlog::info("🔄 Updating materials from pool (size: {})", material_pool_.size());

    // 同步 material_pool 到 DeviceScene
    pimpl_->scene_->update_from_pool(material_pool_);

    spdlog::info("✅ Materials updated successfully.");
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
