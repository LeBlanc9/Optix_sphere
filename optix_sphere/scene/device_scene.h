#pragma once

#include "simulation/optix_context.h"
#include "utils/device/device_buffer.cuh"
#include "simulation/device_params.h"
#include "geometry/obj_loader.h"
#include "material.h"
#include <vector>
#include <memory>
#include <map>

/**
 * @brief Manages the GPU scene geometry and acceleration structures
 */
class DeviceScene {
public:
    DeviceScene(const OptixContext& context);

    /**
     * Build GPU acceleration structures from mesh with material pool
     * @param mesh Pure geometry data with material names
     * @param material_pool Vector of material instances
     * @param material_mapping Map from mesh material names to pool indices
     */
    void build(
        const Mesh& mesh,
        const std::vector<std::shared_ptr<Material>>& material_pool,
        const std::map<std::string, size_t>& material_mapping
    );

    /**
     * Update materials from pool (fast path, no GAS rebuild)
     * @param material_pool Updated material pool
     */
    void update_from_pool(const std::vector<std::shared_ptr<Material>>& material_pool);

    OptixTraversableHandle get_traversable() const { return traversable_; }

    /**
     * Get materials for SBT construction
     */
    const std::vector<std::shared_ptr<Material>>& get_materials() const { return materials_; }

    /**
     * Get detector parameters for NEE
     */
    float3 get_detector_position() const { return detector_position_; }
    float3 get_detector_normal() const { return detector_normal_; }
    float get_detector_radius() const { return detector_radius_; }
    float get_detector_total_area() const { return detector_total_area_; }
    float get_detector_na() const { return detector_na_; }
    float get_detector_cos_max_angle() const { return detector_cos_max_angle_; }

    /**
     * Flip the detector normal direction
     */
    void flip_detector_normal() {
        detector_normal_.x = -detector_normal_.x;
        detector_normal_.y = -detector_normal_.y;
        detector_normal_.z = -detector_normal_.z;
    }

    /**
     * Get vertex and index buffers for triangle normal calculation
     */
    const float3* get_vertex_buffer_ptr() const {
        return reinterpret_cast<const float3*>(vertex_buffer_.get_cu_ptr());
    }
    const uint3* get_index_buffer_ptr() const {
        return reinterpret_cast<const uint3*>(index_buffer_.get_cu_ptr());
    }

private:
    const OptixContext& context_;

    // Geometry Acceleration Structure
    DeviceBuffer gas_buffer_;
    OptixTraversableHandle traversable_ = 0;

    // Triangle mesh buffers
    DeviceBuffer vertex_buffer_;
    DeviceBuffer index_buffer_;

    // Material system - references to material pool
    std::vector<std::shared_ptr<Material>> materials_;

    // Material mapping (mesh name -> pool index)
    std::map<std::string, size_t> material_mapping_;

    // Mesh material names (in order) - needed for update_from_pool
    std::vector<std::string> mesh_material_names_;

    // Extracted detector parameters for NEE
    float3 detector_position_ = {0, 0, 0};
    float3 detector_normal_ = {0, 0, 0};
    float detector_radius_ = 0.0f;
    float detector_total_area_ = 0.0f;
    float detector_na_ = 1.0f;           // Numerical Aperture (默认 1.0 = 无限制)
    float detector_cos_max_angle_ = 0.0f; // cos(θ_max) for angle check
};
