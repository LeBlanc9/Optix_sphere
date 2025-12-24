#pragma once

#include "simulation/optix_context.h"
#include "utils/device/device_buffer.cuh"
#include "scene/scene_types.h"
#include "simulation/device_params.h"
#include "geometry/mesh_loader.h"
#include "material.h"
#include <vector>
#include <memory>

/**
 * @brief Manages the scene geometry and acceleration structures.
 *
 * This class is responsible for loading triangle mesh scenes and building
 * the OptiX Geometry Acceleration Structure (GAS) required for tracing.
 * It uses a polymorphic material system to manage different surface types.
 */
class Scene {
public:
    Scene(const OptixContext& context);

    /**
     * Build scene from triangle mesh file
     * @param mesh_path Path to the OBJ mesh file
     * @param sphere_params Sphere parameters (center and reflectance for wall material)
     */
    void build_scene(const std::string& mesh_path, const Sphere& sphere_params);

    OptixTraversableHandle get_traversable() const { return traversable_; }

    /**
     * Get materials for SBT construction
     * Returns vector of materials in order corresponding to MaterialType enum
     */
    const std::vector<std::unique_ptr<Material>>& get_materials() const { return materials_; }

    /**
     * Get detector parameters extracted from mesh
     * These are used for NEE (Next Event Estimation)
     */
    float3 get_detector_position() const { return detector_position_; }
    float3 get_detector_normal() const { return detector_normal_; }
    float get_detector_radius() const { return detector_radius_; }
    float get_detector_total_area() const { return detector_total_area_; }

private:
    const OptixContext& context_;

    // Geometry Acceleration Structure
    DeviceBuffer gas_buffer_;
    OptixTraversableHandle traversable_ = 0;

    // Triangle mesh buffers
    DeviceBuffer vertex_buffer_;
    DeviceBuffer index_buffer_;

    // Material system - polymorphic materials for different surface types
    std::vector<std::unique_ptr<Material>> materials_;

    // Extracted detector parameters (from mesh) for NEE
    float3 detector_position_ = {0, 0, 0};
    float3 detector_normal_ = {0, 0, 0};
    float detector_radius_ = 0.0f;
    float detector_total_area_ = 0.0f;
};
