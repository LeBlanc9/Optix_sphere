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
     */
    void build_scene(const std::string& mesh_path);

    /**
     * Build scene from triangle mesh file with custom material factories
     * @param mesh_path Path to the OBJ mesh file
     * @param material_factories Map of OBJ material names to MaterialFactory
     */
    void build_scene(
        const std::string& mesh_path,
        const std::map<std::string, MaterialFactory>& material_factories
    );

    OptixTraversableHandle get_traversable() const { return traversable_; }

    /**
     * Get materials for SBT construction
     * Returns vector of materials in order corresponding to MaterialType enum
     */
    const std::vector<std::shared_ptr<Material>>& get_materials() const { return materials_; }

    /**
     * Get detector parameters extracted from mesh
     * These are used for NEE (Next Event Estimation)
     */
    float3 get_detector_position() const { return detector_position_; }
    float3 get_detector_normal() const { return detector_normal_; }
    float get_detector_radius() const { return detector_radius_; }
    float get_detector_total_area() const { return detector_total_area_; }

    /**
     * Get vertex and index buffers for triangle normal calculation
     * These are used in closest-hit shaders to compute geometric normals
     */
    const float3* get_vertex_buffer_ptr() const {
        return reinterpret_cast<const float3*>(vertex_buffer_.get_cu_ptr());
    }
    const uint3* get_index_buffer_ptr() const {
        return reinterpret_cast<const uint3*>(index_buffer_.get_cu_ptr());
    }

    /**
     * Update a single material without rebuilding the scene geometry
     * Fast operation that only updates material parameters and SBT
     * @param name Material name to update
     * @param factory MaterialFactory function to create new material instance
     * @throws std::runtime_error if material name not found
     *
     * @example
     * ```cpp
     * // Update wall material with new reflectance
     * scene.update_material("wall_material", material::lambertian(0.98));
     * ```
     */
    void update_material(
        const std::string& name,
        const MaterialFactory& factory
    );

    /**
     * Get material names registered in the scene
     * @return Vector of material names
     */
    std::vector<std::string> get_material_names() const;

private:
    const OptixContext& context_;

    // Geometry Acceleration Structure
    DeviceBuffer gas_buffer_;
    OptixTraversableHandle traversable_ = 0;

    // Triangle mesh buffers
    DeviceBuffer vertex_buffer_;
    DeviceBuffer index_buffer_;

    // Material system - polymorphic materials for different surface types
    std::vector<std::shared_ptr<Material>> materials_;

    // Material name to index mapping (for fast material updates)
    std::map<std::string, size_t> material_name_to_index_;

    // Extracted detector parameters (from mesh) for NEE
    float3 detector_position_ = {0, 0, 0};
    float3 detector_normal_ = {0, 0, 0};
    float detector_radius_ = 0.0f;
    float detector_total_area_ = 0.0f;
};
