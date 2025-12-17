#pragma once

#include "core/optix_context.h"
#include "core/device_buffer.h"
#include "scene_types.h"
#include "device_params.h"

/**
 * @brief Manages the scene geometry and acceleration structures.
 *
 * This class is responsible for taking high-level scene descriptions
 * (like Sphere objects) and building the OptiX Geometry Acceleration
 * Structure (GAS) required for tracing. It also prepares the Shader
 * Binding Table (SBT) records for the geometry it manages.
 */
class Scene {
public:
    Scene(const OptixContext& context);

    // For now, we build a simple, hardcoded ideal sphere scene.
    // In the future, this could take a list of objects.
    void build_ideal_sphere(const Sphere& sphere);

    OptixTraversableHandle get_traversable() const { return traversable_; }
    
    const DeviceBuffer& get_sphere_data_buffer() const { return sphere_data_buffer_; }

private:
    const OptixContext& context_;
    
    // Geometry Acceleration Structure
    DeviceBuffer gas_buffer_;
    OptixTraversableHandle traversable_ = 0;

    // Shader Binding Table records for the geometry in this scene
    DeviceBuffer sphere_data_buffer_;
};
