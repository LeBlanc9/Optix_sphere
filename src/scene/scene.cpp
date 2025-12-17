#include "scene.h"
#include <iostream>

Scene::Scene(const OptixContext& context) : context_(context) {
    // Constructor can be empty for now
}

void Scene::build_ideal_sphere(const Sphere& sphere) {
    std::cout << "Building scene for an ideal sphere..." << std::endl;

    // 1. Create the Axis-Aligned Bounding Box (AABB) for our custom primitive
    OptixAabb aabb = {
        sphere.center.x - sphere.radius,
        sphere.center.y - sphere.radius,
        sphere.center.z - sphere.radius,
        sphere.center.x + sphere.radius,
        sphere.center.y + sphere.radius,
        sphere.center.z + sphere.radius
    };

    DeviceBuffer aabb_buffer;
    aabb_buffer.upload(&aabb, sizeof(OptixAabb));

    // 2. Define the build input for the custom primitive
    OptixBuildInput sphere_input = {};
    sphere_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    
    // Set AABB buffer
    CUdeviceptr d_aabb = aabb_buffer.get_cu_ptr();
    sphere_input.customPrimitiveArray.aabbBuffers = &d_aabb;
    sphere_input.customPrimitiveArray.numPrimitives = 1;

    // We have one SBT record for this one primitive
    unsigned int input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    sphere_input.customPrimitiveArray.flags = input_flags;
    sphere_input.customPrimitiveArray.numSbtRecords = 1;

    // 3. Compute memory usage for the Geometry Acceleration Structure (GAS)
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_.get(),
        &accel_options,
        &sphere_input,
        1, // number of build inputs
        &gas_buffer_sizes
    ));

    // 4. Allocate memory for the GAS
    DeviceBuffer temp_buffer(gas_buffer_sizes.tempSizeInBytes);
    gas_buffer_.alloc(gas_buffer_sizes.outputSizeInBytes);

    // 5. Build the GAS
    OPTIX_CHECK(optixAccelBuild(
        context_.get(),
        0, // CUDA stream
        &accel_options,
        &sphere_input,
        1, // num build inputs
        temp_buffer.get_cu_ptr(),
        gas_buffer_sizes.tempSizeInBytes,
        gas_buffer_.get_cu_ptr(),
        gas_buffer_sizes.outputSizeInBytes,
        &traversable_,
        nullptr, // emitted properties
        0
    ));

    // The temp buffer and AABB buffer can be freed now as they are owned by DeviceBuffer RAII objects
    
    // 6. Build the Shader Binding Table (SBT) record for the hitgroup
    SphereSbtData sbt_data;
    sbt_data.center = sphere.center;
    sbt_data.radius = sphere.radius;
    sbt_data.reflectance = sphere.reflectance;
    
    sphere_data_buffer_.upload(&sbt_data, sizeof(SphereSbtData));

    std::cout << "✅ Scene built successfully." << std::endl;
}
