#include "scene.h"
#include <iostream>

Scene::Scene(const OptixContext& context) : context_(context) {
    // Constructor can be empty for now
}

void Scene::build_scene(const Sphere& sphere, const Detector& detector) {
    std::cout << "Building scene with sphere and detector..." << std::endl;

    // 1. Create AABBs for both sphere and detector
    OptixAabb aabbs[2];

    // Sphere AABB
    aabbs[0].minX = sphere.center.x - sphere.radius;
    aabbs[0].minY = sphere.center.y - sphere.radius;
    aabbs[0].minZ = sphere.center.z - sphere.radius;
    aabbs[0].maxX = sphere.center.x + sphere.radius;
    aabbs[0].maxY = sphere.center.y + sphere.radius;
    aabbs[0].maxZ = sphere.center.z + sphere.radius;

    // Detector (disk) AABB - conservative bounding box
    aabbs[1].minX = detector.position.x - detector.radius;
    aabbs[1].minY = detector.position.y - detector.radius;
    aabbs[1].minZ = detector.position.z - detector.radius;
    aabbs[1].maxX = detector.position.x + detector.radius;
    aabbs[1].maxY = detector.position.y + detector.radius;
    aabbs[1].maxZ = detector.position.z + detector.radius;

    DeviceBuffer aabb_buffer;
    aabb_buffer.upload(aabbs, 2 * sizeof(OptixAabb));

    // Create SBT index offset buffer: tells which primitive uses which SBT record
    // primitive 0 (sphere) -> SBT record 0
    // primitive 1 (detector) -> SBT record 1
    unsigned int sbt_indices[2] = { 0, 1 };
    DeviceBuffer sbt_index_buffer;
    sbt_index_buffer.upload(sbt_indices, 2 * sizeof(unsigned int));

    // 2. Define build input for custom primitives
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    CUdeviceptr d_aabb = aabb_buffer.get_cu_ptr();
    build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
    build_input.customPrimitiveArray.numPrimitives = 2; // sphere + detector

    // SBT index offset buffer
    build_input.customPrimitiveArray.sbtIndexOffsetBuffer = sbt_index_buffer.get_cu_ptr();
    build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(unsigned int);
    build_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(unsigned int);

    // We have two SBT records: one for sphere, one for detector
    unsigned int input_flags[2] = { OPTIX_GEOMETRY_FLAG_NONE, OPTIX_GEOMETRY_FLAG_NONE };
    build_input.customPrimitiveArray.flags = input_flags;
    build_input.customPrimitiveArray.numSbtRecords = 2;

    // 3. Compute memory usage for the GAS
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_.get(),
        &accel_options,
        &build_input,
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
        &build_input,
        1, // num build inputs
        temp_buffer.get_cu_ptr(),
        gas_buffer_sizes.tempSizeInBytes,
        gas_buffer_.get_cu_ptr(),
        gas_buffer_sizes.outputSizeInBytes,
        &traversable_,
        nullptr, // emitted properties
        0
    ));

    // 6. Build SBT records for sphere and detector
    SphereSbtData sphere_sbt;
    sphere_sbt.center = sphere.center;
    sphere_sbt.radius = sphere.radius;
    sphere_sbt.reflectance = sphere.reflectance;
    sphere_data_buffer_.upload(&sphere_sbt, sizeof(SphereSbtData));

    DiskSbtData detector_sbt;
    detector_sbt.center = detector.position;
    detector_sbt.normal = detector.normal;
    detector_sbt.radius = detector.radius;
    detector_data_buffer_.upload(&detector_sbt, sizeof(DiskSbtData));

    std::cout << "✅ Scene built successfully (sphere + detector)." << std::endl;
}
