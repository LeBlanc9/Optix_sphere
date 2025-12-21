#include "optix_sbt_builder.h"
#include "device_params.h"
#include <spdlog/spdlog.h>

OptixSBTBuilder::OptixSBTBuilder() {
}

void OptixSBTBuilder::build_sbt(const OptixPipelineBuilder& pipeline_builder, const Scene& scene) {
    create_raygen_record(pipeline_builder.get_raygen_pg());
    create_miss_records(pipeline_builder.get_miss_pg(), pipeline_builder.get_miss_shadow_pg());
    create_hitgroup_records(
        pipeline_builder.get_sphere_hitgroup_pg(),
        pipeline_builder.get_detector_hitgroup_pg(),
        pipeline_builder.get_sphere_shadow_hitgroup_pg(),
        pipeline_builder.get_detector_shadow_hitgroup_pg(),
        scene
    );
    spdlog::info("✅ SBT created");
}

void OptixSBTBuilder::create_raygen_record(OptixProgramGroup raygen_pg) {
    char raygen_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &raygen_header));
    raygen_sbt_record_.upload(&raygen_header, sizeof(raygen_header));
    sbt_.raygenRecord = raygen_sbt_record_.get_cu_ptr();
}

void OptixSBTBuilder::create_miss_records(OptixProgramGroup miss_pg, OptixProgramGroup miss_shadow_pg) {
    // Miss records (2 ray types: radiance + shadow)
    size_t miss_record_size = OPTIX_SBT_RECORD_HEADER_SIZE;
    miss_sbt_record_.alloc(2 * miss_record_size);

    char miss_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &miss_header));
    CUDA_CHECK(cudaMemcpy(
        (void*)miss_sbt_record_.get_cu_ptr(),
        &miss_header,
        OPTIX_SBT_RECORD_HEADER_SIZE,
        cudaMemcpyHostToDevice
    ));

    char miss_shadow_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_shadow_pg, &miss_shadow_header));
    CUDA_CHECK(cudaMemcpy(
        (void*)(miss_sbt_record_.get_cu_ptr() + miss_record_size),
        &miss_shadow_header,
        OPTIX_SBT_RECORD_HEADER_SIZE,
        cudaMemcpyHostToDevice
    ));

    sbt_.missRecordBase = miss_sbt_record_.get_cu_ptr();
    sbt_.missRecordStrideInBytes = miss_record_size;
    sbt_.missRecordCount = 2;
}

void OptixSBTBuilder::create_hitgroup_records(
    OptixProgramGroup sphere_hitgroup_pg,
    OptixProgramGroup detector_hitgroup_pg,
    OptixProgramGroup sphere_shadow_hitgroup_pg,
    OptixProgramGroup detector_shadow_hitgroup_pg,
    const Scene& scene)
{
    // Hitgroup records (2 primitives × 2 ray types = 4 records)
    // SBT layout: [sphere_radiance, detector_radiance, sphere_shadow, detector_shadow]
    size_t sphere_record_size = OPTIX_SBT_RECORD_HEADER_SIZE + sizeof(SphereSbtData);
    size_t detector_record_size = OPTIX_SBT_RECORD_HEADER_SIZE + sizeof(DiskSbtData);
    size_t record_size = sphere_record_size > detector_record_size ? sphere_record_size : detector_record_size;
    size_t aligned_record_size = ((record_size + OPTIX_SBT_RECORD_ALIGNMENT - 1) / OPTIX_SBT_RECORD_ALIGNMENT) * OPTIX_SBT_RECORD_ALIGNMENT;

    // Allocate space for 4 records
    hitgroup_sbt_records_.alloc(4 * aligned_record_size);

    // Build radiance ray records
    // Sphere radiance (index 0: primitive 0, ray type 0)
    char sphere_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(sphere_hitgroup_pg, &sphere_header));
    CUdeviceptr sphere_record_ptr = hitgroup_sbt_records_.get_cu_ptr();
    CUDA_CHECK(cudaMemcpy((void*)sphere_record_ptr, &sphere_header, OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)(sphere_record_ptr + OPTIX_SBT_RECORD_HEADER_SIZE),
        (void*)scene.get_sphere_data_buffer().get_cu_ptr(), sizeof(SphereSbtData), cudaMemcpyDeviceToDevice));

    // Detector radiance (index 1: primitive 1, ray type 0)
    char detector_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(detector_hitgroup_pg, &detector_header));
    CUdeviceptr detector_record_ptr = hitgroup_sbt_records_.get_cu_ptr() + aligned_record_size;
    CUDA_CHECK(cudaMemcpy((void*)detector_record_ptr, &detector_header, OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)(detector_record_ptr + OPTIX_SBT_RECORD_HEADER_SIZE),
        (void*)scene.get_detector_data_buffer().get_cu_ptr(), sizeof(DiskSbtData), cudaMemcpyDeviceToDevice));

    // Build shadow ray records
    // Sphere shadow (index 2: primitive 0, ray type 1)
    char sphere_shadow_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(sphere_shadow_hitgroup_pg, &sphere_shadow_header));
    CUdeviceptr sphere_shadow_ptr = hitgroup_sbt_records_.get_cu_ptr() + 2 * aligned_record_size;
    CUDA_CHECK(cudaMemcpy((void*)sphere_shadow_ptr, &sphere_shadow_header, OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)(sphere_shadow_ptr + OPTIX_SBT_RECORD_HEADER_SIZE),
        (void*)scene.get_sphere_data_buffer().get_cu_ptr(), sizeof(SphereSbtData), cudaMemcpyDeviceToDevice));

    // Detector shadow (index 3: primitive 1, ray type 1)
    char detector_shadow_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(detector_shadow_hitgroup_pg, &detector_shadow_header));
    CUdeviceptr detector_shadow_ptr = hitgroup_sbt_records_.get_cu_ptr() + 3 * aligned_record_size;
    CUDA_CHECK(cudaMemcpy((void*)detector_shadow_ptr, &detector_shadow_header, OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)(detector_shadow_ptr + OPTIX_SBT_RECORD_HEADER_SIZE),
        (void*)scene.get_detector_data_buffer().get_cu_ptr(), sizeof(DiskSbtData), cudaMemcpyDeviceToDevice));

    // Point SBT to the records
    sbt_.hitgroupRecordBase = hitgroup_sbt_records_.get_cu_ptr();
    sbt_.hitgroupRecordStrideInBytes = aligned_record_size;
    sbt_.hitgroupRecordCount = 4;
}
