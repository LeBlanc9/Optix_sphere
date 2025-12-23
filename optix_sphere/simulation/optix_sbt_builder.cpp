#include "optix_sbt_builder.h"
#include "simulation/device_params.h"
#include <spdlog/spdlog.h>
#include <vector>

OptixSBTBuilder::OptixSBTBuilder() {
}

void OptixSBTBuilder::build_sbt(const OptixPipelineBuilder& pipeline_builder, const Scene& scene) {
    create_raygen_record(pipeline_builder.get_program_group("__raygen__forward_trace"));
    create_miss_records(pipeline_builder.get_program_group("__miss__sphere"),
                       pipeline_builder.get_program_group("__miss__shadow"));

    // Create hitgroup records using polymorphic material system
    create_hitgroup_records(pipeline_builder, scene);

    spdlog::info("✅ SBT created successfully");
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
    const OptixPipelineBuilder& pipeline_builder,
    const Scene& scene)
{
    // Data-driven SBT construction using material polymorphism
    // SBT layout: [mat0_radiance, mat1_radiance, mat2_radiance, mat3_radiance,
    //              mat0_shadow, mat1_shadow, mat2_shadow, mat3_shadow]
    // Radiance rays use offset 0-3, shadow rays use offset 4-7 with stride 1

    const auto& materials = scene.get_materials();

    // Calculate maximum record size for all materials
    size_t max_record_size = OPTIX_SBT_RECORD_HEADER_SIZE;
    for (const auto& material : materials) {
        if (material) {
            size_t mat_record_size = OPTIX_SBT_RECORD_HEADER_SIZE + material->get_sbt_data_size();
            if (mat_record_size > max_record_size) {
                max_record_size = mat_record_size;
            }
        }
    }

    // Align record size
    size_t aligned_record_size = ((max_record_size + OPTIX_SBT_RECORD_ALIGNMENT - 1) /
                                  OPTIX_SBT_RECORD_ALIGNMENT) * OPTIX_SBT_RECORD_ALIGNMENT;

    // Allocate space for 8 records (4 radiance + 4 shadow)
    hitgroup_sbt_records_.alloc(8 * aligned_record_size);

    // Create radiance ray records (indices 0-3)
    std::vector<char> host_sbt_data(aligned_record_size);

    for (size_t i = 0; i < materials.size(); ++i) {
        if (!materials[i]) continue;

        // Query material for its kernel name (polymorphism!)
        std::string kernel_name = materials[i]->get_kernel_name();

        // Look up the program group from PipelineBuilder (data-driven!)
        OptixProgramGroup radiance_pg = pipeline_builder.get_program_group(kernel_name);
        if (!radiance_pg) {
            spdlog::error("Failed to find program group for kernel: {}", kernel_name);
            continue;
        }

        // Clear host buffer
        std::memset(host_sbt_data.data(), 0, aligned_record_size);

        // Pack OptiX header
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        OPTIX_CHECK(optixSbtRecordPackHeader(radiance_pg, &header));
        std::memcpy(host_sbt_data.data(), header, OPTIX_SBT_RECORD_HEADER_SIZE);

        // Write material-specific SBT data using polymorphism
        materials[i]->write_sbt_data(host_sbt_data.data() + OPTIX_SBT_RECORD_HEADER_SIZE);

        // Upload to GPU
        CUdeviceptr record_ptr = hitgroup_sbt_records_.get_cu_ptr() + i * aligned_record_size;
        CUDA_CHECK(cudaMemcpy(
            (void*)record_ptr,
            host_sbt_data.data(),
            aligned_record_size,
            cudaMemcpyHostToDevice
        ));
    }

    // Create shadow ray records (indices 4-7)
    // Shadow records only need headers (no material data)
    for (size_t i = 0; i < materials.size(); ++i) {
        if (!materials[i]) continue;

        // Query material for its shadow kernel name (polymorphism!)
        std::string shadow_kernel_name = materials[i]->get_shadow_kernel_name();

        // Look up the shadow program group from PipelineBuilder (data-driven!)
        OptixProgramGroup shadow_pg = pipeline_builder.get_program_group(shadow_kernel_name);
        if (!shadow_pg) {
            spdlog::error("Failed to find shadow program group for kernel: {}", shadow_kernel_name);
            continue;
        }

        char shadow_header[OPTIX_SBT_RECORD_HEADER_SIZE];
        OPTIX_CHECK(optixSbtRecordPackHeader(shadow_pg, &shadow_header));

        CUdeviceptr shadow_ptr = hitgroup_sbt_records_.get_cu_ptr() + (4 + i) * aligned_record_size;
        CUDA_CHECK(cudaMemcpy(
            (void*)shadow_ptr,
            &shadow_header,
            OPTIX_SBT_RECORD_HEADER_SIZE,
            cudaMemcpyHostToDevice
        ));
    }

    // Point SBT to the records
    sbt_.hitgroupRecordBase = hitgroup_sbt_records_.get_cu_ptr();
    sbt_.hitgroupRecordStrideInBytes = aligned_record_size;
    sbt_.hitgroupRecordCount = 8;  // 4 radiance + 4 shadow
}
