#pragma once

#include "core/device_buffer.h"
#include "scene/scene.h"
#include "optix_pipeline_builder.h"
#include <optix.h>

/**
 * @brief 负责构建 Shader Binding Table (SBT)
 *
 * SBT 定义了不同光线类型和几何体的程序绑定关系
 */
class OptixSBTBuilder {
public:
    OptixSBTBuilder();
    ~OptixSBTBuilder() = default;

    /**
     * @brief 构建完整的 SBT
     * @param pipeline_builder Pipeline builder（提供 program groups）
     * @param scene Scene（提供几何体数据）
     */
    void build_sbt(const OptixPipelineBuilder& pipeline_builder, const Scene& scene);

    /**
     * @brief 获取构建好的 SBT
     */
    const OptixShaderBindingTable& get_sbt() const { return sbt_; }

private:
    void create_raygen_record(OptixProgramGroup raygen_pg);
    void create_miss_records(OptixProgramGroup miss_pg, OptixProgramGroup miss_shadow_pg);
    void create_hitgroup_records(
        OptixProgramGroup sphere_hitgroup_pg,
        OptixProgramGroup detector_hitgroup_pg,
        OptixProgramGroup sphere_shadow_hitgroup_pg,
        OptixProgramGroup detector_shadow_hitgroup_pg,
        const Scene& scene
    );

    OptixShaderBindingTable sbt_ = {};

    // Device buffers for SBT records
    DeviceBuffer raygen_sbt_record_;
    DeviceBuffer miss_sbt_record_;
    DeviceBuffer hitgroup_sbt_records_;
};
