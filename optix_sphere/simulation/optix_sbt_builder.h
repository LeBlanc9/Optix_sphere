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
     * @param scene Scene（提供几何体数据和材质）
     */
    void build_sbt(const OptixPipelineBuilder& pipeline_builder, const Scene& scene);

    /**
     * @brief 获取构建好的 SBT
     */
    const OptixShaderBindingTable& get_sbt() const { return sbt_; }

private:
    void create_raygen_record(OptixProgramGroup raygen_pg);
    void create_miss_records(OptixProgramGroup miss_pg, OptixProgramGroup miss_shadow_pg);

    /**
     * @brief Create hitgroup records using polymorphic material system
     *
     * This method uses material polymorphism to dynamically:
     * 1. Query each material for its kernel name
     * 2. Look up the corresponding program group from PipelineBuilder
     * 3. Pack SBT headers and material-specific data
     *
     * No hardcoded program group arrays - fully data-driven!
     */
    void create_hitgroup_records(
        const OptixPipelineBuilder& pipeline_builder,
        const Scene& scene
    );

    OptixShaderBindingTable sbt_ = {};

    // Device buffers for SBT records
    DeviceBuffer raygen_sbt_record_;
    DeviceBuffer miss_sbt_record_;
    DeviceBuffer hitgroup_sbt_records_;
};
