#pragma once

#include "core/optix_context.h"
#include <optix.h>
#include <string>

/**
 * @brief 负责构建 OptiX Pipeline 和 Program Groups
 *
 * 将 module 创建、program group 创建、pipeline 创建等逻辑封装
 */
class OptixPipelineBuilder {
public:
    OptixPipelineBuilder(const OptixContext& context);
    ~OptixPipelineBuilder();

    // 禁止拷贝
    OptixPipelineBuilder(const OptixPipelineBuilder&) = delete;
    OptixPipelineBuilder& operator=(const OptixPipelineBuilder&) = delete;

    /**
     * @brief 从 PTX 文件创建 module
     */
    void create_module_from_file(const std::string& ptx_path);

    /**
     * @brief 从嵌入的 PTX 字符串创建 module
     */
    void create_module_from_string(const std::string& ptx_code);

    /**
     * @brief 创建所有 program groups（raygen, miss, hitgroups）
     */
    void create_program_groups();

    /**
     * @brief 创建 pipeline
     */
    void create_pipeline();

    // Getters
    OptixModule get_module() const { return module_; }
    OptixPipeline get_pipeline() const { return pipeline_; }

    OptixProgramGroup get_raygen_pg() const { return raygen_pg_; }
    OptixProgramGroup get_miss_pg() const { return miss_pg_; }
    OptixProgramGroup get_miss_shadow_pg() const { return miss_shadow_pg_; }

    OptixProgramGroup get_sphere_hitgroup_pg() const { return sphere_hitgroup_pg_; }
    OptixProgramGroup get_detector_hitgroup_pg() const { return detector_hitgroup_pg_; }
    OptixProgramGroup get_sphere_shadow_hitgroup_pg() const { return sphere_shadow_hitgroup_pg_; }
    OptixProgramGroup get_detector_shadow_hitgroup_pg() const { return detector_shadow_hitgroup_pg_; }

private:
    const OptixContext& context_;

    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options_ = {};

    // Program groups for radiance rays (ray type 0)
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup sphere_hitgroup_pg_ = nullptr;
    OptixProgramGroup detector_hitgroup_pg_ = nullptr;

    // Program groups for shadow rays (ray type 1)
    OptixProgramGroup miss_shadow_pg_ = nullptr;
    OptixProgramGroup sphere_shadow_hitgroup_pg_ = nullptr;
    OptixProgramGroup detector_shadow_hitgroup_pg_ = nullptr;
};
