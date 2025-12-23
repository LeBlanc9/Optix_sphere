#pragma once

#include "core/optix_context.h"
#include <optix.h>
#include <string>
#include <unordered_map>

/**
 * @brief 负责构建 OptiX Pipeline 和 Program Groups
 *
 * 将 module 创建、program group 创建、pipeline 创建等逻辑封装
 * 使用 kernel 名称作为键来动态查找 program groups
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

    /**
     * @brief Get program group by kernel name (modern, unified interface)
     * @param kernel_name The OptiX kernel function name (e.g., "__closesthit__lambertian")
     * @return The corresponding program group, or nullptr if not found
     */
    OptixProgramGroup get_program_group(const std::string& kernel_name) const;

private:
    const OptixContext& context_;

    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options_ = {};

    // Unified program group storage (kernel name -> program group)
    std::unordered_map<std::string, OptixProgramGroup> program_groups_;
};
