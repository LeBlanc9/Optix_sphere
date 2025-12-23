#pragma once

#include "core/optix_context.h"
#include "core/device_buffer.h"
#include "scene/scene.h"
#include "scene/scene_types.h"
#include "simulation_result.h"
#include "optix_pipeline_builder.h"
#include "optix_sbt_builder.h"
#include <memory>

/**
 * @brief Monte Carlo 路径追踪器
 *
 * 负责启动和管理 OptiX 光线追踪模拟
 * 使用 OptixPipelineBuilder 和 OptixSBTBuilder 构建 OptiX 组件
 */
class PathTracer {
public:
    // 从 PTX 文件构造
    PathTracer(const OptixContext& context, const Scene& scene, const std::string& ptx_path);

    // 从嵌入的 PTX 字符串构造
    PathTracer(const OptixContext& context, const Scene& scene, const char* ptx_code, bool is_embedded);

    ~PathTracer();

    // 禁止拷贝和移动
    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;

    /**
     * @brief 启动模拟
     * @param config 模拟配置
     * @param light 光源
     * @param detector 探测器
     * @return 模拟结果
     */
    SimulationResult launch(const SimConfig& config, const LightSource& light, const Detector& detector);

private:
    void initialize(bool from_file, const std::string& ptx_path_or_code);

    const OptixContext& context_;
    const Scene& scene_;

    // 使用 builder 构建 pipeline 和 SBT
    std::unique_ptr<OptixPipelineBuilder> pipeline_builder_;
    std::unique_ptr<OptixSBTBuilder> sbt_builder_;
};
