#pragma once

#include "simulation/optix_context.h"
#include "utils/device/device_buffer.cuh"
#include "scene/device_scene.h"
#include "simulation_result.h"
#include "optix_pipeline_builder.h"
#include "optix_sbt_builder.h"
#include "photon/sources.h"    // Data-only source definitions
#include "photon/photon_batch.h"      // For PhotonBatch
#include <memory>

// Forward declaration from simulator.h
struct SimConfig;

/**
 * @brief Data-driven Monte Carlo path tracer
 *
 * Uses GPU-generated photon sources for maximum flexibility and performance.
 * Supports arbitrary light sources via the PhotonSource interface.
 */
class PathTracer {
public:
    // 从 PTX 文件构造
    PathTracer(const OptixContext& context, const DeviceScene& scene, const std::string& ptx_path);

    // 从嵌入的 PTX 字符串构造
    PathTracer(const OptixContext& context, const DeviceScene& scene, const char* ptx_code, bool is_embedded);

    ~PathTracer();

    // 禁止拷贝和移动
    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;

    /**
     * @brief Launch simulation with a pre-existing batch of photons on the GPU.
     * @param config Simulation configuration
     * @param input_batch A batch of photons already on the GPU (position, direction, weight).
     * @return Simulation result
     */
    SimulationResult launch_from_batch(
        const SimConfig& config,
        const phonder::PhotonBatch& input_batch);

private:
    void initialize(bool from_file, const std::string& ptx_path_or_code);

    const OptixContext& context_;
    const DeviceScene& scene_;

    // 使用 builder 构建 pipeline 和 SBT
    std::unique_ptr<OptixPipelineBuilder> pipeline_builder_;
    std::unique_ptr<OptixSBTBuilder> sbt_builder_;
};
