#pragma once

#include "simulation/optix_context.h"
#include "utils/device/device_buffer.cuh"
#include "scene/scene.h"
#include "scene/scene_types.h"
#include "simulation_result.h"
#include "optix_pipeline_builder.h"
#include "optix_sbt_builder.h"
#include "photon/sources.h"    // Data-only source definitions
#include "photon/launchers.h"  // C++ API for generate_photons_on_device
#include <memory>

/**
 * @brief Data-driven Monte Carlo path tracer
 *
 * Uses GPU-generated photon sources for maximum flexibility and performance.
 * Supports arbitrary light sources via the PhotonSource interface.
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
     * @brief Launch simulation with data-driven photon source
     * @param config Simulation configuration
     * @param photon_source Photon source generator (e.g., IsotropicPointSource, SpotSource)
     * @param detector Detector parameters
     * @return Simulation result
     */
    SimulationResult launch(
        const SimConfig& config,
        phonder::PhotonSource& photon_source);

private:
    void initialize(bool from_file, const std::string& ptx_path_or_code);

    const OptixContext& context_;
    const Scene& scene_;

    // 使用 builder 构建 pipeline 和 SBT
    std::unique_ptr<OptixPipelineBuilder> pipeline_builder_;
    std::unique_ptr<OptixSBTBuilder> sbt_builder_;
};
