#pragma once

#include "core/optix_context.h"
#include "core/device_buffer.h"
#include "scene/scene.h"
#include "scene_types.h"
#include "simulation_result.h"

/**
 * @brief The core simulation engine.
 *
 * This class sets up the OptiX pipeline and shader binding table (SBT)
 * for a path tracing simulation. It takes a scene, runs the simulation,
 * and provides the results.
 */
class PathTracer {
public:
    PathTracer(const OptixContext& context, const Scene& scene, const std::string& ptx_path);
    ~PathTracer();

    // Disable copy/move
    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;

    /**
     * @brief Runs the simulation.
     * @param config High-level simulation settings.
     * @param light The light source for the simulation.
     * @param detector The detector to measure flux.
     * @return The simulation result.
     */
    SimulationResult launch(const SimConfig& config, const LightSource& light, const Detector& detector);

private:
    void create_module(const std::string& ptx_path);
    void create_program_groups();
    void create_pipeline();
    void create_sbt();

    const OptixContext& context_;
    const Scene& scene_;

    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    
    // Program groups
    OptixProgramGroup raygen_pg_ = nullptr;
    OptixProgramGroup miss_pg_ = nullptr;
    OptixProgramGroup sphere_hitgroup_pg_ = nullptr;
    OptixProgramGroup detector_hitgroup_pg_ = nullptr;

    // Shader Binding Table
    OptixShaderBindingTable sbt_ = {};
    DeviceBuffer raygen_sbt_record_;
    DeviceBuffer miss_sbt_record_;
    DeviceBuffer hitgroup_sbt_records_; // Now contains both sphere and detector records
};
