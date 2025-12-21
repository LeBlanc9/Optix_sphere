#include "path_tracer.h"
#include "device_params.h"
#include "constants.h"
#include <spdlog/spdlog.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

PathTracer::PathTracer(const OptixContext& context, const Scene& scene, const std::string& ptx_path)
    : context_(context), scene_(scene)
{
    try {
        initialize(true, ptx_path);
    } catch (const std::exception& e) {
        spdlog::error("Error during PathTracer setup: {}", e.what());
        throw;
    }
}

PathTracer::PathTracer(const OptixContext& context, const Scene& scene, const char* ptx_code, bool is_embedded)
    : context_(context), scene_(scene)
{
    try {
        initialize(false, std::string(ptx_code));
    } catch (const std::exception& e) {
        spdlog::error("Error during PathTracer setup: {}", e.what());
        throw;
    }
}

PathTracer::~PathTracer() {
    // Builders will clean up themselves in their destructors
}

void PathTracer::initialize(bool from_file, const std::string& ptx_path_or_code) {
    // 1. Create pipeline builder and build pipeline
    pipeline_builder_ = std::make_unique<OptixPipelineBuilder>(context_);

    if (from_file) {
        pipeline_builder_->create_module_from_file(ptx_path_or_code);
    } else {
        spdlog::info("Using embedded PTX code...");
        pipeline_builder_->create_module_from_string(ptx_path_or_code);
    }

    pipeline_builder_->create_program_groups();
    pipeline_builder_->create_pipeline();

    // 2. Create SBT builder and build SBT
    sbt_builder_ = std::make_unique<OptixSBTBuilder>();
    sbt_builder_->build_sbt(*pipeline_builder_, scene_);
}

SimulationResult PathTracer::launch(const SimConfig& config, const LightSource& light, const Detector& detector) {
    spdlog::info("\n🚀 Launching simulation...");

    // 1. Prepare device buffers for statistics
    DeviceBuffer flux_buffer(sizeof(double));
    double zero_d = 0.0;
    flux_buffer.upload(&zero_d, sizeof(double));

    DeviceBuffer detected_rays_buffer(sizeof(unsigned long long));
    unsigned long long zero_ull = 0;
    detected_rays_buffer.upload(&zero_ull, sizeof(unsigned long long));

    DeviceBuffer total_bounces_buffer(sizeof(unsigned long long));
    total_bounces_buffer.upload(&zero_ull, sizeof(unsigned long long));

    // 2. Prepare random seeds
    std::vector<unsigned int> seeds(config.num_rays);
    for (size_t i = 0; i < config.num_rays; ++i) {
        seeds[i] = config.random_seed + static_cast<unsigned int>(i * 1234567);
    }
    DeviceBuffer seed_buffer;
    seed_buffer.upload(seeds.data(), seeds.size() * sizeof(unsigned int));

    // 3. Setup device parameters
    DeviceParams params;
    params.traversable = scene_.get_traversable();
    params.flux_buffer = reinterpret_cast<double*>(flux_buffer.get_cu_ptr());
    params.detected_rays_buffer = reinterpret_cast<unsigned long long*>(detected_rays_buffer.get_cu_ptr());
    params.total_bounces_buffer = reinterpret_cast<unsigned long long*>(total_bounces_buffer.get_cu_ptr());
    params.seed_buffer = reinterpret_cast<unsigned int*>(seed_buffer.get_cu_ptr());
    params.num_rays = config.num_rays;
    params.max_bounces = config.max_bounces;
    params.power_per_ray = light.power / static_cast<double>(config.num_rays);
    params.use_nee = config.use_nee;
    params.light_source.position = light.position;
    params.detector.position = detector.position;
    params.detector.normal = detector.normal;
    params.detector.radius = detector.radius;

    DeviceBuffer params_buffer;
    params_buffer.upload(&params, sizeof(DeviceParams));

    // 4. Launch OptiX kernel
    OPTIX_CHECK(optixLaunch(
        pipeline_builder_->get_pipeline(),
        0, // CUDA stream
        params_buffer.get_cu_ptr(),
        params_buffer.size(),
        &sbt_builder_->get_sbt(),
        config.num_rays,
        1,
        1
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Retrieve results
    double flux;
    unsigned long long detected_rays;
    unsigned long long total_bounces;

    flux_buffer.download(&flux, sizeof(double));
    detected_rays_buffer.download(&detected_rays, sizeof(unsigned long long));
    total_bounces_buffer.download(&total_bounces, sizeof(unsigned long long));

    // 6. Calculate statistics
    SimulationResult result;
    result.total_rays = config.num_rays;
    result.detected_rays = detected_rays;
    result.detected_flux = flux * params.power_per_ray;

    double detector_area = M_PI * detector.radius * detector.radius;
    result.irradiance = result.detected_flux / detector_area;

    if (config.num_rays > 0) {
        result.avg_bounces = static_cast<double>(total_bounces) / static_cast<double>(config.num_rays);
    } else {
        result.avg_bounces = 0.0;
    }

    spdlog::info("✅ Simulation complete");
    return result;
}
