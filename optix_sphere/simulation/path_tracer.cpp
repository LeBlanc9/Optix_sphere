#include "path_tracer.h"
#include "simulation/device_params.h"
#include "constants.h"
#include "photon/batch.cuh"
#include "photon/launchers.h"
#include <spdlog/spdlog.h>
#include <thrust/device_vector.h>
#include <math.h>


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

    // 2. Create SBT builder (will be rebuilt in launch() with data-driven raygen)
    sbt_builder_ = std::make_unique<OptixSBTBuilder>();
}

SimulationResult PathTracer::launch(
    const SimConfig& config,
    phonder::PhotonSource& photon_source)
{
    spdlog::info("\n🚀 Launching data-driven simulation...");

    // 1. Generate photons on GPU using the new launcher
    phonder::DevicePhotonBatch d_batch;
    generate_photons_on_device(photon_source, d_batch, config.num_rays, config.random_seed);

    unsigned int photon_count = d_batch.size();
    if (photon_count == 0) {
        spdlog::warn("Photon generation resulted in 0 photons. Aborting launch.");
        return {};
    }
    spdlog::info("   Generated {} photons on GPU (SoA format)", photon_count);

    // 2. Build SBT with data-driven raygen
    sbt_builder_->build_sbt(*pipeline_builder_, scene_);

    // 3. Prepare device buffers for statistics
    DeviceBuffer flux_buffer(sizeof(double));
    double zero_d = 0.0;
    flux_buffer.upload(&zero_d, sizeof(double));

    DeviceBuffer detected_rays_buffer(sizeof(unsigned long long));
    unsigned long long zero_ull = 0;
    detected_rays_buffer.upload(&zero_ull, sizeof(unsigned long long));

    DeviceBuffer total_bounces_buffer(sizeof(unsigned long long));
    total_bounces_buffer.upload(&zero_ull, sizeof(unsigned long long));

    // 4. Setup device parameters (no seed_buffer needed for data-driven mode)
    DeviceParams params;
    params.traversable = scene_.get_traversable();
    params.flux_buffer = reinterpret_cast<double*>(flux_buffer.get_cu_ptr());
    params.detected_rays_buffer = reinterpret_cast<unsigned long long*>(detected_rays_buffer.get_cu_ptr());
    params.total_bounces_buffer = reinterpret_cast<unsigned long long*>(total_bounces_buffer.get_cu_ptr());
    params.seed_buffer = nullptr;  // Data-driven mode uses photon_seed_base instead
    params.num_rays = photon_count;
    params.max_bounces = config.max_bounces;
    params.power_per_ray = 1.0;  // Power is already in photon weights
    params.use_nee = config.use_nee;

    // Data-driven mode: SoA photon arrays directly from DevicePhotonBatch (zero-copy)
    params.photon_positions = thrust::raw_pointer_cast(d_batch.positions.data());
    params.photon_directions = thrust::raw_pointer_cast(d_batch.directions.data());
    params.photon_weights = thrust::raw_pointer_cast(d_batch.weights.data());
    params.photon_seed_base = config.random_seed;
    params.num_input_photons = photon_count;

    // Use extracted analytical detector parameters from mesh
    params.detector.position = scene_.get_detector_position();
    params.detector.normal = scene_.get_detector_normal();
    params.detector.radius = scene_.get_detector_radius();

    // Detector triangles are not used for NEE (we use analytical disk approximation)
    params.detector_triangles = nullptr;
    params.num_detector_triangles = 0;
    params.detector_total_area = scene_.get_detector_total_area();

    DeviceBuffer params_buffer;
    params_buffer.upload(&params, sizeof(DeviceParams));

    // 6. Launch OptiX kernel
    OPTIX_CHECK(optixLaunch(
        pipeline_builder_->get_pipeline(),
        0, // CUDA stream
        params_buffer.get_cu_ptr(),
        params_buffer.size(),
        &sbt_builder_->get_sbt(),
        photon_count,  // Launch width = number of photons
        1,
        1
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. Retrieve results
    double flux;
    unsigned long long detected_rays;
    unsigned long long total_bounces;

    flux_buffer.download(&flux, sizeof(double));
    detected_rays_buffer.download(&detected_rays, sizeof(unsigned long long));
    total_bounces_buffer.download(&total_bounces, sizeof(unsigned long long));

    // 8. Calculate statistics with normalization
    // 归一化：每个光子初始权重是source.weight，需要除以光子数来归一化到实际总功率
    double source_power = phonder::get_source_power(photon_source);
    double normalization_factor = (photon_count > 0) ? (source_power / static_cast<double>(photon_count)) : 0.0;

    SimulationResult result;
    result.total_rays = photon_count;
    result.detected_rays = detected_rays;
    result.detected_flux = flux * normalization_factor;  // 归一化到实际功率

    double detector_area = scene_.get_detector_total_area();
    if (detector_area > 1e-9) {
        result.irradiance = result.detected_flux / detector_area;
    } else {
        result.irradiance = 0.0;
    }

    if (photon_count > 0) {
        result.avg_bounces = static_cast<double>(total_bounces) / static_cast<double>(photon_count);
    } else {
        result.avg_bounces = 0.0;
    }

    spdlog::info("✅ Simulation complete");
    return result;
}
