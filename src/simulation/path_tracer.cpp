#include "path_tracer.h"
#include "device_params.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to read a file into a string
static std::string read_file_to_string(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}


PathTracer::PathTracer(const OptixContext& context, const Scene& scene, const std::string& ptx_path)
    : context_(context), scene_(scene)
{
    try {
        create_module(ptx_path);
        create_program_groups();
        create_pipeline();
        create_sbt();
    } catch (const std::exception& e) {
        std::cerr << "Error during PathTracer setup: " << e.what() << std::endl;
        // Cleanup would happen in the destructor
        throw;
    }
}

PathTracer::~PathTracer() {
    if (pipeline_) optixPipelineDestroy(pipeline_);
    if (raygen_pg_) optixProgramGroupDestroy(raygen_pg_);
    if (miss_pg_) optixProgramGroupDestroy(miss_pg_);
    if (hitgroup_pg_) optixProgramGroupDestroy(hitgroup_pg_);
    if (module_) optixModuleDestroy(module_);
}

void PathTracer::create_module(const std::string& ptx_path) {
    std::cout << "Loading PTX from " << ptx_path << "..." << std::endl;
    std::string ptx_code = read_file_to_string(ptx_path);

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    pipeline_compile_options_.usesMotionBlur = false;
    pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options_.numPayloadValues = 2; // Using 64-bit pointer passing (2x 32-bit values)
    pipeline_compile_options_.numAttributeValues = 0;
    pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
    
    char log[2048];
    size_t log_size = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(
        context_.get(),
        &module_compile_options,
        &pipeline_compile_options_,
        ptx_code.c_str(),
        ptx_code.size(),
        log,
        &log_size,
        &module_
    ));
    if (log_size > 1) std::cout << "Module creation log: " << log << std::endl;
    std::cout << "✅ Module created" << std::endl;
}

void PathTracer::create_program_groups() {
    OptixProgramGroupOptions pg_options = {};
    char log[2048];
    size_t log_size;

    // Raygen
    OptixProgramGroupDesc raygen_desc = {};
    raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_desc.raygen.module = module_;
    raygen_desc.raygen.entryFunctionName = "__raygen__forward_trace";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &raygen_desc, 1, &pg_options, log, &log_size, &raygen_pg_));

    // Miss
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = "__miss__sphere";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &miss_desc, 1, &pg_options, log, &log_size, &miss_pg_));

    // Hitgroup
    OptixProgramGroupDesc hitgroup_desc = {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = module_;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
    hitgroup_desc.hitgroup.moduleIS = module_;
    hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &hitgroup_desc, 1, &pg_options, log, &log_size, &hitgroup_pg_));
    
    std::cout << "✅ Program groups created" << std::endl;
}

void PathTracer::create_pipeline() {
    OptixProgramGroup program_groups[] = { raygen_pg_, miss_pg_, hitgroup_pg_ };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2; // Keep it low, recursion is handled in the loop

    char log[2048];
    size_t log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        context_.get(),
        &pipeline_compile_options_,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &log_size,
        &pipeline_
    ));
    std::cout << "✅ Pipeline created" << std::endl;

    // Set pipeline stack size
    unsigned int directCallableStackSizeFromTraversal = 0;
    unsigned int directCallableStackSizeFromState = 0;
    unsigned int continuationStackSize = 8192;
    unsigned int maxTraversableGraphDepth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize,
        maxTraversableGraphDepth
    ));
    std::cout << "✅ Pipeline stack size set" << std::endl;
}

void PathTracer::create_sbt() {
    // Raygen record
    char raygen_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg_, &raygen_header));
    raygen_sbt_record_.upload(&raygen_header, sizeof(raygen_header));
    sbt_.raygenRecord = raygen_sbt_record_.get_cu_ptr();

    // Miss record
    char miss_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg_, &miss_header));
    miss_sbt_record_.upload(&miss_header, sizeof(miss_header));
    sbt_.missRecordBase = miss_sbt_record_.get_cu_ptr();
    sbt_.missRecordStrideInBytes = sizeof(miss_header);
    sbt_.missRecordCount = 1;

    // Hitgroup record
    // 1. Create a host-side buffer for the record
    std::vector<char> hg_record(OPTIX_SBT_RECORD_HEADER_SIZE + sizeof(SphereSbtData));

    // 2. Pack the header
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg_, hg_record.data()));

    // 3. Get the sphere data from the scene and copy it after the header
    const DeviceBuffer& sphere_data_buffer = scene_.get_sphere_data_buffer();
    sphere_data_buffer.download(hg_record.data() + OPTIX_SBT_RECORD_HEADER_SIZE, sizeof(SphereSbtData));

    // 4. Align the record size to OPTIX_SBT_RECORD_ALIGNMENT
    size_t record_size = hg_record.size();
    size_t aligned_size = ((record_size + OPTIX_SBT_RECORD_ALIGNMENT - 1) / OPTIX_SBT_RECORD_ALIGNMENT) * OPTIX_SBT_RECORD_ALIGNMENT;
    hg_record.resize(aligned_size, 0);

    // 5. Upload the complete record to the GPU
    hitgroup_sbt_record_.upload(hg_record.data(), hg_record.size());

    // 6. Point the SBT to the new record
    sbt_.hitgroupRecordBase = hitgroup_sbt_record_.get_cu_ptr();
    sbt_.hitgroupRecordStrideInBytes = aligned_size;
    sbt_.hitgroupRecordCount = 1;

    std::cout << "✅ SBT created" << std::endl;
}

SimulationResult PathTracer::launch(const SimConfig& config, const LightSource& light, const Detector& detector) {
    std::cout << "\n🚀 Launching simulation..." << std::endl;

    // 1. Prepare device buffers
    DeviceBuffer flux_buffer(sizeof(float));
    DeviceBuffer seed_buffer(config.num_rays * sizeof(unsigned int));
    
    float zero = 0.0f;
    flux_buffer.upload(&zero, sizeof(float)); // Reset flux counter

    std::vector<unsigned int> seeds(config.num_rays);
    for (int i = 0; i < config.num_rays; ++i) {
        seeds[i] = i * 1234567; // Simple seed initialization
    }
    seed_buffer.upload(seeds);

    // 2. Populate the device parameter block
    DeviceParams params = {};
    params.traversable = scene_.get_traversable();
    params.flux_buffer = flux_buffer.get<float>();
    params.seed_buffer = seed_buffer.get<unsigned int>();
    params.num_rays = config.num_rays;
    params.max_bounces = config.max_bounces;
    params.power_per_ray = light.power / config.num_rays;
    
    params.light_source.position = light.position;
    params.detector.position = detector.position;
    params.detector.normal = detector.normal;
    params.detector.radius = detector.radius;

    DeviceBuffer params_buffer;
    params_buffer.upload(&params, sizeof(DeviceParams));

    // 3. Launch
    OPTIX_CHECK(optixLaunch(
        pipeline_,
        0, // stream
        params_buffer.get_cu_ptr(),
        sizeof(DeviceParams),
        &sbt_,
        config.num_rays, // width
        1,               // height
        1                // depth
    ));

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "✅ Simulation finished." << std::endl;

    // 4. Retrieve results
    float detected_flux;
    flux_buffer.download(&detected_flux, sizeof(float));

    float detector_area = M_PI * detector.radius * detector.radius;
    
    SimulationResult result = {};
    result.total_rays = config.num_rays;
    result.detected_flux = detected_flux;
    result.irradiance = detected_flux / detector_area;
    
    // detected_rays and avg_bounces would require more buffers to track,
    // skipping for now to keep it simple.
    result.detected_rays = 0; 
    result.avg_bounces = 0.0f;

    return result;
}
