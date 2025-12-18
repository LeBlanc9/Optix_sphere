#include "path_tracer.h"
#include "device_params.h"
#include "constants.h"
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
    if (sphere_hitgroup_pg_) optixProgramGroupDestroy(sphere_hitgroup_pg_);
    if (detector_hitgroup_pg_) optixProgramGroupDestroy(detector_hitgroup_pg_);
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

    // Hitgroup for sphere
    OptixProgramGroupDesc sphere_hitgroup_desc = {};
    sphere_hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sphere_hitgroup_desc.hitgroup.moduleCH = module_;
    sphere_hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
    sphere_hitgroup_desc.hitgroup.moduleIS = module_;
    sphere_hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &sphere_hitgroup_desc, 1, &pg_options, log, &log_size, &sphere_hitgroup_pg_));

    // Hitgroup for detector
    OptixProgramGroupDesc detector_hitgroup_desc = {};
    detector_hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    detector_hitgroup_desc.hitgroup.moduleCH = module_;
    detector_hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__detector";
    detector_hitgroup_desc.hitgroup.moduleIS = module_;
    detector_hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__disk";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &detector_hitgroup_desc, 1, &pg_options, log, &log_size, &detector_hitgroup_pg_));

    std::cout << "✅ Program groups created (raygen, miss, sphere, detector)" << std::endl;
}

void PathTracer::create_pipeline() {
    OptixProgramGroup program_groups[] = { raygen_pg_, miss_pg_, sphere_hitgroup_pg_, detector_hitgroup_pg_ };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1; // For iterative tracer, depth is 1

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

    // Hitgroup records (sphere + detector)
    // Calculate aligned record size (use the larger of the two data structures)
    size_t sphere_record_size = OPTIX_SBT_RECORD_HEADER_SIZE + sizeof(SphereSbtData);
    size_t detector_record_size = OPTIX_SBT_RECORD_HEADER_SIZE + sizeof(DiskSbtData);
    size_t record_size = sphere_record_size > detector_record_size ? sphere_record_size : detector_record_size;
    size_t aligned_record_size = ((record_size + OPTIX_SBT_RECORD_ALIGNMENT - 1) / OPTIX_SBT_RECORD_ALIGNMENT) * OPTIX_SBT_RECORD_ALIGNMENT;

    // Allocate space for 2 records
    hitgroup_sbt_records_.alloc(2 * aligned_record_size);

    // Build sphere record (index 0)
    char sphere_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(sphere_hitgroup_pg_, &sphere_header));

    CUdeviceptr sphere_record_ptr = hitgroup_sbt_records_.get_cu_ptr();
    CUDA_CHECK(cudaMemcpy(
        (void*)sphere_record_ptr,
        &sphere_header,
        OPTIX_SBT_RECORD_HEADER_SIZE,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        (void*)(sphere_record_ptr + OPTIX_SBT_RECORD_HEADER_SIZE),
        (void*)scene_.get_sphere_data_buffer().get_cu_ptr(),
        sizeof(SphereSbtData),
        cudaMemcpyDeviceToDevice
    ));

    // Build detector record (index 1)
    char detector_header[OPTIX_SBT_RECORD_HEADER_SIZE];
    OPTIX_CHECK(optixSbtRecordPackHeader(detector_hitgroup_pg_, &detector_header));

    CUdeviceptr detector_record_ptr = hitgroup_sbt_records_.get_cu_ptr() + aligned_record_size;
    CUDA_CHECK(cudaMemcpy(
        (void*)detector_record_ptr,
        &detector_header,
        OPTIX_SBT_RECORD_HEADER_SIZE,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        (void*)(detector_record_ptr + OPTIX_SBT_RECORD_HEADER_SIZE),
        (void*)scene_.get_detector_data_buffer().get_cu_ptr(),
        sizeof(DiskSbtData),
        cudaMemcpyDeviceToDevice
    ));

    // Point SBT to the records
    sbt_.hitgroupRecordBase = hitgroup_sbt_records_.get_cu_ptr();
    sbt_.hitgroupRecordStrideInBytes = aligned_record_size;
    sbt_.hitgroupRecordCount = 2;

    std::cout << "✅ SBT created" << std::endl;
}

SimulationResult PathTracer::launch(const SimConfig& config, const LightSource& light, const Detector& detector) {
    std::cout << "\n🚀 Launching simulation..." << std::endl;

    // 1. Prepare device buffers for statistics
    DeviceBuffer flux_buffer(sizeof(double));  // 使用double精度
    DeviceBuffer detected_rays_buffer(sizeof(unsigned long long));
    DeviceBuffer total_bounces_buffer(sizeof(unsigned long long));
    DeviceBuffer seed_buffer(config.num_rays * sizeof(unsigned int));

    // Reset statistic counters to zero
    double zero_d = 0.0;  // 使用double
    unsigned long long zero_ull = 0;
    flux_buffer.upload(&zero_d, sizeof(double));
    detected_rays_buffer.upload(&zero_ull, sizeof(unsigned long long));
    total_bounces_buffer.upload(&zero_ull, sizeof(unsigned long long));

    // 初始化随机数种子
    std::vector<unsigned int> seeds(config.num_rays);
    for (size_t i = 0; i < config.num_rays; ++i) {
        seeds[i] = config.random_seed + static_cast<unsigned int>(i * 1234567);
    }
    seed_buffer.upload(seeds.data(), seeds.size() * sizeof(unsigned int));

    // 2. Populate the device parameter block
    DeviceParams params = {};
    params.traversable = scene_.get_traversable();
    params.flux_buffer = flux_buffer.get<double>();
    params.detected_rays_buffer = detected_rays_buffer.get<unsigned long long>();
    params.total_bounces_buffer = total_bounces_buffer.get<unsigned long long>();
    params.seed_buffer = seed_buffer.get<unsigned int>();
    params.num_rays = config.num_rays;
    params.max_bounces = config.max_bounces;
    params.power_per_ray = light.power / config.num_rays;
    params.use_nee = config.use_nee;  // 传递NEE配置

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
    double accumulated_weight;  // 累积的无量纲权重 (double精度)
    unsigned long long detected_rays_count;
    unsigned long long total_bounces_count;

    flux_buffer.download(&accumulated_weight, sizeof(double));
    detected_rays_buffer.download(&detected_rays_count, sizeof(unsigned long long));
    total_bounces_buffer.download(&total_bounces_count, sizeof(unsigned long long));

    // 归一化：将无量纲权重转换为物理单位的通量(W)
    // flux = (accumulated_weight / num_rays) × total_power
    double normalization = light.power / config.num_rays;
    double detected_flux = accumulated_weight * normalization;  // W

    double detector_area = M_PI * detector.radius * detector.radius;  // mm²

    SimulationResult result = {};
    result.total_rays = config.num_rays;
    result.detected_flux = detected_flux;               // W
    result.irradiance = detected_flux / detector_area;  // W/mm²
    result.detected_rays = detected_rays_count;
    result.avg_bounces = (config.num_rays > 0) ?
        static_cast<float>(total_bounces_count) / config.num_rays : 0.0f;

    return result;
}
