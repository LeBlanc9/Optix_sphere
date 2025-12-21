#include "optix_pipeline_builder.h"
#include <fstream>
#include <spdlog/spdlog.h>

// Helper to read a file into a string
static std::string read_file_to_string(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

OptixPipelineBuilder::OptixPipelineBuilder(const OptixContext& context)
    : context_(context)
{
}

OptixPipelineBuilder::~OptixPipelineBuilder() {
    if (pipeline_) optixPipelineDestroy(pipeline_);
    if (raygen_pg_) optixProgramGroupDestroy(raygen_pg_);
    if (miss_pg_) optixProgramGroupDestroy(miss_pg_);
    if (sphere_hitgroup_pg_) optixProgramGroupDestroy(sphere_hitgroup_pg_);
    if (detector_hitgroup_pg_) optixProgramGroupDestroy(detector_hitgroup_pg_);
    if (miss_shadow_pg_) optixProgramGroupDestroy(miss_shadow_pg_);
    if (sphere_shadow_hitgroup_pg_) optixProgramGroupDestroy(sphere_shadow_hitgroup_pg_);
    if (detector_shadow_hitgroup_pg_) optixProgramGroupDestroy(detector_shadow_hitgroup_pg_);
    if (module_) optixModuleDestroy(module_);
}

void OptixPipelineBuilder::create_module_from_file(const std::string& ptx_path) {
    spdlog::info("Loading PTX from {}...", ptx_path);
    std::string ptx_code = read_file_to_string(ptx_path);
    create_module_from_string(ptx_code);
}

void OptixPipelineBuilder::create_module_from_string(const std::string& ptx_code) {
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
    if (log_size > 1) spdlog::info("Module creation log: {}", log);
    spdlog::info("✅ Module created");
}

void OptixPipelineBuilder::create_program_groups() {
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

    // Miss (radiance rays)
    OptixProgramGroupDesc miss_desc = {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = module_;
    miss_desc.miss.entryFunctionName = "__miss__sphere";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &miss_desc, 1, &pg_options, log, &log_size, &miss_pg_));

    // Hitgroup for sphere (radiance rays)
    OptixProgramGroupDesc sphere_hitgroup_desc = {};
    sphere_hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sphere_hitgroup_desc.hitgroup.moduleCH = module_;
    sphere_hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
    sphere_hitgroup_desc.hitgroup.moduleIS = module_;
    sphere_hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &sphere_hitgroup_desc, 1, &pg_options, log, &log_size, &sphere_hitgroup_pg_));

    // Hitgroup for detector (radiance rays)
    OptixProgramGroupDesc detector_hitgroup_desc = {};
    detector_hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    detector_hitgroup_desc.hitgroup.moduleCH = module_;
    detector_hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__detector";
    detector_hitgroup_desc.hitgroup.moduleIS = module_;
    detector_hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__disk";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &detector_hitgroup_desc, 1, &pg_options, log, &log_size, &detector_hitgroup_pg_));

    // Shadow ray miss program
    OptixProgramGroupDesc miss_shadow_desc = {};
    miss_shadow_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_shadow_desc.miss.module = module_;
    miss_shadow_desc.miss.entryFunctionName = "__miss__shadow";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &miss_shadow_desc, 1, &pg_options, log, &log_size, &miss_shadow_pg_));

    // Shadow ray hitgroup for sphere (with any-hit for occlusion test)
    OptixProgramGroupDesc sphere_shadow_hitgroup_desc = {};
    sphere_shadow_hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    sphere_shadow_hitgroup_desc.hitgroup.moduleAH = module_;
    sphere_shadow_hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__sphere_shadow";
    sphere_shadow_hitgroup_desc.hitgroup.moduleIS = module_;
    sphere_shadow_hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &sphere_shadow_hitgroup_desc, 1, &pg_options, log, &log_size, &sphere_shadow_hitgroup_pg_));

    // Shadow ray hitgroup for detector (with any-hit)
    OptixProgramGroupDesc detector_shadow_hitgroup_desc = {};
    detector_shadow_hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    detector_shadow_hitgroup_desc.hitgroup.moduleAH = module_;
    detector_shadow_hitgroup_desc.hitgroup.entryFunctionNameAH = "__anyhit__detector_shadow";
    detector_shadow_hitgroup_desc.hitgroup.moduleIS = module_;
    detector_shadow_hitgroup_desc.hitgroup.entryFunctionNameIS = "__intersection__disk";
    log_size = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &detector_shadow_hitgroup_desc, 1, &pg_options, log, &log_size, &detector_shadow_hitgroup_pg_));

    spdlog::info("✅ Program groups created (radiance + shadow rays)");
}

void OptixPipelineBuilder::create_pipeline() {
    OptixProgramGroup program_groups[] = {
        raygen_pg_,
        miss_pg_, miss_shadow_pg_,
        sphere_hitgroup_pg_, detector_hitgroup_pg_,
        sphere_shadow_hitgroup_pg_, detector_shadow_hitgroup_pg_
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2; // For shadow rays: primary + shadow

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
    spdlog::info("✅ Pipeline created");

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
    spdlog::info("✅ Pipeline stack size set");
}
