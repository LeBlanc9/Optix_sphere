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
    // Destroy pipeline
    if (pipeline_) {
        optixPipelineDestroy(pipeline_);
    }

    // Destroy all program groups
    for (auto& [name, pg] : program_groups_) {
        if (pg) {
            optixProgramGroupDestroy(pg);
        }
    }
    program_groups_.clear();

    // Destroy module
    if (module_) {
        optixModuleDestroy(module_);
    }
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

    // Helper lambda to create and register a program group
    auto create_and_register = [&](const std::string& name, const OptixProgramGroupDesc& desc) {
        OptixProgramGroup pg = nullptr;
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(context_.get(), &desc, 1, &pg_options, log, &log_size, &pg));
        program_groups_[name] = pg;
    };

    // ============================================
    // Raygen and Miss programs
    // ============================================
    {
        // Procedural raygen (isotropic point light source)
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__forward_trace";
        create_and_register("__raygen__forward_trace", desc);
    }
    {
        // Data-driven raygen (reads from input photon array)
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__data_driven";
        create_and_register("__raygen__data_driven", desc);
    }
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = module_;
        desc.miss.entryFunctionName = "__miss__sphere";
        create_and_register("__miss__sphere", desc);
    }
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = module_;
        desc.miss.entryFunctionName = "__miss__shadow";
        create_and_register("__miss__shadow", desc);
    }

    // ============================================
    // Analytical geometry hitgroups (custom primitives)
    // ============================================
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
        desc.hitgroup.moduleIS = module_;
        desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        create_and_register("__closesthit__sphere", desc);
    }
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH = module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__sphere_shadow";
        desc.hitgroup.moduleIS = module_;
        desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        create_and_register("__anyhit__sphere_shadow", desc);
    }

    // ============================================
    // Physics-based material hitgroups
    // ============================================
    {
        // Detector (closest-hit)
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__detector";
        desc.hitgroup.moduleIS = module_;
        desc.hitgroup.entryFunctionNameIS = "__intersection__disk";
        create_and_register("__closesthit__detector", desc);
    }
    {
        // Detector (shadow any-hit)
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH = module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__detector_shadow";
        desc.hitgroup.moduleIS = module_;
        desc.hitgroup.entryFunctionNameIS = "__intersection__disk";
        create_and_register("__anyhit__detector_shadow", desc);
    }
    {
        // Lambertian (triangle mesh closest-hit)
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__lambertian";
        create_and_register("__closesthit__lambertian", desc);
    }
    {
        // Lambertian (shadow any-hit)
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH = module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__lambertian_shadow";
        create_and_register("__anyhit__lambertian_shadow", desc);
    }
    {
        // Absorber (triangle mesh closest-hit)
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__absorber";
        create_and_register("__closesthit__absorber", desc);
    }
    {
        // Absorber (shadow any-hit)
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH = module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__absorber_shadow";
        create_and_register("__anyhit__absorber_shadow", desc);
    }

    spdlog::info("✅ Program groups created ({} total)", program_groups_.size());
}

OptixProgramGroup OptixPipelineBuilder::get_program_group(const std::string& kernel_name) const {
    auto it = program_groups_.find(kernel_name);
    if (it != program_groups_.end()) {
        return it->second;
    }

    spdlog::error("Program group not found for kernel: {}", kernel_name);
    return nullptr;
}

void OptixPipelineBuilder::create_pipeline() {
    // Collect all program groups from the map
    std::vector<OptixProgramGroup> all_program_groups;
    all_program_groups.reserve(program_groups_.size());
    for (const auto& [name, pg] : program_groups_) {
        all_program_groups.push_back(pg);
    }

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2; // For shadow rays: primary + shadow

    char log[2048];
    size_t log_size = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        context_.get(),
        &pipeline_compile_options_,
        &pipeline_link_options,
        all_program_groups.data(),
        static_cast<unsigned int>(all_program_groups.size()),
        log,
        &log_size,
        &pipeline_
    ));
    spdlog::info("✅ Pipeline created ({} program groups)", all_program_groups.size());

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
