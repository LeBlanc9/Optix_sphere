#include "device_scene.h"
#include <spdlog/spdlog.h>
#include <cmath>

// Helper function to extract analytical detector parameters from mesh triangles
namespace {
    struct AnalyticalDetector {
        float3 center;
        float3 normal;
        float radius;
    };

    AnalyticalDetector extract_analytical_detector(const std::vector<float3>& detector_vertices) {
        if (detector_vertices.empty()) {
            return {{0, 0, 0}, {0, 0, 0}, 0};
        }

        // 1. Calculate centroid
        float3 centroid = make_float3(0, 0, 0);
        for (const auto& v : detector_vertices) {
            centroid.x += v.x;
            centroid.y += v.y;
            centroid.z += v.z;
        }
        centroid.x /= detector_vertices.size();
        centroid.y /= detector_vertices.size();
        centroid.z /= detector_vertices.size();

        // 2. Calculate average normal (from all triangles)
        float3 avg_normal = make_float3(0, 0, 0);
        for (size_t i = 0; i + 2 < detector_vertices.size(); i += 3) {
            float3 v0 = detector_vertices[i];
            float3 v1 = detector_vertices[i + 1];
            float3 v2 = detector_vertices[i + 2];

            float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

            // Cross product
            float3 n = make_float3(
                e1.y * e2.z - e1.z * e2.y,
                e1.z * e2.x - e1.x * e2.z,
                e1.x * e2.y - e1.y * e2.x
            );

            avg_normal.x += n.x;
            avg_normal.y += n.y;
            avg_normal.z += n.z;
        }

        // Normalize
        float len = sqrtf(avg_normal.x * avg_normal.x +
                         avg_normal.y * avg_normal.y +
                         avg_normal.z * avg_normal.z);
        if (len > 0) {
            avg_normal.x /= len;
            avg_normal.y /= len;
            avg_normal.z /= len;
        }

        // 3. Calculate effective radius from total area
        // First, calculate total area of all triangles
        float total_area = 0.0f;
        for (size_t i = 0; i + 2 < detector_vertices.size(); i += 3) {
            float3 v0 = detector_vertices[i];
            float3 v1 = detector_vertices[i + 1];
            float3 v2 = detector_vertices[i + 2];

            float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
            float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);

            float3 cp = make_float3(
                e1.y * e2.z - e1.z * e2.y,
                e1.z * e2.x - e1.x * e2.z,
                e1.x * e2.y - e1.y * e2.x
            );

            float tri_area = 0.5f * sqrtf(cp.x * cp.x + cp.y * cp.y + cp.z * cp.z);
            total_area += tri_area;
        }

        // Calculate equivalent radius: area = π × r²  =>  r = sqrt(area / π)
        float radius = sqrtf(total_area / 3.14159265359f);

        return {centroid, avg_normal, radius};
    }
}

DeviceScene::DeviceScene(const OptixContext& context) : context_(context) {
    // Materials will be dynamically created based on mesh configuration
}

void DeviceScene::build(
    const Mesh& mesh,
    const std::map<std::string, MaterialFactory>& material_factories
) {
    spdlog::info("Building GPU scene from mesh: {} vertices, {} triangles, {} materials",
                 mesh.get_vertex_count(), mesh.get_triangle_count(), mesh.get_material_count());

    // 1. Upload vertex and index data to GPU
    vertex_buffer_.upload(mesh.vertices.data(),
                         mesh.vertices.size() * sizeof(float3));
    index_buffer_.upload(mesh.indices.data(),
                        mesh.indices.size() * sizeof(uint3));

    // 3. Build triangle-based GAS
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr d_vertex = vertex_buffer_.get_cu_ptr();
    build_input.triangleArray.vertexBuffers = &d_vertex;
    build_input.triangleArray.numVertices = static_cast<unsigned int>(mesh.vertices.size());
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;

    build_input.triangleArray.indexBuffer = index_buffer_.get_cu_ptr();
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.indices.size());
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

    // Build per-triangle SBT indices based on material index
    std::vector<unsigned int> sbt_indices(mesh.indices.size());
    for (size_t i = 0; i < mesh.triangle_materials.size(); ++i) {
        sbt_indices[i] = static_cast<unsigned int>(mesh.triangle_materials[i]);
    }

    DeviceBuffer sbt_index_buffer;
    sbt_index_buffer.upload(sbt_indices.data(),
                           sbt_indices.size() * sizeof(unsigned int));

    build_input.triangleArray.sbtIndexOffsetBuffer = sbt_index_buffer.get_cu_ptr();
    build_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(unsigned int);
    build_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(unsigned int);

    // Flags for each material (all set to NONE for now)
    std::vector<unsigned int> input_flags(mesh.get_material_count(), OPTIX_GEOMETRY_FLAG_NONE);
    build_input.triangleArray.flags = input_flags.data();
    build_input.triangleArray.numSbtRecords = static_cast<unsigned int>(mesh.get_material_count());

    // 4. Compute memory usage for the GAS
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context_.get(),
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes
    ));

    // 5. Build the GAS
    DeviceBuffer temp_buffer(gas_buffer_sizes.tempSizeInBytes);
    gas_buffer_.alloc(gas_buffer_sizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
        context_.get(),
        0,
        &accel_options,
        &build_input,
        1,
        temp_buffer.get_cu_ptr(),
        gas_buffer_sizes.tempSizeInBytes,
        gas_buffer_.get_cu_ptr(),
        gas_buffer_sizes.outputSizeInBytes,
        &traversable_,
        nullptr,
        0
    ));

    // 6. Create material instances using material factories
    materials_.resize(mesh.get_material_count());
    material_name_to_index_.clear();  // Clear old mappings
    for (size_t i = 0; i < mesh.material_names.size(); ++i) {
        const std::string& name = mesh.material_names[i];
        auto it = material_factories.find(name);
        if (it == material_factories.end()) {
            throw std::runtime_error("Material factory not found for material: " + name);
        }
        materials_[i] = it->second();  // Call factory
        material_name_to_index_[name] = i;  // Store name-to-index mapping
        spdlog::info("Created material [{}]: {}", i, name);
    }

    // 7. Extract detector triangles for NEE
    std::vector<float3> detector_vertices;
    detector_total_area_ = 0.0f;

    for (size_t i = 0; i < mesh.triangle_materials.size(); ++i) {
        int mat_idx = mesh.triangle_materials[i];
        if (mat_idx < static_cast<int>(mesh.material_names.size())) {
            const std::string& name = mesh.material_names[mat_idx];
            // Check if this is a detector material by creating a test instance
            auto it = material_factories.find(name);
            if (it != material_factories.end()) {
                auto test_mat = it->second();
                    if (test_mat->get_kernel_name() == "__closesthit__detector") {
                    uint3 idx = mesh.indices[i];
                    float3 v0 = mesh.vertices[idx.x];
                    float3 v1 = mesh.vertices[idx.y];
                    float3 v2 = mesh.vertices[idx.z];

                detector_vertices.push_back(v0);
                detector_vertices.push_back(v1);
                detector_vertices.push_back(v2);

                // Calculate triangle area
                float3 edge1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
                float3 edge2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
                float3 cross_product = make_float3(
                    edge1.y * edge2.z - edge1.z * edge2.y,
                    edge1.z * edge2.x - edge1.x * edge2.z,
                    edge1.x * edge2.y - edge1.y * edge2.x
                );
                float area = 0.5f * sqrtf(cross_product.x * cross_product.x +
                                          cross_product.y * cross_product.y +
                                          cross_product.z * cross_product.z);
                detector_total_area_ += area;
                }
            }
        }
    }

    if (!detector_vertices.empty()) {
        // Extract analytical detector parameters from mesh
        auto analytical = extract_analytical_detector(detector_vertices);
        detector_position_ = analytical.center;
        detector_normal_ = analytical.normal;
        detector_radius_ = analytical.radius;

        spdlog::info("Extracted detector parameters:");
        spdlog::info("  Position: ({:.6f}, {:.6f}, {:.6f}) mm",
                     analytical.center.x, analytical.center.y, analytical.center.z);
        spdlog::info("  Normal: ({:.6f}, {:.6f}, {:.6f})",
                     analytical.normal.x, analytical.normal.y, analytical.normal.z);
        spdlog::info("  Radius: {:.6f} mm", analytical.radius);
        spdlog::info("  Total area: {:.6f} mm²", detector_total_area_);
        spdlog::info("  Equivalent area (π×r²): {:.6f} mm²",
                     3.14159265359f * analytical.radius * analytical.radius);
    } else {
        spdlog::warn("No detector triangles found in mesh!");
    }

    spdlog::info("✅ Scene built successfully");
}

// ============================================
// Material Update Methods
// ============================================

void DeviceScene::update_material(
    const std::string& name,
    const MaterialFactory& factory
) {
    spdlog::info("Updating material '{}' (fast path, no geometry rebuild)", name);

    // Find material index
    auto it = material_name_to_index_.find(name);
    if (it == material_name_to_index_.end()) {
        std::string error_msg = "Material '" + name + "' not found in scene. Available materials: ";
        auto available = get_material_names();
        for (size_t i = 0; i < available.size(); ++i) {
            if (i > 0) error_msg += ", ";
            error_msg += available[i];
        }
        spdlog::error(error_msg);
        throw std::runtime_error(error_msg);
    }

    size_t mat_idx = it->second;
    if (mat_idx >= materials_.size()) {
        std::string error_msg = "Invalid material index " + std::to_string(mat_idx) +
                               " for '" + name + "'";
        spdlog::error(error_msg);
        throw std::runtime_error(error_msg);
    }

    // Create new material using factory (sphere center already captured in closure)
    materials_[mat_idx] = factory();
    spdlog::info("✅ Updated material [{}]: {}", mat_idx, name);
}

std::vector<std::string> DeviceScene::get_material_names() const {
    std::vector<std::string> names;
    names.reserve(material_name_to_index_.size());
    for (const auto& [name, idx] : material_name_to_index_) {
        names.push_back(name);
    }
    return names;
}