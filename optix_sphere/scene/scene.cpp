#include "scene.h"
#include "geometry/mesh_loader.h"
#include <spdlog/spdlog.h>
#include <limits>
#include <algorithm>

Scene Scene::from_obj(const std::string& file_path) {
    spdlog::info("Loading scene from OBJ file: {}", file_path);

    // Load mesh data (geometry + material names only, no material binding)
    Mesh mesh = MeshLoader::load_obj(file_path);

    spdlog::info("✅ Scene loaded to CPU:");
    spdlog::info("   Vertices: {}", mesh.get_vertex_count());
    spdlog::info("   Triangles: {}", mesh.get_triangle_count());
    spdlog::info("   Materials: {}", mesh.get_material_count());

    // Log material names
    std::string material_list;
    for (size_t i = 0; i < mesh.material_names.size(); ++i) {
        if (i > 0) material_list += ", ";
        material_list += mesh.material_names[i];
    }
    if (!material_list.empty()) {
        spdlog::info("   Material names: [{}]", material_list);
    }

    return Scene(std::move(mesh));
}

Scene::Scene(Mesh&& mesh) : mesh_(std::move(mesh)) {
}

std::vector<std::string> Scene::get_material_names() const {
    return mesh_.material_names;
}

size_t Scene::get_vertex_count() const {
    return mesh_.get_vertex_count();
}

size_t Scene::get_triangle_count() const {
    return mesh_.get_triangle_count();
}

size_t Scene::get_material_count() const {
    return mesh_.get_material_count();
}

std::pair<float3, float3> Scene::get_bounds() const {
    if (mesh_.vertices.empty()) {
        return {make_float3(0, 0, 0), make_float3(0, 0, 0)};
    }

    float3 min_bounds = make_float3(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    );
    float3 max_bounds = make_float3(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    );

    for (const auto& v : mesh_.vertices) {
        min_bounds.x = std::min(min_bounds.x, v.x);
        min_bounds.y = std::min(min_bounds.y, v.y);
        min_bounds.z = std::min(min_bounds.z, v.z);

        max_bounds.x = std::max(max_bounds.x, v.x);
        max_bounds.y = std::max(max_bounds.y, v.y);
        max_bounds.z = std::max(max_bounds.z, v.z);
    }

    return {min_bounds, max_bounds};
}
