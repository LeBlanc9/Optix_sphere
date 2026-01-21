#include "scene.h"
#include "geometry/obj_loader.h"
#include <spdlog/spdlog.h>
#include <limits>
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Scene Scene::from_obj(const std::string& file_path) {
    spdlog::info("Loading scene from OBJ file: {}", file_path);

    // Load mesh data (geometry + material names only, no material binding)
    Mesh mesh = ObjLoader::load_obj(file_path);

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

void Scene::rotate_y(float angle_degrees) {
    float angle_rad = angle_degrees * M_PI / 180.0f;
    float cos_a = std::cos(angle_rad);
    float sin_a = std::sin(angle_rad);

    spdlog::info("Rotating scene around Y-axis by {} degrees", angle_degrees);

    // Rotate vertices
    for (auto& v : mesh_.vertices) {
        float x_new = cos_a * v.x + sin_a * v.z;
        float z_new = -sin_a * v.x + cos_a * v.z;
        v.x = x_new;
        v.z = z_new;
    }

    // Rotate normals (if present)
    for (auto& n : mesh_.normals) {
        float x_new = cos_a * n.x + sin_a * n.z;
        float z_new = -sin_a * n.x + cos_a * n.z;
        n.x = x_new;
        n.z = z_new;
    }

    spdlog::info("✅ Scene rotated successfully");
}

void Scene::translate(const float3& offset) {
    spdlog::info("Translating scene by ({}, {}, {})", offset.x, offset.y, offset.z);

    // Translate all vertices
    for (auto& v : mesh_.vertices) {
        v.x += offset.x;
        v.y += offset.y;
        v.z += offset.z;
    }

    // Normals don't change under translation

    spdlog::info("✅ Scene translated successfully");
}

void Scene::scale(float scale_factor) {
    if (scale_factor <= 0.0f) {
        spdlog::error("Invalid scale factor: {} (must be positive)", scale_factor);
        return;
    }

    spdlog::info("Scaling scene by factor {}", scale_factor);

    // Scale vertices
    for (auto& v : mesh_.vertices) {
        v.x *= scale_factor;
        v.y *= scale_factor;
        v.z *= scale_factor;
    }

    // Normals don't need scaling (they remain unit length)

    spdlog::info("✅ Scene scaled successfully");
}

// Material query methods are now inline delegations to mesh_ in scene.h
