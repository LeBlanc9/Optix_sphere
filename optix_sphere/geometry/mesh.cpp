#include "mesh.h"
#include <algorithm>
#include <set>
#include <spdlog/spdlog.h>

int Mesh::find_material_index(const std::string& material_name) const {
    auto it = std::find(material_names.begin(), material_names.end(), material_name);
    if (it == material_names.end()) {
        return -1;
    }
    return static_cast<int>(std::distance(material_names.begin(), it));
}

size_t Mesh::get_triangle_count_by_material(const std::string& material_name) const {
    int material_index = find_material_index(material_name);
    if (material_index == -1) {
        spdlog::warn("Material '{}' not found in mesh", material_name);
        return 0;
    }

    // Count triangles with this material
    size_t count = 0;
    for (int mat_idx : triangle_materials) {
        if (mat_idx == material_index) {
            ++count;
        }
    }

    return count;
}

size_t Mesh::get_vertex_count_by_material(const std::string& material_name) const {
    int material_index = find_material_index(material_name);
    if (material_index == -1) {
        spdlog::warn("Material '{}' not found in mesh", material_name);
        return 0;
    }

    // Collect unique vertex indices used by this material
    std::set<unsigned int> unique_vertices;

    for (size_t tri_idx = 0; tri_idx < triangle_materials.size(); ++tri_idx) {
        if (triangle_materials[tri_idx] == material_index) {
            // Add the three vertices of this triangle
            const uint3& triangle = indices[tri_idx];
            unique_vertices.insert(triangle.x);
            unique_vertices.insert(triangle.y);
            unique_vertices.insert(triangle.z);
        }
    }

    return unique_vertices.size();
}

std::vector<size_t> Mesh::get_triangle_indices_by_material(const std::string& material_name) const {
    int material_index = find_material_index(material_name);
    if (material_index == -1) {
        spdlog::warn("Material '{}' not found in mesh", material_name);
        return {};
    }

    // Collect all triangle indices with this material
    std::vector<size_t> triangle_indices;
    triangle_indices.reserve(triangle_materials.size() / material_names.size()); // Rough estimate

    for (size_t tri_idx = 0; tri_idx < triangle_materials.size(); ++tri_idx) {
        if (triangle_materials[tri_idx] == material_index) {
            triangle_indices.push_back(tri_idx);
        }
    }

    return triangle_indices;
}
