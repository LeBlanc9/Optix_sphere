#include "mesh.h"
#include "obj_loader.h"
#include <algorithm>
#include <set>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace {

std::string get_dirname(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return "";
    }
    return path.substr(0, pos + 1);
}

std::string get_basename(const std::string& path) {
    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

std::string strip_extension(const std::string& filename) {
    const size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) {
        return filename;
    }
    return filename.substr(0, pos);
}

}  // namespace

Mesh Mesh::from_obj(const std::string& file_path) {
    return ObjLoader::load_obj(file_path);
}

void Mesh::to_obj(const std::string& obj_file_path, const std::string& mtl_file_path) const {
    if (obj_file_path.empty()) {
        throw std::invalid_argument("OBJ output path cannot be empty");
    }

    const std::string obj_dir = get_dirname(obj_file_path);
    const std::string obj_name = get_basename(obj_file_path);
    const std::string default_mtl_name = strip_extension(obj_name) + ".mtl";

    const std::string final_mtl_path = mtl_file_path.empty() ? (obj_dir + default_mtl_name) : mtl_file_path;
    const std::string mtl_ref_name = mtl_file_path.empty() ? default_mtl_name : get_basename(final_mtl_path);

    std::ofstream obj_out(obj_file_path);
    if (!obj_out.is_open()) {
        throw std::runtime_error("Failed to open OBJ file for writing: " + obj_file_path);
    }

    obj_out << "# Exported by optix_sphere::Mesh\n";
    if (!materials.empty()) {
        obj_out << "mtllib " << mtl_ref_name << "\n";
    }
    obj_out << "\n";
    obj_out << std::fixed << std::setprecision(9);

    for (const auto& v : vertices) {
        obj_out << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }

    const bool has_normals = normals.size() == vertices.size() && !normals.empty();
    if (has_normals) {
        for (const auto& n : normals) {
            obj_out << "vn " << n.x << " " << n.y << " " << n.z << "\n";
        }
    }

    obj_out << "\n";
    int current_material = -1;
    for (size_t tri_idx = 0; tri_idx < indices.size(); ++tri_idx) {
        int mat_idx = 0;
        if (tri_idx < triangle_materials.size()) {
            mat_idx = triangle_materials[tri_idx];
        }

        if (!materials.empty() && mat_idx >= 0 && mat_idx < static_cast<int>(materials.size()) && mat_idx != current_material) {
            obj_out << "usemtl " << materials[mat_idx].name << "\n";
            current_material = mat_idx;
        }

        const uint3& tri = indices[tri_idx];
        const unsigned int i0 = tri.x + 1;
        const unsigned int i1 = tri.y + 1;
        const unsigned int i2 = tri.z + 1;

        if (has_normals) {
            obj_out << "f "
                    << i0 << "//" << i0 << " "
                    << i1 << "//" << i1 << " "
                    << i2 << "//" << i2 << "\n";
        } else {
            obj_out << "f " << i0 << " " << i1 << " " << i2 << "\n";
        }
    }

    obj_out.close();

    if (!materials.empty()) {
        std::ofstream mtl_out(final_mtl_path);
        if (!mtl_out.is_open()) {
            throw std::runtime_error("Failed to open MTL file for writing: " + final_mtl_path);
        }

        mtl_out << "# Exported by optix_sphere::Mesh\n\n";
        mtl_out << std::fixed << std::setprecision(6);
        for (const auto& mat : materials) {
            mtl_out << "newmtl " << mat.name << "\n";
            mtl_out << "Ka " << mat.ambient.x << " " << mat.ambient.y << " " << mat.ambient.z << "\n";
            mtl_out << "Kd " << mat.diffuse.x << " " << mat.diffuse.y << " " << mat.diffuse.z << "\n";
            mtl_out << "Ks " << mat.specular.x << " " << mat.specular.y << " " << mat.specular.z << "\n";
            mtl_out << "Ns 0.000000\n";
            mtl_out << "d 1.000000\n\n";
        }
    }

    spdlog::info("Exported mesh to OBJ '{}' and MTL '{}'", obj_file_path, final_mtl_path);
}

std::pair<float3, float3> Mesh::get_bounds() const {
    if (vertices.empty()) {
        float3 zero = make_float3(0.0f, 0.0f, 0.0f);
        return {zero, zero};
    }

    float3 min_v = vertices[0];
    float3 max_v = vertices[0];

    for (const auto& v : vertices) {
        min_v.x = fminf(min_v.x, v.x);
        min_v.y = fminf(min_v.y, v.y);
        min_v.z = fminf(min_v.z, v.z);

        max_v.x = fmaxf(max_v.x, v.x);
        max_v.y = fmaxf(max_v.y, v.y);
        max_v.z = fmaxf(max_v.z, v.z);
    }

    return {min_v, max_v};
}

int Mesh::find_material_index(const std::string& material_name) const {
    for (size_t i = 0; i < materials.size(); ++i) {
        if (materials[i].name == material_name) {
            return static_cast<int>(i);
        }
    }
    return -1;
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

Mesh Mesh::extract_mesh_by_material(const std::string& material_name) const {
    int material_index = find_material_index(material_name);
    if (material_index == -1) {
        spdlog::warn("Material '{}' not found in mesh, returning empty mesh", material_name);
        return Mesh{};
    }

    Mesh result;
    result.materials = {materials[material_index]};

    // Map old vertex index -> new vertex index
    std::map<unsigned int, unsigned int> vertex_map;

    // Extract triangles with this material
    for (size_t tri_idx = 0; tri_idx < triangle_materials.size(); ++tri_idx) {
        if (triangle_materials[tri_idx] == material_index) {
            const uint3& old_tri = indices[tri_idx];

            // Add vertices if not already added
            auto add_vertex = [&](unsigned int old_idx) -> unsigned int {
                auto it = vertex_map.find(old_idx);
                if (it != vertex_map.end()) {
                    return it->second;
                }
                unsigned int new_idx = static_cast<unsigned int>(result.vertices.size());
                result.vertices.push_back(vertices[old_idx]);
                result.normals.push_back(normals[old_idx]);
                vertex_map[old_idx] = new_idx;
                return new_idx;
            };

            // Add triangle with remapped indices
            uint3 new_tri;
            new_tri.x = add_vertex(old_tri.x);
            new_tri.y = add_vertex(old_tri.y);
            new_tri.z = add_vertex(old_tri.z);
            result.indices.push_back(new_tri);
            result.triangle_materials.push_back(0); // Only one material in result
        }
    }

    spdlog::info("Extracted material '{}': {} vertices, {} triangles",
                 material_name, result.vertices.size(), result.indices.size());

    return result;
}

std::map<std::string, Mesh> Mesh::split_by_material() const {
    std::map<std::string, Mesh> result;

    for (const auto& mat : materials) {
        result[mat.name] = extract_mesh_by_material(mat.name);
    }

    spdlog::info("Split mesh into {} parts by material", result.size());
    return result;
}

void Mesh::remove_mesh_by_material(const std::string& material_name) {
    int material_index = find_material_index(material_name);
    if (material_index == -1) {
        spdlog::warn("Material '{}' not found in mesh, nothing to remove", material_name);
        return;
    }

    // Build new material list and index mapping
    std::vector<MeshMaterial> new_materials;
    std::map<int, int> old_to_new_material_index;

    for (size_t i = 0; i < materials.size(); ++i) {
        if (static_cast<int>(i) != material_index) {
            old_to_new_material_index[i] = new_materials.size();
            new_materials.push_back(materials[i]);
        }
    }

    if (new_materials.empty()) {
        spdlog::warn("Removing material '{}' would result in empty mesh, clearing all data", material_name);
        vertices.clear();
        normals.clear();
        indices.clear();
        triangle_materials.clear();
        materials.clear();
        return;
    }

    // Build new geometry
    std::vector<float3> new_vertices;
    std::vector<float3> new_normals;
    std::vector<uint3> new_indices;
    std::vector<int> new_triangle_materials;
    std::map<unsigned int, unsigned int> vertex_map;

    for (size_t tri_idx = 0; tri_idx < indices.size(); ++tri_idx) {
        int tri_material = triangle_materials[tri_idx];

        // Skip triangles with the material to remove
        if (tri_material == material_index) {
            continue;
        }

        const uint3& old_tri = indices[tri_idx];

        auto add_vertex = [&](unsigned int old_idx) -> unsigned int {
            auto it = vertex_map.find(old_idx);
            if (it != vertex_map.end()) {
                return it->second;
            }
            unsigned int new_idx = static_cast<unsigned int>(new_vertices.size());
            new_vertices.push_back(vertices[old_idx]);
            new_normals.push_back(normals[old_idx]);
            vertex_map[old_idx] = new_idx;
            return new_idx;
        };

        uint3 new_tri;
        new_tri.x = add_vertex(old_tri.x);
        new_tri.y = add_vertex(old_tri.y);
        new_tri.z = add_vertex(old_tri.z);
        new_indices.push_back(new_tri);

        int new_mat_idx = old_to_new_material_index[tri_material];
        new_triangle_materials.push_back(new_mat_idx);
    }

    // Replace mesh data
    vertices = std::move(new_vertices);
    normals = std::move(new_normals);
    indices = std::move(new_indices);
    triangle_materials = std::move(new_triangle_materials);
    materials = std::move(new_materials);

    spdlog::debug("Removed material '{}': mesh now has {} vertices, {} triangles",
                 material_name, vertices.size(), indices.size());
}

void Mesh::info() const {
    spdlog::info("========================================");
    spdlog::info("Mesh Statistics");
    spdlog::info("========================================");
    spdlog::info("Vertices: {}", vertices.size());
    spdlog::info("Triangles: {}", indices.size());
    spdlog::info("Materials: {}", materials.size());
    spdlog::info("");

    if (!materials.empty()) {
        spdlog::info("----------------------------------------");
        spdlog::info("Materials:");
        spdlog::info("----------------------------------------");

        for (const auto& mat : materials) {
            size_t tri_count = get_triangle_count_by_material(mat.name);
            size_t vert_count = get_vertex_count_by_material(mat.name);
            float percentage = (indices.size() > 0) ? (100.0f * tri_count / indices.size()) : 0.0f;

            spdlog::info("  '{}' (Kd: {:.2f}, {:.2f}, {:.2f})", 
                         mat.name, mat.diffuse.x, mat.diffuse.y, mat.diffuse.z);
            spdlog::info("    Triangles: {} ({:.1f}%)", tri_count, percentage);
            spdlog::info("    Vertices: {}", vert_count);
        }
    }

    spdlog::info("========================================");
}

void Mesh::translate(const float3& offset) {
    for (auto& v : vertices) {
        v.x += offset.x;
        v.y += offset.y;
        v.z += offset.z;
    }
}

void Mesh::rotate_y(float angle_degrees) {
    float angle_rad = angle_degrees * M_PI / 180.0f;
    float cos_a = std::cos(angle_rad);
    float sin_a = std::sin(angle_rad);

    for (auto& v : vertices) {
        float x = v.x * cos_a - v.z * sin_a;
        float z = v.x * sin_a + v.z * cos_a;
        v.x = x;
        v.z = z;
    }

    for (auto& n : normals) {
        float x = n.x * cos_a - n.z * sin_a;
        float z = n.x * sin_a + n.z * cos_a;
        n.x = x;
        n.z = z;
    }
}

void Mesh::scale(float factor) {
    for (auto& v : vertices) {
        v.x *= factor;
        v.y *= factor;
        v.z *= factor;
    }
}
