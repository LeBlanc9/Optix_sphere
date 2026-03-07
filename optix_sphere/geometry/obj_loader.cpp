#include "obj_loader.h"
#include <tiny_obj_loader.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <map>

std::map<std::string, MaterialFactory> ObjLoader::get_default_materials() {
    return material::get_default_materials();
}

Mesh ObjLoader::load_obj(const std::string& filepath) {
    spdlog::info("Loading OBJ file: {}", filepath);

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // 获取 OBJ 文件所在目录（用于加载 MTL 文件）
    std::string mtl_base_dir = filepath.substr(0, filepath.find_last_of("/\\") + 1);
    if (mtl_base_dir.empty()) {
        mtl_base_dir = "./";
    }

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                filepath.c_str(), mtl_base_dir.c_str());

    if (!warn.empty()) {
        spdlog::warn("OBJ loader warning: {}", warn);
    }

    if (!err.empty()) {
        spdlog::error("OBJ loader error: {}", err);
    }

    if (!ret) {
        throw std::runtime_error("Failed to load OBJ file: " + filepath);
    }

    Mesh mesh;

    // Build material name list and index mapping
    // Map: OBJ material name -> our internal material index
    std::map<std::string, int> material_name_to_index;

    for (const auto& obj_mat : materials) {
        std::string mat_name = obj_mat.name;

        // Check if we've already added this material
        if (material_name_to_index.find(mat_name) == material_name_to_index.end()) {
            int new_index = static_cast<int>(mesh.material_names.size());
            material_name_to_index[mat_name] = new_index;
            mesh.material_names.push_back(mat_name);
            spdlog::debug("Material '{}' -> index {}", mat_name, new_index);
        }
    }

    // 遍历所有形状（shapes）
    for (const auto& shape : shapes) {
        size_t index_offset = 0;

        // 遍历所有面（faces）
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            int fv = shape.mesh.num_face_vertices[f];

            if (fv != 3) {
                throw std::runtime_error(
                    "Non-triangle face detected! Please triangulate the mesh in Blender. "
                    "Face " + std::to_string(f) + " has " + std::to_string(fv) + " vertices."
                );
            }

            // 获取该三角形的材质 ID
            int material_id = shape.mesh.material_ids[f];

            // 确定材质索引
            int mat_index = 0;  // Default to first material
            if (material_id >= 0 && material_id < static_cast<int>(materials.size())) {
                std::string mat_name = materials[material_id].name;
                auto it = material_name_to_index.find(mat_name);
                if (it != material_name_to_index.end()) {
                    mat_index = it->second;
                }
            } else {
                // Face has no material assigned in OBJ file
                // Use the first available material as default
                if (!mesh.material_names.empty()) {
                    mat_index = 0;  // Use first material
                }
                // Only warn once (on first occurrence)
                static bool warned_once = false;
                if (!warned_once) {
                    spdlog::warn("Some faces have no material assigned in OBJ, using first material as default");
                    warned_once = true;
                }
            }

            // 存储三角形的三个顶点
            uint3 triangle_indices;
            for (int v = 0; v < 3; v++) {
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                // 顶点位置
                float3 vertex = make_float3(
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2]
                );

                // 法线（如果有）
                float3 normal = make_float3(0, 0, 0);
                if (idx.normal_index >= 0) {
                    normal = make_float3(
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2]
                    );
                }

                mesh.vertices.push_back(vertex);
                mesh.normals.push_back(normal);

                // 记录索引
                unsigned int vertex_index = static_cast<unsigned int>(mesh.vertices.size() - 1);
                if (v == 0) triangle_indices.x = vertex_index;
                else if (v == 1) triangle_indices.y = vertex_index;
                else triangle_indices.z = vertex_index;
            }

            mesh.indices.push_back(triangle_indices);
            mesh.triangle_materials.push_back(mat_index);

            index_offset += fv;
        }
    }

    return mesh;
}

