#include "mesh_loader.h"
#include <tiny_obj_loader.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_map>

std::unordered_map<std::string, MaterialType> MeshLoader::get_default_material_mapping() {
    return {
        // 标准名称
        {"Sphere_Wall", MaterialType::SphereWall},
        {"SphereWall", MaterialType::SphereWall},
        {"Detector", MaterialType::Detector},
        {"Baffle", MaterialType::Baffle},
        {"Port_Hole", MaterialType::PortHole},
        {"PortHole", MaterialType::PortHole},

        // 常见别名
        {"wall_material", MaterialType::SphereWall},
        {"detector_material", MaterialType::Detector},
        {"baffle_material", MaterialType::Baffle},
        {"porthole_material", MaterialType::PortHole},
    };
}

LoadedMesh MeshLoader::load_obj(
    const std::string& filepath,
    const std::unordered_map<std::string, MaterialType>& material_mapping
) {
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

    spdlog::info("✅ Loaded OBJ: {} vertices, {} shapes, {} materials",
                 attrib.vertices.size() / 3, shapes.size(), materials.size());

    LoadedMesh mesh;
    mesh.material_map = material_mapping;

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

            // 确定材质类型
            MaterialType mat_type = MaterialType::Unknown;
            if (material_id >= 0 && material_id < static_cast<int>(materials.size())) {
                std::string mat_name = materials[material_id].name;
                auto it = material_mapping.find(mat_name);
                if (it != material_mapping.end()) {
                    mat_type = it->second;
                } else {
                    spdlog::warn("Unknown material '{}', treating as SphereWall", mat_name);
                    mat_type = MaterialType::SphereWall;
                }
            } else {
                spdlog::warn("Face {} has no material assigned, treating as SphereWall", f);
                mat_type = MaterialType::SphereWall;
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
            mesh.triangle_materials.push_back({mat_type, material_id});

            index_offset += fv;
        }
    }

    spdlog::info("✅ Processed mesh: {} vertices, {} triangles",
                 mesh.vertices.size(), mesh.indices.size());

    // 统计各材质的三角形数量
    std::unordered_map<MaterialType, int> mat_counts;
    for (const auto& tri_mat : mesh.triangle_materials) {
        mat_counts[tri_mat.type]++;
    }

    spdlog::info("Material distribution:");
    const char* mat_names[] = {"SphereWall", "Detector", "Baffle", "PortHole", "Unknown"};
    for (const auto& [mat_type, count] : mat_counts) {
        int mat_idx = static_cast<int>(mat_type);
        if (mat_idx >= 0 && mat_idx < 5) {
            spdlog::info("  {}: {} triangles", mat_names[mat_idx], count);
        }
    }

    return mesh;
}
