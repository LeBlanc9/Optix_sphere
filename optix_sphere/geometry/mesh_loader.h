#pragma once

#include <string>
#include <vector>
#include <map>
#include <cuda_runtime.h>
#include "../material.h"

/**
 * @brief Pure mesh geometry data with material names (no material instances)
 *
 * This structure contains only geometric data and material name references
 * loaded from mesh files. Material factories are bound later during GPU scene build.
 */
struct Mesh {
    std::vector<float3> vertices;   // 顶点位置
    std::vector<float3> normals;    // 顶点法线
    std::vector<uint3> indices;     // 三角形索引（每个 uint3 是一个三角形的三个顶点索引）
    std::vector<int> triangle_materials;  // 每个三角形的材质索引（对应 material_names 中的位置）

    // 材质名称列表（从 OBJ 文件读取，按索引顺序）
    std::vector<std::string> material_names;  // ["wall", "detector", ...]

    // 统计信息
    size_t get_triangle_count() const { return indices.size(); }
    size_t get_vertex_count() const { return vertices.size(); }
    size_t get_material_count() const { return material_names.size(); }
};

class MeshLoader {
public:
    /**
     * @brief Load mesh geometry and material names from OBJ file
     * @param filepath Path to OBJ file
     * @return Mesh with geometry and material name references
     */
    static Mesh load_obj(const std::string& filepath);

    // 获取默认的材质配置（from material::get_default_materials）
    static std::map<std::string, MaterialFactory> get_default_materials();
};
