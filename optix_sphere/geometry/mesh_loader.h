#pragma once

#include <string>
#include <vector>
#include <map>
#include <cuda_runtime.h>
#include "../material.h"

// 加载的网格数据
struct LoadedMesh {
    std::vector<float3> vertices;   // 顶点位置
    std::vector<float3> normals;    // 顶点法线
    std::vector<uint3> indices;     // 三角形索引（每个 uint3 是一个三角形的三个顶点索引）
    std::vector<int> triangle_materials;  // 每个三角形的材质索引（对应 material_factories 中的位置）

    // 材质工厂列表（按索引顺序，用于构建 OptiX SBT）
    std::vector<std::pair<std::string, MaterialFactory>> material_factories;

    // 统计信息
    size_t get_triangle_count() const { return indices.size(); }
    size_t get_vertex_count() const { return vertices.size(); }
    size_t get_material_count() const { return material_factories.size(); }
};

class MeshLoader {
public:
    // 加载 OBJ 文件
    // filepath: OBJ 文件路径
    // material_factories: 材质名称到 MaterialFactory 的映射（OBJ 材质名 -> 材质工厂函数）
    static LoadedMesh load_obj(
        const std::string& filepath,
        const std::map<std::string, MaterialFactory>& material_factories
    );

    // 获取默认的材质配置（from material::get_default_materials）
    static std::map<std::string, MaterialFactory> get_default_materials();
};
