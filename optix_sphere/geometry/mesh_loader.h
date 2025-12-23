#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

// 材质类型枚举
enum class MaterialType {
    SphereWall = 0,   // 普通反射壁
    Detector = 1,     // 探测器
    Baffle = 2,       // 挡板
    PortHole = 3,     // 开孔（完全吸收）
    Unknown = 4       // 未知材质
};

// 每个三角形的材质信息
struct TriangleMaterial {
    MaterialType type;
    int material_id;  // OBJ 文件中的原始材质 ID
};

// 加载的网格数据
struct LoadedMesh {
    std::vector<float3> vertices;   // 顶点位置
    std::vector<float3> normals;    // 顶点法线
    std::vector<uint3> indices;     // 三角形索引（每个 uint3 是一个三角形的三个顶点索引）
    std::vector<TriangleMaterial> triangle_materials;  // 每个三角形的材质

    // 材质名称到类型的映射（用于调试和统计）
    std::unordered_map<std::string, MaterialType> material_map;

    // 统计信息
    size_t get_triangle_count() const { return indices.size(); }
    size_t get_vertex_count() const { return vertices.size(); }
};

class MeshLoader {
public:
    // 加载 OBJ 文件
    // filepath: OBJ 文件路径
    // material_mapping: 材质名称到 MaterialType 的映射
    static LoadedMesh load_obj(
        const std::string& filepath,
        const std::unordered_map<std::string, MaterialType>& material_mapping
    );

    // 获取默认的材质名称映射（对应 Blender 中的材质名称）
    static std::unordered_map<std::string, MaterialType> get_default_material_mapping();
};
