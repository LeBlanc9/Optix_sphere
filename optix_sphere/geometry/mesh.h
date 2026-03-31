#pragma once

#include <string>
#include <vector>
#include <map>
#include <cuda_runtime.h>

/**
 * @brief Material information for a mesh
 */
struct MeshMaterial {
    std::string name;
    float3 diffuse = {0.8f, 0.8f, 0.8f};
    float3 ambient = {0.0f, 0.0f, 0.0f};
    float3 specular = {0.0f, 0.0f, 0.0f};
};

/**
 * @brief Triangle mesh with material information
 */
struct Mesh {
    std::vector<float3> vertices;           // Vertex positions
    std::vector<float3> normals;            // Vertex normals
    std::vector<uint3> indices;             // Triangle indices (each uint3 = one triangle's 3 vertex indices)
    std::vector<int> triangle_materials;    // Material index per triangle (index into materials)
    std::vector<MeshMaterial> materials;    // Material properties list

    // Constructors
    Mesh() = default;
    Mesh(const Mesh&) = default;
    Mesh(Mesh&&) = default;
    Mesh& operator=(const Mesh&) = default;
    Mesh& operator=(Mesh&&) = default;

    static Mesh from_obj(const std::string& file_path);

    /**
     * @brief Export mesh to OBJ and MTL files
     * @param obj_file_path Output OBJ path
     * @param mtl_file_path Output MTL path. If empty, uses <obj_basename>.mtl in the same directory.
     */
    void to_obj(const std::string& obj_file_path, const std::string& mtl_file_path = "") const;

    // ========================================================================
    // Basic Statistics
    // ========================================================================

    size_t get_triangle_count() const { return indices.size(); }

    size_t get_vertex_count() const { return vertices.size(); }

    size_t get_material_count() const { return materials.size(); }

    std::pair<float3, float3> get_bounds() const;

    // ========================================================================
    // Material-based Queries
    // ========================================================================

   size_t get_triangle_count_by_material(const std::string& material_name) const;

   size_t get_vertex_count_by_material(const std::string& material_name) const;

    // ========================================================================
    // Mesh Splitting
    // ========================================================================

    std::map<std::string, Mesh> split_by_material() const;

    Mesh extract_mesh_by_material(const std::string& material_name) const;

    void remove_mesh_by_material(const std::string& material_name);

    // ========================================================================
    // Transformation
    // ========================================================================

    void translate(const float3& offset);

    void rotate_y(float angle_degrees);

    void scale(float factor);

    // ========================================================================
    // Debugging and Statistics
    // ========================================================================

    void info() const;

private:
    /**
     * @brief Find material index by name
     * @param material_name Material name to search for
     * @return Material index, or -1 if not found
     */
    int find_material_index(const std::string& material_name) const;
};
