#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>

/**
 * @brief Triangle mesh with material information
 *
 * This structure contains pure geometric data and material name references.
 * Material factories are bound later during GPU scene build.
 */
struct Mesh {
    std::vector<float3> vertices;           // Vertex positions
    std::vector<float3> normals;            // Vertex normals
    std::vector<uint3> indices;             // Triangle indices (each uint3 = one triangle's 3 vertex indices)
    std::vector<int> triangle_materials;    // Material index per triangle (index into material_names)
    std::vector<std::string> material_names; // Material name list (e.g., ["wall", "detector", ...])

    // ========================================================================
    // Basic Statistics
    // ========================================================================

    /**
     * @brief Get number of triangles in the mesh
     * @return Triangle count
     */
    size_t get_triangle_count() const { return indices.size(); }

    /**
     * @brief Get number of vertices in the mesh
     * @return Vertex count
     */
    size_t get_vertex_count() const { return vertices.size(); }

    /**
     * @brief Get number of materials in the mesh
     * @return Material count
     */
    size_t get_material_count() const { return material_names.size(); }

    // ========================================================================
    // Material-based Queries
    // ========================================================================

    /**
     * @brief Get number of triangles with a specific material
     * @param material_name Material name to query
     * @return Number of triangles using this material, or 0 if material not found
     */
    size_t get_triangle_count_by_material(const std::string& material_name) const;

    /**
     * @brief Get number of unique vertices used by triangles with a specific material
     * @param material_name Material name to query
     * @return Number of unique vertices used by this material, or 0 if material not found
     *
     * Note: A vertex shared between multiple materials will be counted for each material.
     */
    size_t get_vertex_count_by_material(const std::string& material_name) const;

    /**
     * @brief Get all triangle indices that use a specific material
     * @param material_name Material name to query
     * @return Vector of triangle indices (0-based), empty if material not found
     */
    std::vector<size_t> get_triangle_indices_by_material(const std::string& material_name) const;

private:
    /**
     * @brief Find material index by name
     * @param material_name Material name to search for
     * @return Material index, or -1 if not found
     */
    int find_material_index(const std::string& material_name) const;
};
