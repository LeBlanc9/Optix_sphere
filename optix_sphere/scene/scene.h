#pragma once

#include "geometry/mesh_loader.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <utility>

/**
 * @brief CPU-side scene representation (loaded mesh data before GPU build)
 *
 * This class holds the mesh data loaded from OBJ files in CPU memory.
 * It provides query methods to inspect the scene before building GPU structures.
 */
class Scene {
public:
    /**
     * @brief Load scene from OBJ file (static factory method)
     * @param file_path Path to OBJ file
     * @return Scene object with loaded mesh data
     */
    static Scene from_obj(const std::string& file_path);

    /**
     * @brief Create scene from existing mesh data (static factory method)
     * @param mesh Mesh data to wrap in a Scene
     * @return Scene object containing the mesh
     *
     * @example
     * ```cpp
     * Mesh procedural_mesh = generate_sphere_mesh(radius);
     * Scene scene = Scene::from_mesh(std::move(procedural_mesh));
     * ```
     */
    static Scene from_mesh(Mesh mesh) {
        return Scene(std::move(mesh));
    }

    /**
     * @brief Construct Scene from loaded mesh data
     * @param mesh Loaded mesh data from OBJ file
     */
    explicit Scene(Mesh&& mesh);

    /**
     * @brief Get all material names found in the mesh
     * @return Vector of material names from OBJ file
     */
    std::vector<std::string> get_material_names() const;

    /**
     * @brief Get number of vertices in the mesh
     * @return Vertex count
     */
    size_t get_vertex_count() const;

    /**
     * @brief Get number of triangles in the mesh
     * @return Triangle count
     */
    size_t get_triangle_count() const;

    /**
     * @brief Get number of materials in the mesh
     * @return Material count
     */
    size_t get_material_count() const;

    /**
     * @brief Get axis-aligned bounding box of the entire mesh
     * @return Pair of (min_corner, max_corner) in mm
     */
    std::pair<float3, float3> get_bounds() const;

    /**
     * @brief Get the underlying mesh data (for Simulator to build GPU scene)
     * @return Reference to the loaded mesh
     */
    const Mesh& get_mesh() const { return mesh_; }

private:
    Mesh mesh_;
};
