#pragma once

#include "geometry/obj_loader.h"
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

    /**
     * @brief Rotate the entire scene around Y-axis
     * @param angle_degrees Rotation angle in degrees (positive = counter-clockwise from top view)
     *
     * @example
     * ```cpp
     * Scene scene = Scene::from_obj("sphere.obj");
     * scene.rotate_y(180.0f);  // Flip around Y-axis
     * ```
     */
    void rotate_y(float angle_degrees);

    /**
     * @brief Translate the entire scene
     * @param offset Translation vector (x, y, z) in mm
     *
     * @example
     * ```cpp
     * Scene scene = Scene::from_obj("sphere.obj");
     * scene.translate(make_float3(10.0f, 0.0f, 0.0f));  // Move 10mm in X direction
     * ```
     */
    void translate(const float3& offset);

    /**
     * @brief Scale the entire scene uniformly
     * @param scale_factor Scaling factor (1.0 = no change, 2.0 = double size)
     */
    void scale(float scale_factor);

    /**
     * @brief Get number of triangles with a specific material
     * @param material_name Material name to query
     * @return Number of triangles using this material, or 0 if material not found
     *
     * @example
     * ```cpp
     * Scene scene = Scene::from_obj("sphere.obj");
     * size_t wall_triangles = scene.get_triangle_count_by_material("wall_material");
     * size_t detector_triangles = scene.get_triangle_count_by_material("detector_material");
     * ```
     */
    size_t get_triangle_count_by_material(const std::string& material_name) const {
        return mesh_.get_triangle_count_by_material(material_name);
    }

    /**
     * @brief Get number of unique vertices used by triangles with a specific material
     * @param material_name Material name to query
     * @return Number of unique vertices used by this material, or 0 if material not found
     *
     * Note: This counts unique vertex indices referenced by triangles of this material.
     * A vertex shared between multiple materials will be counted for each material.
     *
     * @example
     * ```cpp
     * Scene scene = Scene::from_obj("sphere.obj");
     * size_t detector_vertices = scene.get_vertex_count_by_material("detector_material");
     * ```
     */
    size_t get_vertex_count_by_material(const std::string& material_name) const {
        return mesh_.get_vertex_count_by_material(material_name);
    }

    /**
     * @brief Get all triangle indices that use a specific material
     * @param material_name Material name to query
     * @return Vector of triangle indices (0-based), empty if material not found
     *
     * @example
     * ```cpp
     * Scene scene = Scene::from_obj("sphere.obj");
     * auto detector_triangles = scene.get_triangle_indices_by_material("detector_material");
     * // detector_triangles[0] is the first triangle using detector_material
     * ```
     */
    std::vector<size_t> get_triangle_indices_by_material(const std::string& material_name) const {
        return mesh_.get_triangle_indices_by_material(material_name);
    }

private:
    Mesh mesh_;
};
