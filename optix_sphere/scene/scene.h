#pragma once

#include "scene_node.h"
#include "geometry/obj_loader.h"
#include <vector>
#include <string>
#include <map>
#include <utility>

/**
 * @brief Scene graph containing multiple nodes
 */
class Scene {
public:
    // Factory methods
    static Scene from_obj(const std::string& file_path);
    static Scene from_mesh(Mesh mesh);

    // Constructors
    Scene() = default;

    // Node management
    void add_node(const std::string& name, Mesh mesh);
    void add_node(SceneNode node);

    SceneNode* get_node(const std::string& name);
    const SceneNode* get_node(const std::string& name) const;
    SceneNode* get_node(size_t index);
    const SceneNode* get_node(size_t index) const;

    size_t get_node_count() const { return nodes_.size(); }
    const std::vector<SceneNode>& get_nodes() const { return nodes_; }

    bool has_node(const std::string& name) const;

    // Node control
    void enable_node(const std::string& name);
    void disable_node(const std::string& name);
    void enable_node(size_t index);
    void disable_node(size_t index);
    bool is_node_enabled(const std::string& name) const;
    bool is_node_enabled(size_t index) const;

    // Get merged mesh (only enabled nodes, with transforms applied)
    Mesh get_merged_mesh() const;
    const Mesh& get_mesh() const;  // For backward compatibility (returns cached merged mesh)

    // Queries (aggregated across enabled nodes)
    std::vector<std::string> get_material_names() const;
    size_t get_vertex_count() const;
    size_t get_triangle_count() const;
    size_t get_material_count() const;
    std::pair<float3, float3> get_bounds() const;

    // Global transformations (apply to all nodes)
    void rotate_y(float angle_degrees);
    void translate(const float3& offset);
    void scale(float scale_factor);

    // Material-based queries (convenience methods)
    size_t get_triangle_count_by_material(const std::string& material_name) const;
    size_t get_vertex_count_by_material(const std::string& material_name) const;

    // Statistics and debugging
    void info() const;

private:
    std::vector<SceneNode> nodes_;
    mutable Mesh merged_cache_;
    mutable bool merged_cache_valid_ = false;

    void invalidate_cache() const { merged_cache_valid_ = false; }
};
