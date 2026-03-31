#include "scene.h"
#include "geometry/obj_loader.h"
#include <spdlog/spdlog.h>
#include <limits>
#include <algorithm>
#include <set>

Scene Scene::from_obj(const std::string& file_path) {
    spdlog::info("Loading scene from OBJ file: {}", file_path);

    Mesh mesh = ObjLoader::load_obj(file_path);

    spdlog::info("✅ Scene loaded to CPU:");
    spdlog::info("   Vertices: {}", mesh.get_vertex_count());
    spdlog::info("   Materials: {}", mesh.get_material_count());

    // Create a single node from the loaded mesh
    Scene scene;
    scene.add_node("root", std::move(mesh));

    return scene;
}

Scene Scene::from_mesh(Mesh mesh) {
    Scene scene;
    scene.add_node("root", std::move(mesh));
    return scene;
}

void Scene::add_node(const std::string& name, Mesh mesh) {
    nodes_.emplace_back(name, std::move(mesh));
    invalidate_cache();
    spdlog::info("Added node '{}' with {} triangles", name, nodes_.back().mesh.get_triangle_count());
}

void Scene::add_node(SceneNode node) {
    spdlog::info("Added node '{}' with {} triangles", node.name, node.mesh.get_triangle_count());
    nodes_.push_back(std::move(node));
    invalidate_cache();
}

SceneNode* Scene::get_node(const std::string& name) {
    for (auto& node : nodes_) {
        if (node.name == name) {
            return &node;
        }
    }
    return nullptr;
}

const SceneNode* Scene::get_node(const std::string& name) const {
    for (const auto& node : nodes_) {
        if (node.name == name) {
            return &node;
        }
    }
    return nullptr;
}

SceneNode* Scene::get_node(size_t index) {
    if (index >= nodes_.size()) {
        return nullptr;
    }
    return &nodes_[index];
}

const SceneNode* Scene::get_node(size_t index) const {
    if (index >= nodes_.size()) {
        return nullptr;
    }
    return &nodes_[index];
}

bool Scene::has_node(const std::string& name) const {
    return get_node(name) != nullptr;
}

void Scene::enable_node(const std::string& name) {
    SceneNode* node = get_node(name);
    if (node) {
        node->enabled = true;
        invalidate_cache();
        spdlog::info("Enabled node '{}'", name);
    } else {
        spdlog::warn("Node '{}' not found", name);
    }
}

void Scene::disable_node(const std::string& name) {
    SceneNode* node = get_node(name);
    if (node) {
        node->enabled = false;
        invalidate_cache();
        spdlog::info("Disabled node '{}'", name);
    } else {
        spdlog::warn("Node '{}' not found", name);
    }
}

void Scene::enable_node(size_t index) {
    SceneNode* node = get_node(index);
    if (node) {
        node->enabled = true;
        invalidate_cache();
        spdlog::info("Enabled node [{}] '{}'", index, node->name);
    } else {
        spdlog::warn("Node index {} out of range", index);
    }
}

void Scene::disable_node(size_t index) {
    SceneNode* node = get_node(index);
    if (node) {
        node->enabled = false;
        invalidate_cache();
        spdlog::info("Disabled node [{}] '{}'", index, node->name);
    } else {
        spdlog::warn("Node index {} out of range", index);
    }
}

bool Scene::is_node_enabled(const std::string& name) const {
    const SceneNode* node = get_node(name);
    return node ? node->enabled : false;
}

bool Scene::is_node_enabled(size_t index) const {
    const SceneNode* node = get_node(index);
    return node ? node->enabled : false;
}

Mesh Scene::get_merged_mesh() const {
    if (merged_cache_valid_) {
        return merged_cache_;
    }

    Mesh merged;

    // Map: (old_node_index, old_vertex_index) -> new_vertex_index
    std::map<std::pair<size_t, unsigned int>, unsigned int> vertex_map;

    // Material name -> material index in merged mesh
    std::map<std::string, int> material_map;

    for (size_t node_idx = 0; node_idx < nodes_.size(); ++node_idx) {
        const SceneNode& node = nodes_[node_idx];

        if (!node.enabled) {
            continue;  // Skip disabled nodes
        }

        // Get transformed mesh
        Mesh transformed = node.get_transformed_mesh();

        // Add materials from this node
        for (const auto& mat : transformed.materials) {
            if (material_map.find(mat.name) == material_map.end()) {
                int new_mat_idx = static_cast<int>(merged.materials.size());
                merged.materials.push_back(mat);
                material_map[mat.name] = new_mat_idx;
            }
        }

        // Add triangles
        for (size_t tri_idx = 0; tri_idx < transformed.indices.size(); ++tri_idx) {
            const uint3& old_tri = transformed.indices[tri_idx];

            auto add_vertex = [&](unsigned int old_idx) -> unsigned int {
                auto key = std::make_pair(node_idx, old_idx);
                auto it = vertex_map.find(key);
                if (it != vertex_map.end()) {
                    return it->second;
                }

                unsigned int new_idx = static_cast<unsigned int>(merged.vertices.size());
                merged.vertices.push_back(transformed.vertices[old_idx]);
                merged.normals.push_back(transformed.normals[old_idx]);
                vertex_map[key] = new_idx;
                return new_idx;
            };

            uint3 new_tri;
            new_tri.x = add_vertex(old_tri.x);
            new_tri.y = add_vertex(old_tri.y);
            new_tri.z = add_vertex(old_tri.z);
            merged.indices.push_back(new_tri);

            // Map material index
            int old_mat_idx = transformed.triangle_materials[tri_idx];
            const std::string& mat_name = transformed.materials[old_mat_idx].name;
            int new_mat_idx = material_map[mat_name];
            merged.triangle_materials.push_back(new_mat_idx);
        }
    }

    spdlog::debug("Merged {} enabled nodes into single mesh: {} vertices, {} triangles",
                  nodes_.size(), merged.vertices.size(), merged.indices.size());

    merged_cache_ = merged;
    merged_cache_valid_ = true;

    return merged;
}

const Mesh& Scene::get_mesh() const {
    if (!merged_cache_valid_) {
        merged_cache_ = get_merged_mesh();
    }
    return merged_cache_;
}

std::vector<std::string> Scene::get_material_names() const {
    std::set<std::string> unique_materials;

    for (const auto& node : nodes_) {
        if (node.enabled) {
            for (const auto& mat : node.mesh.materials) {
                unique_materials.insert(mat.name);
            }
        }
    }

    return std::vector<std::string>(unique_materials.begin(), unique_materials.end());
}

size_t Scene::get_vertex_count() const {
    return get_mesh().get_vertex_count();
}

size_t Scene::get_triangle_count() const {
    size_t count = 0;
    for (const auto& node : nodes_) {
        if (node.enabled) {
            count += node.mesh.get_triangle_count();
        }
    }
    return count;
}

size_t Scene::get_material_count() const {
    return get_material_names().size();
}

std::pair<float3, float3> Scene::get_bounds() const {
    if (nodes_.empty()) {
        return {make_float3(0, 0, 0), make_float3(0, 0, 0)};
    }

    float3 min_bounds = make_float3(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    );
    float3 max_bounds = make_float3(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    );

    bool has_enabled = false;

    for (const auto& node : nodes_) {
        if (!node.enabled) continue;
        has_enabled = true;

        Mesh transformed = node.get_transformed_mesh();
        for (const auto& v : transformed.vertices) {
            min_bounds.x = std::min(min_bounds.x, v.x);
            min_bounds.y = std::min(min_bounds.y, v.y);
            min_bounds.z = std::min(min_bounds.z, v.z);

            max_bounds.x = std::max(max_bounds.x, v.x);
            max_bounds.y = std::max(max_bounds.y, v.y);
            max_bounds.z = std::max(max_bounds.z, v.z);
        }
    }

    if (!has_enabled) {
        return {make_float3(0, 0, 0), make_float3(0, 0, 0)};
    }

    return {min_bounds, max_bounds};
}

void Scene::rotate_y(float angle_degrees) {
    spdlog::info("Rotating all scene nodes around Y-axis by {} degrees", angle_degrees);

    for (auto& node : nodes_) {
        node.rotate_y(angle_degrees);
    }

    invalidate_cache();
}

void Scene::translate(const float3& offset) {
    spdlog::info("Translating all scene nodes by ({}, {}, {})", offset.x, offset.y, offset.z);

    for (auto& node : nodes_) {
        node.translate(offset);
    }

    invalidate_cache();
}

void Scene::scale(float scale_factor) {
    if (scale_factor <= 0.0f) {
        spdlog::error("Invalid scale factor: {} (must be positive)", scale_factor);
        return;
    }

    spdlog::info("Scaling all scene nodes by factor {}", scale_factor);

    for (auto& node : nodes_) {
        node.scale_uniform(scale_factor);
    }

    invalidate_cache();
}

size_t Scene::get_triangle_count_by_material(const std::string& material_name) const {
    return get_mesh().get_triangle_count_by_material(material_name);
}

size_t Scene::get_vertex_count_by_material(const std::string& material_name) const {
    return get_mesh().get_vertex_count_by_material(material_name);
}

void Scene::info() const {
    spdlog::info("========================================");
    spdlog::info("Scene Statistics");
    spdlog::info("========================================");
    spdlog::info("Total Nodes: {}", nodes_.size());

    size_t enabled_count = 0;
    size_t disabled_count = 0;
    size_t total_triangles = 0;
    size_t total_vertices = 0;

    for (const auto& node : nodes_) {
        if (node.enabled) {
            enabled_count++;
            total_triangles += node.mesh.get_triangle_count();
            total_vertices += node.mesh.get_vertex_count();
        } else {
            disabled_count++;
        }
    }

    spdlog::info("  Enabled: {}", enabled_count);
    spdlog::info("  Disabled: {}", disabled_count);
    spdlog::info("");
    spdlog::info("Total Triangles (enabled): {}", total_triangles);
    spdlog::info("Total Vertices (enabled): {}", total_vertices);
    spdlog::info("Total Materials: {}", get_material_count());
    spdlog::info("");
    spdlog::info("----------------------------------------");
    spdlog::info("Scene Nodes:");
    spdlog::info("----------------------------------------");

    for (const auto& node : nodes_) {
        std::string status = node.enabled ? "✓" : "✗";
        spdlog::info("{} Node: '{}'", status, node.name);
        spdlog::info("    Triangles: {}", node.mesh.get_triangle_count());
        spdlog::info("    Vertices: {}", node.mesh.get_vertex_count());
        spdlog::info("    Materials: {}", node.mesh.get_material_count());

        // List materials and triangle counts
        if (node.mesh.get_material_count() > 0) {
            for (const auto& mat : node.mesh.materials) {
                size_t mat_tri_count = node.mesh.get_triangle_count_by_material(mat.name);
                spdlog::info("      - '{}': {} triangles", mat.name, mat_tri_count);
            }
        }

        // Show transform if non-identity
        bool has_transform =
            node.transform.position.x != 0.0f || node.transform.position.y != 0.0f || node.transform.position.z != 0.0f ||
            node.transform.rotation.x != 0.0f || node.transform.rotation.y != 0.0f || node.transform.rotation.z != 0.0f ||
            node.transform.scale.x != 1.0f || node.transform.scale.y != 1.0f || node.transform.scale.z != 1.0f;

        if (has_transform) {
            spdlog::info("    Transform:");
            if (node.transform.position.x != 0.0f || node.transform.position.y != 0.0f || node.transform.position.z != 0.0f) {
                spdlog::info("      Position: ({:.3f}, {:.3f}, {:.3f})",
                    node.transform.position.x, node.transform.position.y, node.transform.position.z);
            }
            if (node.transform.rotation.x != 0.0f || node.transform.rotation.y != 0.0f || node.transform.rotation.z != 0.0f) {
                spdlog::info("      Rotation: ({:.1f}°, {:.1f}°, {:.1f}°)",
                    node.transform.rotation.x, node.transform.rotation.y, node.transform.rotation.z);
            }
            if (node.transform.scale.x != 1.0f || node.transform.scale.y != 1.0f || node.transform.scale.z != 1.0f) {
                spdlog::info("      Scale: ({:.3f}, {:.3f}, {:.3f})",
                    node.transform.scale.x, node.transform.scale.y, node.transform.scale.z);
            }
        }
        spdlog::info("");
    }

    spdlog::info("========================================");
}
