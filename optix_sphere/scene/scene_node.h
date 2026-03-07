#pragma once

#include "geometry/mesh.h"
#include <cuda_runtime.h>
#include <string>
#include <memory>

/**
 * @brief Transform component for scene nodes
 */
struct Transform {
    float3 position = {0.0f, 0.0f, 0.0f};
    float3 rotation = {0.0f, 0.0f, 0.0f};  // Euler angles in degrees (X, Y, Z)
    float3 scale = {1.0f, 1.0f, 1.0f};

    // Apply transform to a vertex
    float3 apply_to_point(const float3& p) const;

    // Apply transform to a normal (no translation, no scaling)
    float3 apply_to_normal(const float3& n) const;
};

/**
 * @brief Scene node containing a mesh with transform and visibility
 *
 * Similar to Blender's object system - each node has:
 * - A mesh (geometry)
 * - Transform (position, rotation, scale)
 * - Name (for identification)
 * - Enabled flag (for visibility/rendering control)
 */
class SceneNode {
public:
    // Constructors
    SceneNode() = default;
    SceneNode(std::string name, Mesh mesh);

    // Properties
    std::string name;
    Mesh mesh;
    Transform transform;
    bool enabled = true;

    // Get transformed mesh (applies transform to mesh)
    Mesh get_transformed_mesh() const;

    // Convenience methods
    void set_position(const float3& pos) { transform.position = pos; }
    void set_rotation(const float3& rot) { transform.rotation = rot; }
    void set_scale(const float3& s) { transform.scale = s; }
    void set_uniform_scale(float s) { transform.scale = {s, s, s}; }

    // Quick transforms
    void translate(const float3& offset);
    void rotate_y(float angle_degrees);
    void scale_uniform(float factor);
};
