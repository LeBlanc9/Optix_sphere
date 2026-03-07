#include "scene_node.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SceneNode::SceneNode(std::string name, Mesh mesh)
    : name(std::move(name)), mesh(std::move(mesh)) {}

float3 Transform::apply_to_point(const float3& p) const {
    // Apply in order: Scale -> Rotate -> Translate
    float3 result = p;

    // 1. Scale
    result.x *= scale.x;
    result.y *= scale.y;
    result.z *= scale.z;

    // 2. Rotate (Euler angles: X -> Y -> Z order)
    // Rotate X
    if (rotation.x != 0.0f) {
        float rad = rotation.x * M_PI / 180.0f;
        float cos_a = std::cos(rad);
        float sin_a = std::sin(rad);
        float y_new = cos_a * result.y - sin_a * result.z;
        float z_new = sin_a * result.y + cos_a * result.z;
        result.y = y_new;
        result.z = z_new;
    }

    // Rotate Y
    if (rotation.y != 0.0f) {
        float rad = rotation.y * M_PI / 180.0f;
        float cos_a = std::cos(rad);
        float sin_a = std::sin(rad);
        float x_new = cos_a * result.x + sin_a * result.z;
        float z_new = -sin_a * result.x + cos_a * result.z;
        result.x = x_new;
        result.z = z_new;
    }

    // Rotate Z
    if (rotation.z != 0.0f) {
        float rad = rotation.z * M_PI / 180.0f;
        float cos_a = std::cos(rad);
        float sin_a = std::sin(rad);
        float x_new = cos_a * result.x - sin_a * result.y;
        float y_new = sin_a * result.x + cos_a * result.y;
        result.x = x_new;
        result.y = y_new;
    }

    // 3. Translate
    result.x += position.x;
    result.y += position.y;
    result.z += position.z;

    return result;
}

float3 Transform::apply_to_normal(const float3& n) const {
    // Normals: only rotation, no translation or scale
    float3 result = n;

    // Rotate X
    if (rotation.x != 0.0f) {
        float rad = rotation.x * M_PI / 180.0f;
        float cos_a = std::cos(rad);
        float sin_a = std::sin(rad);
        float y_new = cos_a * result.y - sin_a * result.z;
        float z_new = sin_a * result.y + cos_a * result.z;
        result.y = y_new;
        result.z = z_new;
    }

    // Rotate Y
    if (rotation.y != 0.0f) {
        float rad = rotation.y * M_PI / 180.0f;
        float cos_a = std::cos(rad);
        float sin_a = std::sin(rad);
        float x_new = cos_a * result.x + sin_a * result.z;
        float z_new = -sin_a * result.x + cos_a * result.z;
        result.x = x_new;
        result.z = z_new;
    }

    // Rotate Z
    if (rotation.z != 0.0f) {
        float rad = rotation.z * M_PI / 180.0f;
        float cos_a = std::cos(rad);
        float sin_a = std::sin(rad);
        float x_new = cos_a * result.x - sin_a * result.y;
        float y_new = sin_a * result.x + cos_a * result.y;
        result.x = x_new;
        result.y = y_new;
    }

    return result;
}

Mesh SceneNode::get_transformed_mesh() const {
    Mesh result = mesh;  // Copy

    // Apply transform to all vertices and normals
    for (auto& v : result.vertices) {
        v = transform.apply_to_point(v);
    }

    for (auto& n : result.normals) {
        n = transform.apply_to_normal(n);
    }

    return result;
}

void SceneNode::translate(const float3& offset) {
    transform.position.x += offset.x;
    transform.position.y += offset.y;
    transform.position.z += offset.z;
}

void SceneNode::rotate_y(float angle_degrees) {
    transform.rotation.y += angle_degrees;
}

void SceneNode::scale_uniform(float factor) {
    transform.scale.x *= factor;
    transform.scale.y *= factor;
    transform.scale.z *= factor;
}
