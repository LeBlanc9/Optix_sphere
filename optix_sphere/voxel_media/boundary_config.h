#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <string>

namespace phonder::voxel {

/**
 * @brief Boundary face enumeration
 */
enum class BoundaryFace : uint8_t {
    NegX = 0,  // -X face
    NegY = 1,  // -Y face
    NegZ = 2,  // -Z face
    PosX = 3,  // +X face
    PosY = 4,  // +Y face
    PosZ = 5   // +Z face
};

/**
 * @brief Configuration for boundary photon collection
 *
 * Default behavior: When photon hits boundary, it undergoes Fresnel reflection/refraction
 *                   with the ambient medium. The photon may reflect back
 *                   into the grid or refract out into the ambient medium.
 *
 * Optional collection: User can enable collection on specific boundaries to capture
 *                      photons that exit through those faces (for analysis).
 *                      Collection does NOT affect the physics - it just records the photon.
 *
 * Example usage:
 *   BoundaryCollectionConfig bc;
 *   bc.enable_positive_faces();           // Collect photons exiting from +X/+Y/+Z
 *   bc.enable_negative_faces();           // Collect photons exiting from -X/-Y/-Z
 *   bc.enable_face(BoundaryFace::PosZ);   // Collect only from +Z (transmitted)
 *   bc.enable_z_faces();                  // Collect from ±Z (reflected + transmitted)
 */
struct BoundaryCollectionConfig {
    // Which faces collect photons
    bool collect_faces[6] = {
        false,  // -X
        false,  // -Y
        false,  // -Z
        false,  // +X
        false,  // +Y
        false   // +Z
    };

    // Collection radii for spatial filtering (negative = no limit)
    float collection_radius_positive = -1.0f;  // Radius for +X/+Y/+Z faces
    float collection_radius_negative = -1.0f;  // Radius for -X/-Y/-Z faces

    // Center point for radius calculation (default: grid center)
    float3 collection_center = make_float3(0.0f, 0.0f, 0.0f);
    bool use_grid_center = true;  // Auto-set center to grid center

    /**
     * @brief Enable collection on a specific face
     */
    void enable_face(BoundaryFace face) {
        collect_faces[static_cast<int>(face)] = true;
    }

    /**
     * @brief Disable collection on a specific face
     */
    void disable_face(BoundaryFace face) {
        collect_faces[static_cast<int>(face)] = false;
    }

    /**
     * @brief Enable collection on all positive faces (+X, +Y, +Z)
     */
    void enable_positive_faces() {
        collect_faces[3] = true;  // +X
        collect_faces[4] = true;  // +Y
        collect_faces[5] = true;  // +Z
    }

    /**
     * @brief Enable collection on all negative faces (-X, -Y, -Z)
     */
    void enable_negative_faces() {
        collect_faces[0] = true;  // -X
        collect_faces[1] = true;  // -Y
        collect_faces[2] = true;  // -Z
    }

    /**
     * @brief Enable collection on all faces
     */
    void enable_all_faces() {
        for (int i = 0; i < 6; i++) {
            collect_faces[i] = true;
        }
    }

    /**
     * @brief Enable collection on ±X faces
     */
    void enable_x_faces() {
        collect_faces[0] = true;  // -X
        collect_faces[3] = true;  // +X
    }

    /**
     * @brief Enable collection on ±Y faces
     */
    void enable_y_faces() {
        collect_faces[1] = true;  // -Y
        collect_faces[4] = true;  // +Y
    }

    /**
     * @brief Enable collection on ±Z faces (most common use case)
     */
    void enable_z_faces() {
        collect_faces[2] = true;  // -Z
        collect_faces[5] = true;  // +Z
    }

    /**
     * @brief Disable collection on all faces
     */
    void disable_all_faces() {
        for (int i = 0; i < 6; i++) {
            collect_faces[i] = false;
        }
    }

    /**
     * @brief Set collection radius for positive faces
     */
    void set_positive_radius(float radius) {
        collection_radius_positive = radius;
    }

    /**
     * @brief Set collection radius for negative faces
     */
    void set_negative_radius(float radius) {
        collection_radius_negative = radius;
    }

    /**
     * @brief Set collection center point
     */
    void set_center(float x, float y, float z) {
        collection_center = make_float3(x, y, z);
        use_grid_center = false;
    }

    /**
     * @brief Check if a face collects photons
     */
    bool is_collecting(BoundaryFace face) const {
        return collect_faces[static_cast<int>(face)];
    }

    /**
     * @brief Check if any positive face collects photons
     */
    bool has_positive_collection() const {
        return collect_faces[3] || collect_faces[4] || collect_faces[5];
    }

    /**
     * @brief Check if any negative face collects photons
     */
    bool has_negative_collection() const {
        return collect_faces[0] || collect_faces[1] || collect_faces[2];
    }

    /**
     * @brief Get human-readable description
     */
    std::string to_string() const {
        std::string result = "BoundaryCollection: ";

        if (collect_faces[0]) result += "-X ";
        if (collect_faces[1]) result += "-Y ";
        if (collect_faces[2]) result += "-Z ";
        if (collect_faces[3]) result += "+X ";
        if (collect_faces[4]) result += "+Y ";
        if (collect_faces[5]) result += "+Z ";

        if (result == "BoundaryCollection: ") {
            result += "(none)";
        }

        return result;
    }
};

} // namespace phonder::voxel
