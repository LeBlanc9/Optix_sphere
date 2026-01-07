#pragma once

#include "photon_batch.h"

namespace phonder {

/**
 * @brief Translate photon positions by a fixed offset
 *
 * This function shifts all photon positions in a batch by a constant 3D offset.
 * Useful for moving photons from MediaSimulator output to integrating sphere ports.
 *
 * @param input_batch The input photon batch
 * @param offset The 3D translation vector (mm)
 * @return A new photon batch with translated positions (directions unchanged)
 *
 * @example
 * ```cpp
 * Move reflected photons to sphere port at x=25mm
 * auto aligned_batch = translate_photons(
 *     media_result.reflected_batch,
 *     make_float3(25.0f, 0.0f, 0.0f)
 * );
 * ```
 */
PhotonBatch translate_photons(
    const PhotonBatch& input_batch,
    float3 offset
);

/**
 * @brief Translate photon positions in-place by a fixed offset
 *
 * This function modifies the input batch directly, avoiding memory allocation.
 * More efficient than translate_photons() for large batches.
 *
 * @param batch The photon batch to modify (will be changed)
 * @param offset The 3D translation vector (mm)
 *
 * @example
 * ```cpp
 * // In-place modification (no copy)
 * translate_photons_inplace(batch, make_float3(0.0f, 0.0f, -12.7f));
 * ```
 */
void translate_photons_inplace(
    PhotonBatch& batch,
    float3 offset
);

} // namespace phonder
