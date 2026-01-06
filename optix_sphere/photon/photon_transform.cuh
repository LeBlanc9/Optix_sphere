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

} // namespace phonder
