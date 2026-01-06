#pragma once
#include "sources.h"
#include "photon_batch.h"

// This header is the public API for the CUDA photon generation launchers.
// It acts as the bridge between the C++ application logic and the CUDA implementation.

namespace phonder {

/**
 * @brief Generates photons on the device based on the given source parameters.
 * 
 * This is a C-style bridge function. Its implementation is in a .cu file.
 * The C++ compiler only sees this declaration, while the CUDA toolchain
 * provides the implementation at link time.
 * 
 * @param source A variant holding the parameters of the source to use.
 * @param batch_out The batch to be filled with generated photons. The data will be generated on the device.
 * @param num_photons The number of photons to generate.
 * @param seed An optional random seed.
 */
void generate_photons_on_device(
    const PhotonSource& source,
    PhotonBatch& batch_out,
    int num_photons,
    unsigned long long seed
);

} // namespace phonder
