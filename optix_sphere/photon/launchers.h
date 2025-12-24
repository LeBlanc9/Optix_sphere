#pragma once
#include "sources.h"

// This header is the PURE C++ public API for the CUDA photon generation launchers.
// It acts as the bridge between the C++ application logic and the CUDA implementation.

namespace phonder {

// Forward-declare the CUDA-side device batch structure.
// The implementation details of this struct are hidden from the C++ side.
struct DevicePhotonBatch;

/**
 * @brief Generates photons on the device based on the given source parameters.
 * 
 * This is a C-style bridge function. Its implementation is in a .cu file.
 * The C++ compiler only sees this declaration, while the CUDA toolchain
 * provides the implementation at link time.
 * 
 * @param source A variant holding the parameters of the source to use.
 * @param batch_out The device batch to be filled with generated photons.
 * @param num_photons The number of photons to generate.
 * @param seed An optional random seed.
 */
void generate_photons_on_device(
    const PhotonSource& source,
    DevicePhotonBatch& batch_out,
    int num_photons,
    unsigned long long seed
);

} // namespace phonder
