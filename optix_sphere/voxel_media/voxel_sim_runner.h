#pragma once
#include <cuda_runtime.h>
#include "voxel_grid.cuh"
#include "kernels/device_types.cuh"
#include "voxel_sim_config.h"
#include "voxel_simulator.h"
#include "photon/photon_batch.h"

namespace phonder::voxel {

/**
 * @brief Run voxel Monte Carlo simulation
 *
 * This function handles ALL device-side operations:
 * - Allocate device memory for input/output batches
 * - Copy input batch to device
 * - Prepare kernel parameters
 * - Launch kernel
 * - Copy results back to host (counters)
 * - Resize output batches based on actual counts
 *
 * @param grid_struct Device pointer to Grid structure (already on device)
 * @param input_batch Input photon batch (on device)
 * @param boundary_config Boundary collection configuration
 * @param enable_specular Enable specular reflection at entry
 * @param seed Random seed
 * @return SimulationResult with output batches on device
 */
SimulationResult run_voxel_simulation(
    const Grid* grid_struct,               // Device pointer
    const PhotonBatch& input_batch,        // Device batch
    const BoundaryCollectionConfig& boundary_config,
    bool enable_specular,
    bool merge_specular,
    unsigned long long seed
);

} // namespace phonder::voxel
