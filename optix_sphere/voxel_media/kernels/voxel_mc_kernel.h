#pragma once
#include <cuda_runtime.h>
#include "voxel_media/voxel_grid.cuh"
#include "device_types.cuh"

namespace phonder::voxel {

/**
 * @brief Main Monte Carlo simulation kernel for voxel media
 *
 * This kernel simulates photon transport through a 3D voxel grid using
 * Monte Carlo ray tracing with MCX-style physics.
 *
 * All parameters are packed into MCKernelParams structure.
 */
__global__ void voxel_kernel(const MCKernelParams params);

} // namespace phonder::voxel
