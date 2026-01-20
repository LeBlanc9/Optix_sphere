#include "photon/sources.h"
#include "utils/device/math.cuh"
#include <curand_kernel.h>
#include "photon/kernels.cuh"

namespace phonder {

// ============================================
// CUDA Kernels
// ============================================

void IsotropicPointSource::generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const {
    batch.resize(num_photons);
    const int block_size = 256;
    const int grid_size = (num_photons + block_size - 1) / block_size;
    generate_isotropic_point_kernel<<<grid_size, block_size>>>(
        batch.positions(),
        batch.directions(),
        batch.weights(),
        num_photons, position, weight, seed
    );
    cudaDeviceSynchronize();
}

void CollimatedBeamSource::generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const {
    batch.resize(num_photons);
    const int block_size = 256;
    const int grid_size = (num_photons + block_size - 1) / block_size;
    generate_collimated_beam_kernel<<<grid_size, block_size>>>(
        batch.positions(),
        batch.directions(),
        batch.weights(),
        num_photons, position, direction, weight
    );
    cudaDeviceSynchronize();
}

void SpotSource::generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const {
    batch.resize(num_photons);
    const int block_size = 256;
    const int grid_size = (num_photons + block_size - 1) / block_size;
    generate_spot_source_kernel<<<grid_size, block_size>>>(
        batch.positions(),
        batch.directions(),
        batch.weights(),
        num_photons, center_position, disk_normal, direction, radius, weight, seed
    );
    cudaDeviceSynchronize();
}

void GaussianBeamSource::generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const {
    batch.resize(num_photons);
    const int block_size = 256;
    const int grid_size = (num_photons + block_size - 1) / block_size;
    generate_gaussian_source_kernel<<<grid_size, block_size>>>(
        batch.positions(),
        batch.directions(),
        batch.weights(),
        num_photons, center_position, direction, beam_waist, weight, seed
    );
    cudaDeviceSynchronize();
}

void FocusedSpotSource::generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const {
    batch.resize(num_photons);
    const int block_size = 256;
    const int grid_size = (num_photons + block_size - 1) / block_size;
    generate_focused_spot_source_kernel<<<grid_size, block_size>>>(
        batch.positions(),
        batch.directions(),
        batch.weights(),
        num_photons, spot_center, spot_radius, convergence_half_angle_rad,
        main_axis, source_distance, weight, seed
    );
    cudaDeviceSynchronize();
}

} // namespace phonder
