#pragma once
#include <memory>
#include "voxel_grid.cuh"
#include "voxel_grid_builder.h"
#include "photon/sources.h"
#include "photon/photon_batch.h"
#include "layered_media/media_simulator.cuh"  // For HostMediaSimulationResult
#include "kernels/voxel_mc_kernel.h"  // Kernel declaration

namespace phonder::voxel {

/**
 * @brief Configuration for voxel media simulation
 */
struct SimConfig {
    Grid* device_grid = nullptr;  // Device pointer to voxel grid
    PhotonSource source;                // Light source configuration
    int gpu_id = 0;

    // Exit detection
    float exit_z_min = -1.0f;  // Z coordinate for top exit (reflection)
    float exit_z_max = -1.0f;  // Z coordinate for bottom exit (transmission)
};

/**
 * @brief Results of voxel media simulation
 */
struct SimulationResult {
    PhotonBatch specular_batch;      // Specular reflection at entry
    PhotonBatch reflected_batch;     // Diffuse reflection from -Z face
    PhotonBatch transmitted_batch;   // Transmission from +Z face

    /**
     * @brief Copy results to host
     */
    HostMediaSimulationResult to_host() const {
        // Calculate total specular reflection weight
        double specular_weight = 0.0;
        if (specular_batch.size() > 0) {
            auto host_spec = specular_batch.to_host();
            for (const auto& w : host_spec.weights) {
                specular_weight += w;
            }
        }

        return {
            reflected_batch.to_host(),
            transmitted_batch.to_host(),
            specular_weight
        };
    }
};

/**
 * @brief Simulates photon transport through voxel media
 *
 * This class performs Monte Carlo simulation in a 3D voxel grid.
 * It supports arbitrary optical property distributions.
 */
class Simulator {
public:
    /**
     * @brief Constructor
     * @param grid_builder Reference to GridBuilder (must outlive this simulator)
     * @param source Light source configuration
     * @param gpu_id GPU device ID
     */
    __host__ Simulator(
        GridBuilder& grid_builder,
        const PhotonSource& source,
        int gpu_id = 0
    );

    /**
     * @brief Run simulation with specified number of photons
     * @param num_photons Number of photons to simulate
     * @return Simulation results on device
     */
    __host__ SimulationResult run(int num_photons);

    /**
     * @brief Run simulation with input photon batch
     * @param input_batch Initial photon batch
     * @return Simulation results on device
     */
    __host__ SimulationResult run(const PhotonBatch& input_batch);

    /**
     * @brief Get the device grid pointer
     */
    __host__ const Grid* get_device_grid() const {
        return config_.device_grid;
    }

    /**
     * @brief Update the voxel grid without recreating the simulator
     * @param grid_builder New grid builder
     */
    __host__ void update_grid(GridBuilder& grid_builder) {
        config_.device_grid = grid_builder.get_device_grid();

        // Update exit boundaries based on grid dimensions
        const auto& host_grid = grid_builder.get_host_grid();
        config_.exit_z_min = 0.0f;
        config_.exit_z_max = host_grid.nz * host_grid.dz;
    }

private:
    SimConfig config_;
};

} // namespace phonder::voxel
