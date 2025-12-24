#pragma once
#include "layered_medium.cuh"
#include "photon/sources.h"
#include "photon/batch.h"
#include "photon/batch.cuh"
#include <memory>
#include <numeric>

namespace phonder {

struct MediaSimConfig {
    LayeredMedium medium;
    PhotonSource source; // Holds a variant of source parameters
    int gpu_id = 0;
    // --- Filter parameters ---
    float reflected_radius = -1.0f;    // Maximum radius for reflected photons (-1 for no filter)
    float transmitted_radius = -1.0f;  // Maximum radius for transmitted photons (-1 for no filter)
};

/**
 * @brief CPU-based result structure for media simulation (for debugging)
 * Contains batches of reflected and transmitted photons on the host.
 */
struct MediaSimulationResult {
    HostPhotonBatch reflected_batch;
    HostPhotonBatch transmitted_batch;
    double specular_reflection_weight = 0.0;

    double reflected_weight() const {
        return reflected_batch.size() > 0 ? 
            std::accumulate(reflected_batch.weights.begin(), reflected_batch.weights.end(), 0.0) : 0.0;
    }
    double transmitted_weight() const {
        return transmitted_batch.size() > 0 ?
            std::accumulate(transmitted_batch.weights.begin(), transmitted_batch.weights.end(), 0.0) : 0.0;
    }
};

/**
 * @brief GPU-based result structure for media simulation (for performance)
 * Contains batches of reflected and transmitted photons on the GPU.
 */
struct MediaSimulationDeviceResult {
    DevicePhotonBatch reflected_batch;
    DevicePhotonBatch transmitted_batch;
    double specular_reflection_weight = 0.0;
    
    MediaSimulationResult to_host() const;
};

/**
 * @brief Simulates photon transport through a layered medium.
 * 
 * This class takes a light source and a layered medium definition. Its primary
 * function is to run a Monte Carlo simulation and return the photons that exit
 * from the top and bottom surfaces of the medium.
 */
class MediaSimulator {
public:
    __host__ MediaSimulator(const MediaSimConfig& config) : config_(config) {}

    __host__ MediaSimulationDeviceResult run(int num_photons);
    
    __host__ MediaSimulationDeviceResult run(const DevicePhotonBatch& input_batch);
    
    __host__ MediaSimulationResult run_and_copy_to_cpu(int num_photons);

    __host__ const LayeredMedium& get_medium() const { return config_.medium; }

    /**
     * @brief Updates the layered medium without recreating the simulator.
     * This allows efficient parameter updates during optimization loops.
     * @param new_medium The new medium configuration
     */
    __host__ void update_medium(const LayeredMedium& new_medium) {
        config_.medium = new_medium;
    }

private:
    MediaSimConfig config_;
};

} // namespace phonder