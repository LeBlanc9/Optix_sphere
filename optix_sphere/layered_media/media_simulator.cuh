#pragma once
#include "layered_medium.cuh"
#include "photon/sources.h"
#include "photon/photon_batch.h"
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
 * @brief Represents the host-side copy of the simulation results.
 */
struct HostMediaSimulationResult {
    HostPhotonBatch reflected_batch;
    HostPhotonBatch transmitted_batch;
    double specular_reflection_weight = 0.0;
};


/**
 * @brief Holds the results of a media simulation, primarily on the device.
 * Data can be explicitly copied to the host when needed for analysis.
 */
struct MediaSimulationResult {
    PhotonBatch reflected_batch;
    PhotonBatch transmitted_batch;
    double specular_reflection_weight = 0.0;

    /**
     * @brief Copies the simulation results from the device to the host for analysis.
     * @return A new struct containing the results in host-side std::vectors.
     */
    HostMediaSimulationResult to_host() const {
        return {
            reflected_batch.to_host(),
            transmitted_batch.to_host(),
            specular_reflection_weight
        };
    }
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

    __host__ MediaSimulationResult run(int num_photons);
    
    __host__ MediaSimulationResult run(const PhotonBatch& input_batch);
    
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