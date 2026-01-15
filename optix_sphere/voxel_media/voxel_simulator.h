#pragma once
#include <memory>
#include "voxel_grid.cuh"
#include "voxel_grid_builder.h"
#include "voxel_sim_config.h"
#include "photon/sources.h"
#include "photon/photon_batch.h"
#include "layered_media/media_simulator.cuh"  // For HostMediaSimulationResult

namespace phonder::voxel {

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
 * @brief Voxel media simulator
 *
 * Always initialized from SimConfig.
 * Use GridBuilder to construct grids, then pass to SimConfig.
 */
class Simulator {
public:
    /**
     * @brief Constructor from SimConfig
     * @param config Simulation configuration with grid and material data
     */
    explicit Simulator(const SimConfig& config);

    ~Simulator();

    // Disable copy, enable move
    Simulator(const Simulator&) = delete;
    Simulator& operator=(const Simulator&) = delete;
    Simulator(Simulator&&) = default;
    Simulator& operator=(Simulator&&) = default;

    /**
     * @brief Run simulation with new photons from source
     * @param num_photons Number of photons to simulate
     * @return Simulation results on device
     */
    SimulationResult run(int num_photons);

    /**
     * @brief Run simulation with input photon batch
     * @param input_batch Initial photon batch
     * @return Simulation results on device
     */
    SimulationResult run(const PhotonBatch& input_batch);

    /**
     * @brief Update material properties without recreating grid
     * @param materials New material table, shape: (num_materials, 4)
     * @param num_materials Number of materials
     */
    void update_materials(const float* materials, int num_materials);

    /**
     * @brief Update light source
     * @param source New source configuration
     */
    void update_source(const PhotonSource& source);

private:
    void initialize_from_config(const SimConfig& config);
    void upload_grid_to_device();
    void free_device_memory();

    // Device memory
    struct DeviceData {
        unsigned char* grid = nullptr;           // Material IDs
        Grid* grid_struct = nullptr;             // Grid structure
        // Note: materials stored in constant memory (c_materials)
    } device_data_;

    // Host configuration
    SimConfig config_;

    // Host-side copies (owned by Simulator when using SimConfig constructor)
    std::vector<unsigned char> owned_grid_;
    std::vector<float> owned_materials_;

    bool owns_data_ = false;  // Whether this Simulator owns the grid data
};

} // namespace phonder::voxel
