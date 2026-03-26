#pragma once
#include <memory>
#include "voxel_grid.cuh"
#include "voxel_sim_config.h"
#include "photon/sources.h"
#include "photon/photon_batch.h"
#include "layered_media/media_simulator.cuh"  // For HostMediaSimulationResult

namespace phonder::voxel {

/**
 * @brief Results of voxel media simulation
 */
struct SimulationResult {
    PhotonBatch specular_batch;         // Specular reflection at entry
    PhotonBatch negative_boundary_batch; // Photons exiting from -X/-Y/-Z faces
    PhotonBatch positive_boundary_batch; // Photons exiting from +X/+Y/+Z faces

    /**
     * @brief Copy results to host (backward compatibility)
     *
     * For backward compatibility with layered media:
     * - reflected_batch = negative_boundary_batch (typically -Z face)
     * - transmitted_batch = positive_boundary_batch (typically +Z face)
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
            negative_boundary_batch.to_host(),  // reflected
            positive_boundary_batch.to_host(),  // transmitted
            specular_weight
        };
    }
};

/**
 * @brief Voxel media simulator
 *
 * Always initialized from SimConfig.
 * SimConfig owns all data (std::vector), so no dangling pointer issues.
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
     * @brief Run simulation with an explicit source and photon count
     * @param source Photon source used to generate input photons for this run
     * @param num_photons Number of photons to simulate
     * @return Simulation results on device
     */
    SimulationResult run(const std::shared_ptr<PhotonSource>& source, int num_photons);

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
    void update_source(std::shared_ptr<PhotonSource> source);

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

    // Host configuration (owns data via std::vector)
    SimConfig config_;
};

} // namespace phonder::voxel
