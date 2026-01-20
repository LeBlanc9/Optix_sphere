#pragma once

#include <vector>
#include <memory>
#include <numeric>

struct float3;


namespace phonder {

struct DeviceData;

struct HostPhotonBatch {
    std::vector<float3> positions;
    std::vector<float3> directions;
    std::vector<double> weights;

    size_t size() const { return positions.size(); }
    bool empty() const { return positions.empty(); }
    double total_weight() const {
        return std::accumulate(weights.begin(), weights.end(), 0.0);
    }
};


/**
 * @class PhotonBatch
 * @brief A class managing a photon batch on the GPU, with a clean, type-safe API.
 */
class PhotonBatch {
public:
    PhotonBatch(size_t initial_size = 0);
    ~PhotonBatch();

    PhotonBatch(const PhotonBatch& other);
    PhotonBatch& operator=(const PhotonBatch& other);
    PhotonBatch(PhotonBatch&& other) noexcept;
    PhotonBatch& operator=(PhotonBatch&& other) noexcept;

    // --- Device Data Access ---
    // Non-const overloads (returns writable pointers)
    float3* positions();
    float3* directions();
    double* weights();

    // Const overloads (returns read-only pointers)
    const float3* positions() const;
    const float3* directions() const;
    const double* weights() const;

    // --- Management ---
    void resize(size_t new_size);
    size_t size() const;
    bool empty() const;
    void clear();
    double total_weight() const;

    // --- Batch Operations ---
    void append(const PhotonBatch& other);
    void swap(PhotonBatch& other) noexcept;
    static PhotonBatch merge(const std::vector<const PhotonBatch*>& batches);

    // --- Explicit Data Movement ---
    HostPhotonBatch to_host() const;

private:
    std::unique_ptr<DeviceData> device_data_;
};

} // namespace phonder

