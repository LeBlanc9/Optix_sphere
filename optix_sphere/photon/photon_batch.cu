#include "photon_batch.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include "utils/vector_types.h"

namespace phonder {

// The PIMPL struct now only holds device-side data and a size counter.
struct DeviceData {
    thrust::device_vector<float3> d_positions;
    thrust::device_vector<float3> d_directions;
    thrust::device_vector<double> d_weights;
    size_t count = 0;
};

// --- Constructor, Destructor, and Semantics ---
PhotonBatch::PhotonBatch(size_t initial_size) : device_data_(std::make_unique<DeviceData>()) {
    if (initial_size > 0) resize(initial_size);
}
PhotonBatch::~PhotonBatch() = default;
PhotonBatch::PhotonBatch(PhotonBatch&& other) noexcept = default;
PhotonBatch& PhotonBatch::operator=(PhotonBatch&& other) noexcept = default;

PhotonBatch::PhotonBatch(const PhotonBatch& other) : device_data_(std::make_unique<DeviceData>()) {
    *device_data_ = *other.device_data_;
}
PhotonBatch& PhotonBatch::operator=(const PhotonBatch& other) {
    if (this != &other) *device_data_ = *other.device_data_;
    return *this;
}

// --- Explicit Data Movement ---
HostPhotonBatch PhotonBatch::to_host() const {
    HostPhotonBatch h_batch;
    size_t current_size = device_data_->count;
    if (current_size > 0) {
        h_batch.positions.resize(current_size);
        h_batch.directions.resize(current_size);
        h_batch.weights.resize(current_size);
        thrust::copy(device_data_->d_positions.begin(), device_data_->d_positions.end(), h_batch.positions.begin());
        thrust::copy(device_data_->d_directions.begin(), device_data_->d_directions.end(), h_batch.directions.begin());
        thrust::copy(device_data_->d_weights.begin(), device_data_->d_weights.end(), h_batch.weights.begin());
    }
    return h_batch;
}

// --- Device Pointer Accessors ---
float3* PhotonBatch::positions() { return thrust::raw_pointer_cast(device_data_->d_positions.data()); }
float3* PhotonBatch::directions() { return thrust::raw_pointer_cast(device_data_->d_directions.data()); }
double* PhotonBatch::weights() { return thrust::raw_pointer_cast(device_data_->d_weights.data()); }

const float3* PhotonBatch::positions() const { return thrust::raw_pointer_cast(device_data_->d_positions.data()); }
const float3* PhotonBatch::directions() const { return thrust::raw_pointer_cast(device_data_->d_directions.data()); }
const double* PhotonBatch::weights() const { return thrust::raw_pointer_cast(device_data_->d_weights.data()); }

// --- Management ---
void PhotonBatch::resize(size_t new_size) {
    device_data_->d_positions.resize(new_size);
    device_data_->d_directions.resize(new_size);
    device_data_->d_weights.resize(new_size);
    device_data_->count = new_size;
}

size_t PhotonBatch::size() const { return device_data_->count; }
bool PhotonBatch::empty() const { return device_data_->count == 0; }
void PhotonBatch::clear() {
    device_data_->d_positions.clear();
    device_data_->d_directions.clear();
    device_data_->d_weights.clear();
    device_data_->count = 0;
}

double PhotonBatch::total_weight() const {
    if (device_data_->count == 0) return 0.0;
    return thrust::reduce(device_data_->d_weights.begin(), device_data_->d_weights.end(), 0.0);
}

// --- Batch Operations ---

/**
 * @brief Append another batch to this one (requires GPU memory copy)
 */
void PhotonBatch::append(const PhotonBatch& other) {
    if (other.empty()) return;

    size_t old_size = device_data_->count;
    size_t new_size = old_size + other.size();

    // Resize to accommodate both batches
    device_data_->d_positions.resize(new_size);
    device_data_->d_directions.resize(new_size);
    device_data_->d_weights.resize(new_size);

    // Copy other's data to the end
    thrust::copy(
        other.device_data_->d_positions.begin(),
        other.device_data_->d_positions.begin() + other.size(),
        device_data_->d_positions.begin() + old_size
    );
    thrust::copy(
        other.device_data_->d_directions.begin(),
        other.device_data_->d_directions.begin() + other.size(),
        device_data_->d_directions.begin() + old_size
    );
    thrust::copy(
        other.device_data_->d_weights.begin(),
        other.device_data_->d_weights.begin() + other.size(),
        device_data_->d_weights.begin() + old_size
    );

    device_data_->count = new_size;
}

/**
 * @brief Swap data with another batch (zero-copy operation)
 */
void PhotonBatch::swap(PhotonBatch& other) noexcept {
    device_data_->d_positions.swap(other.device_data_->d_positions);
    device_data_->d_directions.swap(other.device_data_->d_directions);
    device_data_->d_weights.swap(other.device_data_->d_weights);
    std::swap(device_data_->count, other.device_data_->count);
}

/**
 * @brief Merge multiple batches into a single new batch
 *
 * @param batches Vector of batch pointers to merge
 * @return New batch containing all photons
 */
PhotonBatch PhotonBatch::merge(const std::vector<const PhotonBatch*>& batches) {
    // Calculate total size
    size_t total_size = 0;
    for (const auto* batch : batches) {
        if (batch && !batch->empty()) {
            total_size += batch->size();
        }
    }

    // Create result batch
    PhotonBatch result(total_size);
    if (total_size == 0) {
        return result;
    }

    // Copy each batch sequentially
    size_t offset = 0;
    for (const auto* batch : batches) {
        if (!batch || batch->empty()) continue;

        size_t batch_size = batch->size();
        thrust::copy(
            batch->device_data_->d_positions.begin(),
            batch->device_data_->d_positions.begin() + batch_size,
            result.device_data_->d_positions.begin() + offset
        );
        thrust::copy(
            batch->device_data_->d_directions.begin(),
            batch->device_data_->d_directions.begin() + batch_size,
            result.device_data_->d_directions.begin() + offset
        );
        thrust::copy(
            batch->device_data_->d_weights.begin(),
            batch->device_data_->d_weights.begin() + batch_size,
            result.device_data_->d_weights.begin() + offset
        );

        offset += batch_size;
    }

    return result;
}

} // namespace phonder
