#include "photon_batch.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>

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

// --- Type-Safe Device Pointer Accessors ---
float3* PhotonBatch::positions_ptr() { return thrust::raw_pointer_cast(device_data_->d_positions.data()); }
float3* PhotonBatch::directions_ptr() { return thrust::raw_pointer_cast(device_data_->d_directions.data()); }
double* PhotonBatch::weights_ptr() { return thrust::raw_pointer_cast(device_data_->d_weights.data()); }

const float3* PhotonBatch::c_positions_ptr() const { return thrust::raw_pointer_cast(device_data_->d_positions.data()); }
const float3* PhotonBatch::c_directions_ptr() const { return thrust::raw_pointer_cast(device_data_->d_directions.data()); }
const double* PhotonBatch::c_weights_ptr() const { return thrust::raw_pointer_cast(device_data_->d_weights.data()); }

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

} // namespace phonder
