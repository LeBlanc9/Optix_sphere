#include "batch.cuh"
#include "batch.h" // For HostPhotonBatch definition
#include <thrust/copy.h>

// This file implements the methods for the CUDA-side device batch.

namespace phonder {

void DevicePhotonBatch::resize(size_t new_size) {
    positions.resize(new_size);
    directions.resize(new_size);
    weights.resize(new_size);
}

void DevicePhotonBatch::clear() {
    positions.clear();
    directions.clear();
    weights.clear();
}

size_t DevicePhotonBatch::size() const {
    return positions.size();
}

bool DevicePhotonBatch::empty() const {
    return positions.empty();
}

PhotonBatchView DevicePhotonBatch::get_view() {
    return {
        thrust::raw_pointer_cast(positions.data()),
        thrust::raw_pointer_cast(directions.data()),
        thrust::raw_pointer_cast(weights.data())
    };
}

PhotonBatchView DevicePhotonBatch::get_view() const {
    // const版本：返回const数据的指针（但PhotonBatchView本身不是const指针类型）
    // 这是安全的，因为调用者应该知道这是const batch的view
    return {
        const_cast<float3*>(thrust::raw_pointer_cast(positions.data())),
        const_cast<float3*>(thrust::raw_pointer_cast(directions.data())),
        const_cast<double*>(thrust::raw_pointer_cast(weights.data()))
    };
}

HostPhotonBatch DevicePhotonBatch::to_host() const {
    HostPhotonBatch h_batch;
    h_batch.resize(this->size());
    if (this->empty()) {
        return h_batch;
    }
    thrust::copy(this->positions.begin(), this->positions.end(), h_batch.positions.begin());
    thrust::copy(this->directions.begin(), this->directions.end(), h_batch.directions.begin());
    thrust::copy(this->weights.begin(), this->weights.end(), h_batch.weights.begin());
    return h_batch;
}

} // namespace phonder
