#pragma once
#include "utils/vector_types.h"
#include <vector>
#include <numeric> // for std::accumulate

// This header is a PURE C++ header.
// It defines the data structure for holding a batch of photons on the host (CPU).

namespace phonder {

struct HostPhotonBatch {
    std::vector<float3> positions;
    std::vector<float3> directions;
    std::vector<double> weights;

    void resize(size_t new_size) {
        positions.resize(new_size);
        directions.resize(new_size);
        weights.resize(new_size);
    }

    void clear() {
        positions.clear();
        directions.clear();
        weights.clear();
    }

    size_t size() const {
        return positions.size();
    }

    bool empty() const {
        return positions.empty();
    }

    double total_weight() const {
        return std::accumulate(weights.begin(), weights.end(), 0.0);
    }
};

} // namespace phonder
