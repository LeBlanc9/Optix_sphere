#pragma once
#include "utils/vector_types.h"
#include <string>
#include <variant>

// This header is a PURE C++ header.
// It defines the parameter-holding structs for different photon sources.
// These are simple data objects, containing no logic.

namespace phonder {

struct IsotropicPointSource {
    float3 position = {0, 0, 0};
    double weight = 1.0;
};

struct CollimatedBeamSource {
    float3 position = {0, 0, 0};
    float3 direction = {0, 0, 1};
    double weight = 1.0;
};

struct SpotSource {
    float3 center_position = {0, 0, 0};
    float3 direction = {0, 0, 1};
    float radius = 1.0f;
    double weight = 1.0;
};

struct GaussianBeamSource {
    float3 center_position = {0, 0, 0};
    float3 direction = {0, 0, 1};
    float beam_waist = 1.0f;
    double weight = 1.0;
};

struct FocusedSpotSource {
    float3 spot_center = {0, 0, 0};
    float spot_radius = 1.0f;
    float convergence_half_angle_rad = 0.1f;
    float3 main_axis = {0, 0, 1};
    float source_distance = 10.0f;
    double weight = 1.0;
};

// A variant to hold any type of source.
// This allows a single 'PhotonSource' object to represent any of the above.
using PhotonSource = std::variant<
    IsotropicPointSource,
    CollimatedBeamSource,
    SpotSource,
    GaussianBeamSource,
    FocusedSpotSource
>;

// Helper function to extract total power from any source
inline double get_source_power(const PhotonSource& source) {
    return std::visit([](const auto& s) { return s.weight; }, source);
}

} // namespace phonder
