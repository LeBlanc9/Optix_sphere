#pragma once
#include "utils/vector_types.h"
#include "photon/photon_batch.h"

namespace phonder {

// Abstract base class for all photon sources
class PhotonSource {
public:
    virtual ~PhotonSource() = default;

    // Generate photons directly into a PhotonBatch on GPU
    virtual void generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const = 0;

public:
    double weight = 1.0;
};

// Isotropic point source
class IsotropicPointSource : public PhotonSource {
public:
    float3 position = {0, 0, 0};

    void generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const override;
};

// Collimated beam source
class CollimatedBeamSource : public PhotonSource {
public:
    float3 position = {0, 0, 0};
    float3 direction = {0, 0, 1};

    void generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const override;
};

// Spot source (uniform disk)
class SpotSource : public PhotonSource {
public:
    float3 center_position = {0, 0, 0};
    float3 disk_normal = {0, 0, 1};
    float3 direction = {0, 0, 1};
    float radius = 1.0f;

    void generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const override;
};

// Gaussian beam source
class GaussianBeamSource : public PhotonSource {
public:
    float3 center_position = {0, 0, 0};
    float3 direction = {0, 0, 1};
    float beam_waist = 1.0f;

    void generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const override;
};

// Focused spot source
class FocusedSpotSource : public PhotonSource {
public:
    float3 spot_center = {0, 0, 0};
    float spot_radius = 1.0f;
    float convergence_half_angle_rad = 0.1f;
    float3 main_axis = {0, 0, 1};
    float source_distance = 10.0f;

    void generate(PhotonBatch& batch, int num_photons, unsigned long long seed) const override;
};

} // namespace phonder
