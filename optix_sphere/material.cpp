#include "material.h"
#include "simulation/device_params.h"
#include <cstring>

// ============================================
// LambertianMaterial Implementation
// ============================================

size_t LambertianMaterial::get_sbt_data_size() const {
    return sizeof(LambertianSbtData);
}

void LambertianMaterial::write_sbt_data(void* dest) const {
    LambertianSbtData* data = static_cast<LambertianSbtData*>(dest);
    data->reflectance = reflectance_;
}

// ============================================
// DetectorMaterial Implementation
// ============================================

size_t DetectorMaterial::get_sbt_data_size() const {
    return sizeof(DetectorSbtData);
}

void DetectorMaterial::write_sbt_data(void* dest) const {
    // DetectorSbtData fields are set elsewhere (in DeviceParams)
    // For now, just zero-initialize
    std::memset(dest, 0, sizeof(DetectorSbtData));
}

// ============================================
// AbsorberMaterial Implementation
// ============================================

size_t AbsorberMaterial::get_sbt_data_size() const {
    return sizeof(AbsorberSbtData);
}

void AbsorberMaterial::write_sbt_data(void* dest) const {
    AbsorberSbtData* data = static_cast<AbsorberSbtData*>(dest);
    data->dummy = 0;
}

// ============================================
// MixedMaterial Implementation
// ============================================

size_t MixedMaterial::get_sbt_data_size() const {
    return sizeof(MixedMaterialSbtData);
}

void MixedMaterial::write_sbt_data(void* dest) const {
    MixedMaterialSbtData* data = static_cast<MixedMaterialSbtData*>(dest);
    data->diffuse_ratio = diffuse_ratio_;
    data->specular_ratio = specular_ratio_;
    data->reflectance = reflectance_;
}

// ============================================
// SphericalLambertianMaterial Implementation
// ============================================

size_t SphericalLambertianMaterial::get_sbt_data_size() const {
    return sizeof(SphericalLambertianSbtData);
}

void SphericalLambertianMaterial::write_sbt_data(void* dest) const {
    SphericalLambertianSbtData* data = static_cast<SphericalLambertianSbtData*>(dest);
    data->reflectance = reflectance_;
    data->center = center_;
}

// ============================================
// SphericalMixedMaterial Implementation
// ============================================

size_t SphericalMixedMaterial::get_sbt_data_size() const {
    return sizeof(SphericalMixedSbtData);
}

void SphericalMixedMaterial::write_sbt_data(void* dest) const {
    SphericalMixedSbtData* data = static_cast<SphericalMixedSbtData*>(dest);
    data->diffuse_ratio = diffuse_ratio_;
    data->specular_ratio = specular_ratio_;
    data->reflectance = reflectance_;
    data->center = center_;
}
