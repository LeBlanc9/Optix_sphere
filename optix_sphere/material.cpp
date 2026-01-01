#include "material.h"
#include "simulation/device_params.h"
#include <cstring>

// ============================================
// LambertianMaterial Implementation
// ============================================

size_t LambertianMaterial::get_sbt_data_size() const {
    return sizeof(SphereWallSbtData);
}

void LambertianMaterial::write_sbt_data(void* dest) const {
    SphereWallSbtData* data = static_cast<SphereWallSbtData*>(dest);
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
    return sizeof(PortHoleSbtData);
}

void AbsorberMaterial::write_sbt_data(void* dest) const {
    PortHoleSbtData* data = static_cast<PortHoleSbtData*>(dest);
    data->dummy = 0;  // Empty struct
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
