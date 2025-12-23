#pragma once

#include <string>
#include <memory>
#include <vector_types.h>

/**
 * Abstract base class for materials based on physical behavior
 *
 * This system uses physics-based material definitions.
 * Materials do not know their "role" in the scene (wall, baffle, etc.).
 * They only know their physical light interaction behavior.
 */
class Material {
public:
    virtual ~Material() = default;

    // Get closest-hit program name for this material
    virtual std::string get_kernel_name() const = 0;

    // Get shadow any-hit program name for this material
    virtual std::string get_shadow_kernel_name() const = 0;

    // Get size of SBT data structure for this material
    virtual size_t get_sbt_data_size() const = 0;

    // Write SBT data to the provided buffer
    virtual void write_sbt_data(void* dest) const = 0;
};

/**
 * LambertianMaterial - Ideal diffuse (Lambertian) reflector
 *
 * Represents a perfectly diffuse surface that scatters light uniformly
 * in all directions in the hemisphere above the surface.
 *
 * Physical properties:
 * - BRDF: ρ/π (constant in all directions)
 * - Reflectance: 0-1 (fraction of light reflected)
 *
 * Usage examples:
 * - High reflectance (0.98): Integrating sphere wall
 * - Low reflectance (0.05): Light-absorbing baffle
 * - Medium reflectance (0.5): Matte surface
 */
class LambertianMaterial : public Material {
public:
    /**
     * Constructor for Lambertian material
     * @param reflectance Fraction of light reflected (0-1)
     * @param center Sphere center (for normal calculation in spherical geometry)
     */
    LambertianMaterial(float reflectance, float3 center)
        : reflectance_(reflectance), center_(center) {}

    std::string get_kernel_name() const override { return "__closesthit__lambertian"; }
    std::string get_shadow_kernel_name() const override { return "__anyhit__lambertian_shadow"; }
    size_t get_sbt_data_size() const override;
    void write_sbt_data(void* dest) const override;

    // Accessor for reflectance
    float get_reflectance() const { return reflectance_; }

private:
    float reflectance_;
    float3 center_;
};

/**
 * DetectorMaterial - Energy recording sensor surface
 *
 * Special functional material that absorbs photons and records their energy.
 * This is not a purely physical material, but rather a measurement device.
 *
 * Behavior:
 * - Absorbs all incident light (no reflection)
 * - Records weighted flux for irradiance calculation
 * - Terminates ray paths
 *
 * Used for: Photodetectors, radiometers, flux measurement surfaces
 */
class DetectorMaterial : public Material {
public:
    DetectorMaterial() = default;

    std::string get_kernel_name() const override { return "__closesthit__detector"; }
    std::string get_shadow_kernel_name() const override { return "__anyhit__detector_shadow"; }
    size_t get_sbt_data_size() const override;
    void write_sbt_data(void* dest) const override;
};

/**
 * AbsorberMaterial - Perfect light absorber (black body)
 *
 * Represents a surface that completely absorbs all incident light.
 * No reflection, no transmission - photons simply terminate.
 *
 * Physical properties:
 * - Reflectance: 0 (complete absorption)
 * - Emittance: 0 (we don't model thermal emission here)
 *
 * Usage examples:
 * - Port holes (light escapes the system)
 * - Black surfaces
 * - Light traps
 */
class AbsorberMaterial : public Material {
public:
    /**
     * Constructor for absorber material
     * @param center Sphere center (for geometric consistency)
     */
    AbsorberMaterial(float3 center)
        : center_(center) {}

    std::string get_kernel_name() const override { return "__closesthit__absorber"; }
    std::string get_shadow_kernel_name() const override { return "__anyhit__absorber_shadow"; }
    size_t get_sbt_data_size() const override;
    void write_sbt_data(void* dest) const override;

private:
    float3 center_;
};
