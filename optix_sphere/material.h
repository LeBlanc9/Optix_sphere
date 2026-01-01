#pragma once

#include <string>
#include <memory>
#include <vector_types.h>
#include <map>
#include <functional>

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

// Type alias for material factory function
// Takes sphere center (for future compatibility) and returns a shared_ptr to Material
// Note: center parameter is currently unused but kept for API compatibility
using MaterialFactory = std::function<std::shared_ptr<Material>(float3)>;

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
     */
    LambertianMaterial(float reflectance)
        : reflectance_(reflectance) {}

    std::string get_kernel_name() const override { return "__closesthit__lambertian"; }
    std::string get_shadow_kernel_name() const override { return "__anyhit__lambertian_shadow"; }
    size_t get_sbt_data_size() const override;
    void write_sbt_data(void* dest) const override;

    // Accessor for reflectance
    float get_reflectance() const { return reflectance_; }

private:
    float reflectance_;
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
     */
    AbsorberMaterial() = default;

    std::string get_kernel_name() const override { return "__closesthit__absorber"; }
    std::string get_shadow_kernel_name() const override { return "__anyhit__absorber_shadow"; }
    size_t get_sbt_data_size() const override;
    void write_sbt_data(void* dest) const override;
};

/**
 * MixedMaterial - Mixed diffuse/specular reflector
 *
 * Represents a surface with both diffuse (Lambertian) and specular components.
 * At each bounce, randomly selects between diffuse and specular reflection
 * based on the configured ratios.
 *
 * Physical properties:
 * - Diffuse ratio: Fraction of light that scatters diffusely (0-1)
 * - Specular ratio: Fraction of light that reflects specularly (0-1)
 * - Note: diffuse_ratio + specular_ratio should equal 1.0
 * - Total reflectance: Overall fraction of light reflected (0-1)
 *
 * Usage examples:
 * - Realistic integrating spheres (0.7 diffuse + 0.3 specular)
 * - Semi-glossy surfaces
 */
class MixedMaterial : public Material {
public:
    /**
     * Constructor for mixed material
     * @param diffuse_ratio Fraction using Lambertian scattering (0-1)
     * @param specular_ratio Fraction using specular reflection (0-1)
     * @param reflectance Total reflectance (0-1)
     */
    MixedMaterial(float diffuse_ratio, float specular_ratio, float reflectance)
        : diffuse_ratio_(diffuse_ratio),
          specular_ratio_(specular_ratio),
          reflectance_(reflectance) {}

    std::string get_kernel_name() const override { return "__closesthit__mixed"; }
    std::string get_shadow_kernel_name() const override { return "__anyhit__mixed_shadow"; }
    size_t get_sbt_data_size() const override;
    void write_sbt_data(void* dest) const override;

    float get_diffuse_ratio() const { return diffuse_ratio_; }
    float get_specular_ratio() const { return specular_ratio_; }
    float get_reflectance() const { return reflectance_; }

private:
    float diffuse_ratio_;
    float specular_ratio_;
    float reflectance_;
};

// ============================================
// Material Factory Helper Functions
// ============================================

/**
 * Convenient factory functions for creating materials
 *
 * Usage in C++:
 *   using namespace material;
 *   std::map<std::string, MaterialFactory> materials;
 *   materials["wall"] = mixed(0.7, 0.3, 0.98);
 *   materials["detector"] = detector();
 *
 * Usage in Python:
 *   from optix_sphere import material
 *   materials = {}
 *   materials["wall"] = material.mixed(0.7, 0.3, 0.98)
 *   materials["detector"] = material.detector()
 */
namespace material {

/**
 * Create a Lambertian (purely diffuse) material factory
 * @param reflectance Surface reflectance (0-1)
 */
inline MaterialFactory lambertian(float reflectance) {
    return [reflectance](float3 center) {
        (void)center;  // Unused, kept for API compatibility
        return std::make_shared<LambertianMaterial>(reflectance);
    };
}

/**
 * Create a mixed (diffuse + specular) material factory
 * @param diffuse_ratio Fraction using Lambertian scattering (0-1)
 * @param specular_ratio Fraction using specular reflection (0-1)
 * @param reflectance Total reflectance (0-1)
 */
inline MaterialFactory mixed(float diffuse_ratio, float specular_ratio, float reflectance) {
    return [diffuse_ratio, specular_ratio, reflectance](float3 center) {
        (void)center;  // Unused, kept for API compatibility
        return std::make_shared<MixedMaterial>(diffuse_ratio, specular_ratio, reflectance);
    };
}

/**
 * Create a detector material factory
 */
inline MaterialFactory detector() {
    return [](float3 center) {
        (void)center;  // Unused
        return std::make_shared<DetectorMaterial>();
    };
}

/**
 * Create an absorber (perfect black body) material factory
 */
inline MaterialFactory absorber() {
    return [](float3 center) {
        (void)center;  // Unused, kept for API compatibility
        return std::make_shared<AbsorberMaterial>();
    };
}

/**
 * Get default material factory mapping
 * Maps common OBJ material names to default material factories
 */
inline std::map<std::string, MaterialFactory> get_default_materials() {
    std::map<std::string, MaterialFactory> materials;

    // Sphere wall - mixed material (70% diffuse + 30% specular)
    materials["Sphere_Wall"] = mixed(0.7f, 0.3f, 0.98f);
    materials["SphereWall"] = mixed(0.7f, 0.3f, 0.98f);
    materials["wall_material"] = mixed(0.7f, 0.3f, 0.98f);

    // Detector
    materials["Detector"] = detector();
    materials["detector_material"] = detector();

    // Baffle - low reflectance Lambertian
    materials["Baffle"] = lambertian(0.05f);
    materials["baffle_material"] = lambertian(0.05f);

    // Port hole - perfect absorber
    materials["Port_Hole"] = absorber();
    materials["PortHole"] = absorber();
    materials["porthole_material"] = absorber();

    return materials;
}

} // namespace material
