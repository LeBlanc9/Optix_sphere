#pragma once

#include <string>
#include <memory>
#include <vector_types.h>
#include <map>
#include <functional>
#include <cmath>

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

    // Get a string description of the material parameters
    virtual std::string description() const = 0;
};

// Type alias for material factory function
// Returns a shared_ptr to Material
// For spherical materials, the center is captured in the factory closure
using MaterialFactory = std::function<std::shared_ptr<Material>()>;

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
    std::string description() const override {
        return "Lambertian(reflectance=" + std::to_string(reflectance_) + ")";
    }

    // Accessors
    float get_reflectance() const { return reflectance_; }
    void set_reflectance(float reflectance) { reflectance_ = reflectance; }

private:
    float reflectance_;
};

/**
 * SphericalLambertianMaterial - Optimized Lambertian for spherical geometry
 *
 * Same physical behavior as LambertianMaterial, but uses spherical normal
 * calculation (from center point) instead of triangle geometric normals.
 * This is faster but ONLY accurate for perfectly spherical surfaces.
 *
 * Performance: ~3-5x faster than TriangleLambertian
 * Accuracy: Perfect for sphere, incorrect for planar/arbitrary geometry
 *
 * Use when:
 * - All surfaces are part of a perfect sphere
 * - No baffles or planar components in the scene
 */
class SphericalLambertianMaterial : public Material {
public:
    /**
     * Constructor for spherical Lambertian material
     * @param reflectance Fraction of light reflected (0-1)
     * @param center Sphere center for normal calculation
     */
    SphericalLambertianMaterial(float reflectance, float3 center)
        : reflectance_(reflectance), center_(center) {}

    std::string get_kernel_name() const override { return "__closesthit__spherical_lambertian"; }
    std::string get_shadow_kernel_name() const override { return "__anyhit__lambertian_shadow"; }
    size_t get_sbt_data_size() const override;
    void write_sbt_data(void* dest) const override;
    std::string description() const override {
        return "SphericalLambertian(reflectance=" + std::to_string(reflectance_) +
               ", center=[" + std::to_string(center_.x) + "," + std::to_string(center_.y) + "," + std::to_string(center_.z) + "])";
    }

    // Accessors
    float get_reflectance() const { return reflectance_; }
    void set_reflectance(float reflectance) { reflectance_ = reflectance; }

    float3 get_center() const { return center_; }
    void set_center(float3 center) { center_ = center; }

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
 * - Optional NA (Numerical Aperture) constraint for angle-limited detection
 *
 * Used for: Photodetectors, radiometers, flux measurement surfaces
 */
class DetectorMaterial : public Material {
public:
    /**
     * Constructor for detector material
     * @param na Numerical Aperture (0-1, default 1.0 = no angle limit)
     * @param n Refractive index of medium (default 1.0 for air)
     */
    DetectorMaterial(float na = 1.0f, float n = 1.0f)
        : na_(na), n_(n) {}

    std::string get_kernel_name() const override { return "__closesthit__detector"; }
    std::string get_shadow_kernel_name() const override { return "__anyhit__detector_shadow"; }
    size_t get_sbt_data_size() const override;
    void write_sbt_data(void* dest) const override;
    std::string description() const override {
        return "Detector(na=" + std::to_string(na_) + ", n=" + std::to_string(n_) + ")";
    }

    // Accessors
    float get_na() const { return na_; }
    void set_na(float na) { na_ = na; }

    float get_n() const { return n_; }
    void set_n(float n) { n_ = n; }

    // 计算最大接收角的余弦值: cos(θ_max) = sqrt(1 - (NA/n)^2)
    float get_cos_max_angle() const {
        float sin_max = na_ / n_;
        if (sin_max >= 1.0f) return 0.0f;  // NA >= n, 接收所有角度
        return sqrtf(1.0f - sin_max * sin_max);
    }

private:
    float na_;  // Numerical Aperture
    float n_;   // Refractive index
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
    std::string description() const override { return "Absorber()"; }
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
    std::string description() const override {
        return "Mixed(diffuse=" + std::to_string(diffuse_ratio_) +
               ", specular=" + std::to_string(specular_ratio_) +
               ", reflectance=" + std::to_string(reflectance_) + ")";
    }

    // Accessors
    float get_diffuse_ratio() const { return diffuse_ratio_; }
    void set_diffuse_ratio(float ratio) { diffuse_ratio_ = ratio; }

    float get_specular_ratio() const { return specular_ratio_; }
    void set_specular_ratio(float ratio) { specular_ratio_ = ratio; }

    float get_reflectance() const { return reflectance_; }
    void set_reflectance(float reflectance) { reflectance_ = reflectance; }

private:
    float diffuse_ratio_;
    float specular_ratio_;
    float reflectance_;
};

/**
 * SphericalMixedMaterial - Optimized mixed material for spherical geometry
 *
 * Same physical behavior as MixedMaterial, but uses spherical normal
 * calculation (from center point) instead of triangle geometric normals.
 * This is faster but ONLY accurate for perfectly spherical surfaces.
 *
 * Performance: ~3-5x faster than TriangleMixed
 * Accuracy: Perfect for sphere, incorrect for planar/arbitrary geometry
 *
 * Use when:
 * - All surfaces are part of a perfect sphere
 * - Need realistic diffuse+specular combination
 * - No baffles or planar components in the scene
 */
class SphericalMixedMaterial : public Material {
public:
    /**
     * Constructor for spherical mixed material
     * @param diffuse_ratio Fraction using Lambertian scattering (0-1)
     * @param specular_ratio Fraction using specular reflection (0-1)
     * @param reflectance Total reflectance (0-1)
     * @param center Sphere center for normal calculation
     */
    SphericalMixedMaterial(float diffuse_ratio, float specular_ratio,
                           float reflectance, float3 center)
        : diffuse_ratio_(diffuse_ratio),
          specular_ratio_(specular_ratio),
          reflectance_(reflectance),
          center_(center) {}

    std::string get_kernel_name() const override { return "__closesthit__spherical_mixed"; }
    std::string get_shadow_kernel_name() const override { return "__anyhit__mixed_shadow"; }
    size_t get_sbt_data_size() const override;
    void write_sbt_data(void* dest) const override;
    std::string description() const override {
        return "SphericalMixed(diffuse=" + std::to_string(diffuse_ratio_) +
               ", specular=" + std::to_string(specular_ratio_) +
               ", reflectance=" + std::to_string(reflectance_) +
               ", center=[" + std::to_string(center_.x) + "," + std::to_string(center_.y) + "," + std::to_string(center_.z) + "])";
    }

    // Accessors
    float get_diffuse_ratio() const { return diffuse_ratio_; }
    void set_diffuse_ratio(float ratio) { diffuse_ratio_ = ratio; }

    float get_specular_ratio() const { return specular_ratio_; }
    void set_specular_ratio(float ratio) { specular_ratio_ = ratio; }

    float get_reflectance() const { return reflectance_; }
    void set_reflectance(float reflectance) { reflectance_ = reflectance; }

    float3 get_center() const { return center_; }
    void set_center(float3 center) { center_ = center; }

private:
    float diffuse_ratio_;
    float specular_ratio_;
    float reflectance_;
    float3 center_;
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
    return [reflectance]() {
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
    return [diffuse_ratio, specular_ratio, reflectance]() {
        return std::make_shared<MixedMaterial>(diffuse_ratio, specular_ratio, reflectance);
    };
}

/**
 * Create a detector material factory
 * @param na Numerical Aperture (0-1, default 1.0 = no angle limit)
 * @param n Refractive index of medium (default 1.0 for air)
 */
inline MaterialFactory detector(float na = 1.0f, float n = 1.0f) {
    return [na, n]() {
        return std::make_shared<DetectorMaterial>(na, n);
    };
}

/**
 * Create an absorber (perfect black body) material factory
 */
inline MaterialFactory absorber() {
    return []() {
        return std::make_shared<AbsorberMaterial>();
    };
}

/**
 * Create a spherical Lambertian material factory (optimized for spheres)
 *
 * Uses spherical normal calculation - ~3-5x faster than triangle normals
 * ONLY use for perfectly spherical surfaces without baffles
 *
 * @param reflectance Surface reflectance (0-1)
 * @param center Sphere center for normal calculation
 */
inline MaterialFactory spherical_lambertian(float reflectance, float3 center) {
    return [reflectance, center]() {
        return std::make_shared<SphericalLambertianMaterial>(reflectance, center);
    };
}

/**
 * Create a spherical mixed material factory (optimized for spheres)
 *
 * Uses spherical normal calculation - ~3-5x faster than triangle normals
 * ONLY use for perfectly spherical surfaces without baffles
 *
 * @param diffuse_ratio Fraction using Lambertian scattering (0-1)
 * @param specular_ratio Fraction using specular reflection (0-1)
 * @param reflectance Total reflectance (0-1)
 * @param center Sphere center for normal calculation
 */
inline MaterialFactory spherical_mixed(float diffuse_ratio, float specular_ratio, float reflectance, float3 center) {
    return [diffuse_ratio, specular_ratio, reflectance, center]() {
        return std::make_shared<SphericalMixedMaterial>(diffuse_ratio, specular_ratio, reflectance, center);
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
