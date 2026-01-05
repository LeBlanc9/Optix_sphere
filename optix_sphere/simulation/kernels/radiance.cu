#include "kernel_utils.cuh"

// ============================================
// Helper Functions
// ============================================

__device__ __forceinline__ float3 compute_triangle_normal() {
    unsigned int prim_idx = optixGetPrimitiveIndex();
    uint3 indices = params.index_buffer[prim_idx];
    float3 v0 = params.vertex_buffer[indices.x];
    float3 v1 = params.vertex_buffer[indices.y];
    float3 v2 = params.vertex_buffer[indices.z];
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    return normalize(cross(edge1, edge2));
}

// ============================================
// Detector and Miss Programs
// ============================================

extern "C" __global__ void __closesthit__detector() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    // Direct illumination: always count
    if (payload->bounce_count == 0) {
        atomicAdd(params.flux_buffer, payload->weight);
        atomicAdd(params.detected_rays_buffer, 1ull);
    }
    // Indirect illumination
    else {
        if (!params.use_nee) {
            // Non-NEE: count all implicit paths
            atomicAdd(params.flux_buffer, payload->weight);
            atomicAdd(params.detected_rays_buffer, 1ull);
        } else {
            // NEE: only count specular bounces (diffuse handled by shadow rays)
            if (payload->last_bounce_was_specular) {
                atomicAdd(params.flux_buffer, payload->weight);
                atomicAdd(params.detected_rays_buffer, 1ull);
            }
        }
    }

    payload->active = 0;
}

extern "C" __global__ void __miss__radiance() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);
    payload->active = 0;
}

// ============================================
// Triangle Mesh Materials
// ============================================

extern "C" __global__ void __closesthit__lambertian() {
    const SphereWallSbtData* material = (SphereWallSbtData*)optixGetSbtDataPointer();
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    float t_hit = optixGetRayTmax();
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = ray_orig + t_hit * ray_dir;

    float3 geometric_normal = compute_triangle_normal();
    float3 shading_normal = dot(ray_dir, geometric_normal) < 0 ? geometric_normal : -geometric_normal;

    // NEE: explicit detector sampling
    if (params.use_nee) {
        double brdf_reflectance = material->reflectance * INV_PI;
        accumulate_nee_contribution(hit_point, shading_normal, brdf_reflectance, payload->weight, 4);
    }

    // Max bounce termination
    atomicAdd(params.total_bounces_buffer, 1ull);
    payload->bounce_count++;
    if (payload->bounce_count >= params.max_bounces) {
        payload->active = 0;
        return;
    }

    // BRDF attenuation
    payload->weight *= material->reflectance;

    // Russian Roulette
    if (!apply_russian_roulette(payload)) {
        return;
    }

    // Sample next direction
    payload->origin = hit_point + shading_normal * 1e-3f;
    payload->direction = sample_lambertian(shading_normal, &payload->seed);
    payload->last_bounce_was_specular = false;
    payload->active = 1;
}

extern "C" __global__ void __closesthit__absorber() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);
    payload->active = 0;
}

extern "C" __global__ void __closesthit__mixed() {
    const MixedMaterialSbtData* material = (MixedMaterialSbtData*)optixGetSbtDataPointer();
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    float t_hit = optixGetRayTmax();
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = ray_orig + t_hit * ray_dir;

    float3 geometric_normal = compute_triangle_normal();
    float3 shading_normal = dot(ray_dir, geometric_normal) < 0 ? geometric_normal : -geometric_normal;

    // NEE: only diffuse component (specular is delta function)
    if (params.use_nee) {
        double brdf_reflectance = material->diffuse_ratio * material->reflectance * INV_PI;
        accumulate_nee_contribution(hit_point, shading_normal, brdf_reflectance, payload->weight, 4);
    }

    // Max bounce termination
    atomicAdd(params.total_bounces_buffer, 1ull);
    payload->bounce_count++;
    if (payload->bounce_count >= params.max_bounces) {
        payload->active = 0;
        return;
    }

    // BRDF attenuation
    payload->weight *= material->reflectance;

    // Russian Roulette
    if (!apply_russian_roulette(payload)) {
        return;
    }

    // Choose between diffuse and specular
    float choice = random_float(&payload->seed);
    float3 new_direction;
    bool is_specular;

    if (choice < material->diffuse_ratio) {
        new_direction = sample_lambertian(shading_normal, &payload->seed);
        is_specular = false;
    } else {
        new_direction = ray_dir - 2.0f * dot(ray_dir, shading_normal) * shading_normal;
        new_direction = normalize(new_direction);
        is_specular = true;
    }

    payload->origin = hit_point + shading_normal * 1e-3f;
    payload->direction = new_direction;
    payload->last_bounce_was_specular = is_specular;
    payload->active = 1;
}

// ============================================
// Spherical Materials (optimized for spheres)
// ============================================

extern "C" __global__ void __closesthit__spherical_lambertian() {
    const SphericalLambertianSbtData* material = (SphericalLambertianSbtData*)optixGetSbtDataPointer();
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    float t_hit = optixGetRayTmax();
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = ray_orig + t_hit * ray_dir;

    float3 geometric_normal = normalize(hit_point - material->center);
    float3 shading_normal = dot(ray_dir, geometric_normal) < 0 ? geometric_normal : -geometric_normal;

    // NEE: explicit detector sampling
    if (params.use_nee) {
        double brdf_reflectance = material->reflectance * INV_PI;
        accumulate_nee_contribution(hit_point, shading_normal, brdf_reflectance, payload->weight, 1);
    }

    // Max bounce termination
    atomicAdd(params.total_bounces_buffer, 1ull);
    payload->bounce_count++;
    if (payload->bounce_count >= params.max_bounces) {
        payload->active = 0;
        return;
    }

    // BRDF attenuation
    payload->weight *= material->reflectance;

    // Russian Roulette
    if (!apply_russian_roulette(payload)) {
        return;
    }

    // Sample next direction
    payload->origin = hit_point + shading_normal * 1e-4f;
    payload->direction = sample_lambertian(shading_normal, &payload->seed);
    payload->last_bounce_was_specular = false;
    payload->active = 1;
}

extern "C" __global__ void __closesthit__spherical_mixed() {
    const SphericalMixedSbtData* material = (SphericalMixedSbtData*)optixGetSbtDataPointer();
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    float t_hit = optixGetRayTmax();
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = ray_orig + t_hit * ray_dir;

    float3 geometric_normal = normalize(hit_point - material->center);
    float3 shading_normal = dot(ray_dir, geometric_normal) < 0 ? geometric_normal : -geometric_normal;

    // NEE: only diffuse component (specular is delta function)
    if (params.use_nee) {
        double brdf_reflectance = material->diffuse_ratio * material->reflectance * INV_PI;
        accumulate_nee_contribution(hit_point, shading_normal, brdf_reflectance, payload->weight, 1);
    }

    // Max bounce termination
    atomicAdd(params.total_bounces_buffer, 1ull);
    payload->bounce_count++;
    if (payload->bounce_count >= params.max_bounces) {
        payload->active = 0;
        return;
    }

    // BRDF attenuation
    payload->weight *= material->reflectance;

    // Russian Roulette
    if (!apply_russian_roulette(payload)) {
        return;
    }

    // Choose between diffuse and specular
    float u = random_float(&payload->seed);
    float3 new_dir;
    bool is_specular;

    if (u < material->diffuse_ratio) {
        new_dir = sample_lambertian(shading_normal, &payload->seed);
        is_specular = false;
    } else {
        new_dir = ray_dir - 2.0f * dot(ray_dir, shading_normal) * shading_normal;
        new_dir = normalize(new_dir);
        is_specular = true;
    }

    payload->origin = hit_point + shading_normal * 1e-4f;
    payload->direction = new_dir;
    payload->last_bounce_was_specular = is_specular;
    payload->active = 1;
}
