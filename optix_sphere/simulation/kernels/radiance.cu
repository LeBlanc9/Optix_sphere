#include "kernel_utils.cuh"

// 探测器命中程序（简单：记录能量并终止）
extern "C" __global__ void __closesthit__detector() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    // 理论正确的 NEE 实现：区分直接和间接照明
    if (payload->bounce_count == 0) {
        // 直接照明：光源直接击中 detector（无论是否使用 NEE 都计数）
        // 因为 NEE 只在表面反弹时采样，不会捕获直接照明
        atomicAdd(params.flux_buffer, payload->weight);
        atomicAdd(params.detected_rays_buffer, 1ull);
    } else {
        // 间接照明：至少经过一次反弹
        if (!params.use_nee) {
            // 非 NEE 模式：通过隐式路径计数
            atomicAdd(params.flux_buffer, payload->weight);
            atomicAdd(params.detected_rays_buffer, 1ull);
        }
        // NEE 模式：不计数隐式路径（已通过 shadow ray 显式采样）
        // 避免双重计数
    }

    payload->active = 0;
}

// 最近命中程序（球体反射）
extern "C" __global__ void __closesthit__sphere() {
    const SphereSbtData* sphere = (SphereSbtData*)optixGetSbtDataPointer();
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    // 1. 计算命中点和法线
    float t_hit = optixGetRayTmax();
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = ray_orig + t_hit * ray_dir;
    float3 geometric_normal = normalize(hit_point - sphere->center);

    // 法线应该总是与射线方向相反（用于内部反射）
    float3 shading_normal = dot(ray_dir, geometric_normal) < 0 ? geometric_normal : -geometric_normal;

    // 2. Next Event Estimation (NEE): 显式采样探测器
    if (params.use_nee) {
        // 计算朝向探测器的方向和距离
        float3 to_detector = params.detector.position - hit_point;
        float distance = length(to_detector);
        float3 dir_to_detector = to_detector * (1.0f / distance);

        // 检查探测器是否在表面法线的正半球内
        float cos_theta_surface = dot(dir_to_detector, shading_normal);
        if (cos_theta_surface > 0.0f) {
            // 检查探测器法线方向（探测器只接收正面入射的光）
            float cos_theta_detector = dot(-dir_to_detector, params.detector.normal);

            if (cos_theta_detector > 0.0f) {
                // 计算几何因子 (立体角投影)
                float detector_area = M_PIf * params.detector.radius * params.detector.radius;
                float geometric_factor = (detector_area * cos_theta_detector) / (distance * distance);

                // Lambertian BRDF: ρ/π * cos(θ)
                double brdf_cosine = sphere->reflectance * INV_PI * cos_theta_surface;

                // NEE 贡献 = weight * BRDF * 几何因子
                double nee_contribution = payload->weight * brdf_cosine * geometric_factor;

                // 发射 shadow ray 检查可见性
                ShadowPayload shadow_payload;
                shadow_payload.occluded = false;

                unsigned long long shadow_ptr = reinterpret_cast<unsigned long long>(&shadow_payload);
                unsigned int s0 = static_cast<unsigned int>(shadow_ptr);
                unsigned int s1 = static_cast<unsigned int>(shadow_ptr >> 32);

                optixTrace(
                    params.traversable,
                    hit_point + shading_normal * 1e-4f,  // 起点（避免自相交）
                    dir_to_detector,                      // 方向
                    1e-4f,                                // tmin
                    distance - 1e-4f,                     // tmax（到探测器的距离）
                    0.0f,                                 // rayTime
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, // 遮挡测试，第一次击中即停止
                    2,                                    // SBT offset (shadow rays start at index 2)
                    1,                                    // SBT stride (1 record per primitive)
                    1,                                    // missSBTIndex (shadow miss is at index 1)
                    s0, s1                                // payload
                );

                // 如果未被遮挡，累积贡献
                if (!shadow_payload.occluded) {
                    atomicAdd(params.flux_buffer, nee_contribution);
                }
            }
        }
    }

    // 3. 反射次数检查 & 俄罗斯轮盘赌
    atomicAdd(params.total_bounces_buffer, 1ull); // Increment total bounces
    payload->bounce_count++;
    if (payload->bounce_count >= params.max_bounces) {
        payload->active = 0;
        return;
    }
    float survival_prob = sphere->reflectance;
    if (random_float(&payload->seed) >= survival_prob) {
        payload->active = 0;
        return;
    }

    // 4. 更新光线状态以进行下一次反弹（间接光照路径）
    payload->weight *= sphere->reflectance / survival_prob; // 更新权重（无偏估计）
    payload->origin = hit_point + shading_normal * 1e-3f; // 避免自相交 (1微米 in mm)
    payload->direction = sample_lambertian(shading_normal, &payload->seed);
    payload->active = 1;
}

// Miss 程序（光线逃逸）
extern "C" __global__ void __miss__sphere() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);
    payload->active = 0;
}

// ============================================
// Triangle mesh closest-hit programs
// ============================================

// Lambertian material (unified for all diffuse surfaces)
// Supports variable reflectance (0-1) for walls, baffles, etc.
// NEE is controlled globally via params.use_nee
extern "C" __global__ void __closesthit__lambertian() {
    // SBT data contains reflectance and sphere center
    const SphereWallSbtData* material = (SphereWallSbtData*)optixGetSbtDataPointer();
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    // Get hit point
    float t_hit = optixGetRayTmax();
    float3 ray_orig = optixGetWorldRayOrigin();
    float3 ray_dir = optixGetWorldRayDirection();
    float3 hit_point = ray_orig + t_hit * ray_dir;

    // Spherical normal: from sphere center to hit point
    float3 geometric_normal = normalize(hit_point - material->center);

    // Ensure normal faces the ray origin (interior reflection)
    float3 shading_normal = dot(ray_dir, geometric_normal) < 0 ? geometric_normal : -geometric_normal;

    // Next Event Estimation (NEE): explicit detector sampling
    if (params.use_nee) {
        float3 to_detector = params.detector.position - hit_point;
        float distance = length(to_detector);
        float3 dir_to_detector = to_detector * (1.0f / distance);

        float cos_theta_surface = dot(dir_to_detector, shading_normal);
        if (cos_theta_surface > 0.0f) {
            float cos_theta_detector = dot(-dir_to_detector, params.detector.normal);

            if (cos_theta_detector > 0.0f) {
                float detector_area = M_PIf * params.detector.radius * params.detector.radius;
                float geometric_factor = (detector_area * cos_theta_detector) / (distance * distance);

                // Lambertian BRDF: ρ/π * cos(θ)
                double brdf_cosine = material->reflectance * INV_PI * cos_theta_surface;
                double nee_contribution = payload->weight * brdf_cosine * geometric_factor;

                // Shadow ray for visibility test
                ShadowPayload shadow_payload;
                shadow_payload.occluded = false;

                unsigned long long shadow_ptr = reinterpret_cast<unsigned long long>(&shadow_payload);
                unsigned int s0 = static_cast<unsigned int>(shadow_ptr);
                unsigned int s1 = static_cast<unsigned int>(shadow_ptr >> 32);

                optixTrace(
                    params.traversable,
                    hit_point + shading_normal * 1e-4f,
                    dir_to_detector,
                    1e-4f,
                    distance - 1e-4f,
                    0.0f,
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                    4,  // SBT offset (shadow records start at index 4)
                    1,  // SBT stride = 1: each material has its own shadow record
                    1,  // missSBTIndex
                    s0, s1
                );

                if (!shadow_payload.occluded) {
                    atomicAdd(params.flux_buffer, nee_contribution);
                }
            }
        }
    }

    // Bounce count check & Russian roulette
    atomicAdd(params.total_bounces_buffer, 1ull);
    payload->bounce_count++;
    if (payload->bounce_count >= params.max_bounces) {
        payload->active = 0;
        return;
    }

    // Russian roulette based on material reflectance
    float survival_prob = material->reflectance;
    if (random_float(&payload->seed) > survival_prob) {
        payload->active = 0;
        return;
    }

    // Update ray state for next bounce (indirect illumination path)
    payload->weight *= material->reflectance / survival_prob;
    payload->origin = hit_point + shading_normal * 1e-3f;
    payload->direction = sample_lambertian(shading_normal, &payload->seed);
    payload->active = 1;
}

// Absorber material (perfect light absorber / black body)
// Used for port holes, black surfaces, light traps
extern "C" __global__ void __closesthit__absorber() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    RayPayload* payload = reinterpret_cast<RayPayload*>(payload_ptr);

    // 完全吸收，光线终止
    payload->active = 0;
}
