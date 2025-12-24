#include "kernel_utils.cuh"

// ============================================
// Ray generation program (procedural mode)
// ============================================
// Generates rays from a point light source with isotropic emission
extern "C" __global__ void __raygen__forward_trace() {
    const uint3 idx = optixGetLaunchIndex();
    unsigned int seed = params.seed_buffer[idx.x];

    // 从光源均匀采样方向 (各向同性点光源)
    float u1 = random_float(&seed);
    float u2 = random_float(&seed);
    float theta = 2.0f * M_PIf * u1;
    float phi = acosf(2.0f * u2 - 1.0f);
    float3 direction = make_float3(sinf(phi) * cosf(theta), sinf(phi) * sinf(theta), cosf(phi));

    // 初始化 payload (无量纲weight)
    RayPayload payload;
    payload.origin = params.light_source.position;
    payload.direction = direction;
    payload.weight = 1.0f;  // 初始权重=1 (无量纲)
    payload.bounce_count = 0;
    payload.active = 1;
    payload.seed = seed;

    // 追踪光线 (循环多次反射)
    for (int i = 0; i < params.max_bounces && payload.active; ++i) {
        unsigned long long payload_ptr = reinterpret_cast<unsigned long long>(&payload);
        unsigned int p0 = static_cast<unsigned int>(payload_ptr);
        unsigned int p1 = static_cast<unsigned int>(payload_ptr >> 32);
        optixTrace(
            params.traversable,
            payload.origin, payload.direction,
            1e-4f, 1e16f, 0.0f,
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0, // SBT offsets
            p0, p1
        );
    }

    params.seed_buffer[idx.x] = payload.seed;
}

// ============================================
// Ray generation program (data-driven mode)
// ============================================
// 从外部 SoA 格式光子数组读取初始状态
// 支持混合模拟 (e.g., MCX/MCML 体散射 → OptiX 表面传输)
extern "C" __global__ void __raygen__data_driven() {
    // 1. Get the photon index from launch ID
    const unsigned int photon_index = optixGetLaunchIndex().x;

    // 2. Bounds check (safety measure)
    if (photon_index >= params.num_input_photons) {
        return;  // Out of bounds, do nothing
    }

    // 3. Load photon data from SoA arrays (GPU合并访问友好)
    const float3 position = params.photon_positions[photon_index];
    const float3 direction = params.photon_directions[photon_index];
    const double weight = params.photon_weights[photon_index];
    const unsigned int seed = params.photon_seed_base + photon_index * 97;

    // 4. Initialize RayPayload from input photon
    RayPayload payload;
    payload.origin = position;
    payload.direction = direction;
    payload.weight = weight;
    payload.bounce_count = 0;  // Reset bounce count for surface simulation
    payload.active = 1;
    payload.seed = seed;

    // 5. Trace the ray (same as procedural mode)
    for (int i = 0; i < params.max_bounces && payload.active; ++i) {
        unsigned long long payload_ptr = reinterpret_cast<unsigned long long>(&payload);
        unsigned int p0 = static_cast<unsigned int>(payload_ptr);
        unsigned int p1 = static_cast<unsigned int>(payload_ptr >> 32);

        optixTrace(
            params.traversable,
            payload.origin,
            payload.direction,
            1e-4f,      // tmin
            1e16f,      // tmax
            0.0f,       // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE,
            0,          // SBT ray type offset
            1,          // SBT ray type stride
            0,          // missSBTIndex
            p0, p1      // payload
        );
    }

    // 6. (Optional) Write back final seed if needed
    if (photon_index < params.num_rays && params.seed_buffer != nullptr) {
        params.seed_buffer[photon_index] = payload.seed;
    }
}
