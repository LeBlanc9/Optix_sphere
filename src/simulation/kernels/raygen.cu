#include "kernel_utils.cuh"

// Ray generation 程序
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
