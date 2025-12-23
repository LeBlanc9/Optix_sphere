#include "kernel_utils.cuh"

// Shadow ray miss 程序（未遮挡）
extern "C" __global__ void __miss__shadow() {
    // Shadow ray 未击中任何物体，表示未被遮挡
    // occluded 已经初始化为 false，无需操作
}

// Shadow ray 球面遮挡检测（any-hit 程序）
extern "C" __global__ void __anyhit__sphere_shadow() {
    // Shadow ray 击中球面，表示被遮挡
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    ShadowPayload* shadow_payload = reinterpret_cast<ShadowPayload*>(payload_ptr);
    shadow_payload->occluded = true;
    optixTerminateRay();  // 提前终止
}

// Shadow ray 探测器遮挡检测（any-hit 程序）
extern "C" __global__ void __anyhit__detector_shadow() {
    // Shadow ray 击中探测器本身，这实际上不应该发生
    // 因为 tmax 设置在探测器之前，但保险起见标记为未遮挡
    // （实际上这意味着光线到达了探测器）
    // 终止 ray 避免继续追踪
    optixTerminateRay();
}

// Shadow ray any-hit for Lambertian materials (walls, baffles, etc.)
// Marks ray as occluded and terminates
extern "C" __global__ void __anyhit__lambertian_shadow() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    ShadowPayload* shadow_payload = reinterpret_cast<ShadowPayload*>(payload_ptr);
    shadow_payload->occluded = true;
    optixTerminateRay();
}

// Shadow ray any-hit for absorber materials (port holes, black surfaces)
// Marks ray as occluded and terminates
extern "C" __global__ void __anyhit__absorber_shadow() {
    unsigned long long payload_ptr = static_cast<unsigned long long>(optixGetPayload_0()) |
                                     (static_cast<unsigned long long>(optixGetPayload_1()) << 32);
    ShadowPayload* shadow_payload = reinterpret_cast<ShadowPayload*>(payload_ptr);
    shadow_payload->occluded = true;
    optixTerminateRay();
}
