#include "kernel_utils.cuh"

// 圆盘相交函数（用于探测器）
extern "C" __global__ void __intersection__disk() {
    const DiskSbtData* disk = (DiskSbtData*)optixGetSbtDataPointer();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    // 圆盘定义：中心在 disk->center，法线为 disk->normal
    float3 oc = ray_orig - disk->center;
    float denom = dot(ray_dir, disk->normal);

    // 检查射线是否平行于圆盘平面
    if (fabsf(denom) > 1e-6f) {
        float t = -dot(oc, disk->normal) / denom;

        if (t >= ray_tmin && t <= ray_tmax) {
            // 计算交点
            float3 hit_point = ray_orig + t * ray_dir;
            float3 to_center = hit_point - disk->center;
            float dist_sq = dot(to_center, to_center);

            // 检查是否在圆盘半径内
            if (dist_sq <= disk->radius * disk->radius) {
                optixReportIntersection(t, 0);
            }
        }
    }
}

// 解析球面相交函数
extern "C" __global__ void __intersection__sphere() {
    const SphereSbtData* sphere = (SphereSbtData*)optixGetSbtDataPointer();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    float3 oc = ray_orig - sphere->center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere->radius * sphere->radius;

    float discriminant = b * b - 4.0f * a * c;
    if (discriminant >= 0.0f) {
        float sqrt_d = sqrtf(discriminant);
        float t1 = (-b - sqrt_d) / (2.0f * a);
        float t2 = (-b + sqrt_d) / (2.0f * a);
        float t = t1;
        if (t < ray_tmin) t = t2;
        if (t >= ray_tmin && t <= ray_tmax) {
            // 计算击中点
            float3 hit_point = ray_orig + t * ray_dir;

            // 排除探测器区域：检查击中点是否在探测器圆盘内
            float3 to_detector = hit_point - params.detector.position;
            float dist_to_detector_sq = dot(to_detector, to_detector);

            // 如果击中点在探测器半径内，不报告球面相交（让探测器处理）
            if (dist_to_detector_sq > params.detector.radius * params.detector.radius) {
                optixReportIntersection(t, 0);
            }
        }
    }
}
