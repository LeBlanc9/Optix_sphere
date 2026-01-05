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
