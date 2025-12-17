#pragma once

// AABB and other common types might be needed here eventually,
// but for now, let's keep it focused on results.

// 模拟结果
struct SimulationResult {
    float detected_flux;        // 探测器收集的总通量 (W)
    float irradiance;           // 辐照度 (W/m²)
    int total_rays;             // 发射的总光线数
    int detected_rays;          // 击中探测器的光线数
    float avg_bounces;          // 平均反射次数
};

// 理论结果
struct TheoryResult {
    float avg_irradiance;       // 平均辐照度 (W/m²)
    float sphere_area;          // 球面面积 (m²)
    float total_flux_in_sphere; // 球内总通量
};
