#pragma once

// AABB and other common types might be needed here eventually,
// but for now, let's keep it focused on results.

// 模拟结果
struct SimulationResult {
    double detected_flux;       // 探测器收集的总通量 (W) - double精度
    double irradiance;          // 辐照度 (W/m²) - double精度
    int total_rays;             // 发射的总光线数
    int detected_rays;          // 击中探测器的光线数
    double avg_bounces;         // 平均反射次数 - double精度
};

// 理论结果
struct TheoryResult {
    double avg_irradiance;       // 平均辐照度 (W/m²) - double精度
    double detected_flux;        // 理论上探测器应接收的通量 (W) - double精度
    double sphere_area;          // 球面面积 (m²) - double精度
    double total_flux_in_sphere; // 球内总通量 - double精度
};
