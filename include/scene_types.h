#pragma once

#include <vector_types.h>
#include "constants.h"

// Using C-style structs for simple data aggregation and easy
// transfer to the device-side structs if needed.
//
// 单位约定：所有长度单位使用毫米(mm)，符合光学系统惯例

// 理想积分球
struct Sphere {
    float3 center = {0.0f, 0.0f, 0.0f};  // mm
    float radius = 50.0f;                 // mm (default: 50mm diameter sphere)
    float reflectance = 0.99f;            // dimensionless [0,1]
};

// 各向同性点光源
struct LightSource {
    float3 position = {0.0f, 0.0f, 0.0f}; // mm
    float power = 1.0f;                    // W
};

// 简单的圆形平面探测器
struct Detector {
    float3 position = {50.0f, 0.0f, 0.0f}; // mm (on sphere surface by default)
    float3 normal = {-1.0f, 0.0f, 0.0f};   // direction (normalized)
    float radius = 0.564f;                 // mm (area = pi*r^2 = 1 mm^2)
};

// 模拟配置参数
struct SimConfig {
    int num_rays = 1'000'000;
    int max_bounces = 50;
    bool use_nee = false;           // 是否启用Next Event Estimation (默认关闭)
    unsigned int random_seed = 0;   // 随机数种子 (0 = 使用时间，非0 = 固定种子)
};

// 辅助函数：根据开孔参数计算弦面探测器的几何位置
// 这个函数计算在球面上开一个圆形孔后，探测器应该放在哪里
//
// 参数：
//   detector: 要配置的探测器对象（输出）
//   sphere: 积分球参数
//   port_hole_radius_mm: 开孔半径 (mm)
//
// 自动计算：
//   - 内陷深度 d = R - √(R² - r_hole²)
//   - 探测器位置 = (R-d, 0, 0)
//   - 探测器半径 = r_hole
//
inline void configure_detector_chord(
    Detector& detector,
    const Sphere& sphere,
    float port_hole_radius_mm  // 开孔半径 (mm)
) {
    // 1. 开孔半径
    float r_hole = port_hole_radius_mm;

    // 2. 计算内陷深度：d = R - √(R² - r_hole²)
    // 这是球心到弦平面的距离
    float R = sphere.radius;
    float sqrt_term = sqrtf(R * R - r_hole * r_hole);
    float inset_depth = R - sqrt_term;

    // 3. 探测器中心位置（假设探测器朝向球心，位于x轴正方向）
    detector.position.x = sqrt_term;  // = R - d
    detector.position.y = 0.0f;
    detector.position.z = 0.0f;

    // 4. 探测器法线（指向球心）
    detector.normal.x = -1.0f;
    detector.normal.y = 0.0f;
    detector.normal.z = 0.0f;

    // 5. 探测器半径应该等于开孔半径（物理正确）
    // 如果需要安全边距，应该在intersection检查中处理，而不是增大探测器
    detector.radius = r_hole;
}
