#pragma once

#include <vector_types.h>

// Using C-style structs for simple data aggregation and easy
// transfer to the device-side structs if needed.

// 理想积分球
struct Sphere {
    float3 center = {0.0f, 0.0f, 0.0f};
    float radius = 0.05f; // 50mm radius
    float reflectance = 0.99f;
};

// 各向同性点光源
struct LightSource {
    float3 position = {0.0f, 0.0f, 0.0f};
    float power = 1.0f; // 1 W
};

// 简单的圆形平面探测器
struct Detector {
    float3 position = {0.049f, 0.0f, 0.0f};
    float3 normal = {-1.0f, 0.0f, 0.0f};
    float radius = 0.001f; // 1mm radius, area is pi*r^2
};

// 模拟配置参数
struct SimConfig {
    int num_rays = 1'000'000;
    int max_bounces = 50;
};
