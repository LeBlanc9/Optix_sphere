#pragma once

#include <vector_types.h>
#include <cmath>

// 单位约定：所有长度单位使用毫米(mm)，符合光学系统惯例

// 理想积分球
struct Sphere {
    float3 center = {0.0f, 0.0f, 0.0f};  // mm
    float radius = 50.0f;                 // mm (default: 50mm diameter sphere)
    float reflectance = 0.99f;            // dimensionless [0,1]
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


struct MeshSceneConfig {
    // 预留给未来扩展（例如：全局缩放、坐标系转换等）
};

