#pragma once

#include <optix.h>
#include "simulation/device_params.h"
#include "constants.h"

// Launch params (passed from CPU)
extern "C" {
    __constant__ DeviceParams params;
}

// Vector math helpers
__device__ __forceinline__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 operator-(const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator*(const float3& a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 operator*(float s, const float3& a) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float length(const float3& v) { return sqrtf(dot(v, v)); }
__device__ __forceinline__ float3 normalize(const float3& v) { float len = length(v); return len > 0.0f ? v * (1.0f / len) : v; }
__device__ __forceinline__ float3 cross(const float3& a, const float3& b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }

// CUDA 随机数生成器 (简单的 LCG)
__device__ float random_float(unsigned int* seed) {
    *seed = (*seed * 1664525u + 1013904223u);
    return (float)(*seed) / (float)0xFFFFFFFFu;
}

// 生成随机整数 [0, max)
__device__ unsigned int random_uint(unsigned int* seed, unsigned int max) {
    *seed = (*seed * 1664525u + 1013904223u);
    return *seed % max;
}

// 生成漫反射方向 (Lambertian BRDF)
__device__ float3 sample_lambertian(const float3& normal, unsigned int* seed) {
    float u1 = random_float(seed);
    float u2 = random_float(seed);
    float r = sqrtf(u1);
    float theta = 2.0f * M_PIf * u2;
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(1.0f - u1);
    float3 up = fabsf(normal.z) < 0.999f ? make_float3(0.0f, 0.0f, 1.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}
