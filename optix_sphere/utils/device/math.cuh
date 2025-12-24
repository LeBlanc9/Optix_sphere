#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// This is the primary CUDA-side math header.
// It includes all host/device functions, operators, and constants.
// It should only be included by .cu files or other .cuh files.

namespace phonder {

// ============================================================================
// Mathematical Constants
// ============================================================================
constexpr float PI = 3.1415926535897932f;
constexpr float M_PI = 3.1415926535897932f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float EPSILON = 1e-6f;

// ============================================================================
// Vector Operations
// ============================================================================

__device__ __host__ __inline__
float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ __inline__
void operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ __host__ __forceinline__ 
float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ __forceinline__ 
float3 operator-(const float3& v) {
    return make_float3(-v.x, -v.y, -v.z);
}

__device__ __host__ __forceinline__ 
float3 operator*(const float3& v, float t) {
    return make_float3(v.x * t, v.y * t, v.z * t);
}

__device__ __host__ __forceinline__ 
float3 operator*(float t, const float3& v) {
    return v * t;
}

__device__ __host__ __inline__
void operator*=(float3& vec, float scalar) {
    vec.x *= scalar;
    vec.y *= scalar;
    vec.z *= scalar;
}

__device__ __host__ __inline__
float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__ __inline__
float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ __forceinline__ 
float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ __host__ __forceinline__ 
float length(const float3& v) {
    return sqrtf(dot(v, v));
}

// NOTE: normalize and normalized use rsqrtf, a device-only intrinsic.
// The __host__ path will be slower or may not compile with non-NVCC compilers
// if not handled carefully, but for internal .cu file usage it's fine.
__device__ __host__ __forceinline__ 
void normalize(float3& v) {
    float inv_len = rsqrtf(dot(v, v));
    v.x *= inv_len;
    v.y *= inv_len;
    v.z *= inv_len;
}

__device__ __host__ __forceinline__ 
float3 normalized(const float3& v) {
    float inv_len = rsqrtf(dot(v, v));
    return make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
}


__device__ __host__ __forceinline__
float distance(const float3& a, const float3& b) {
    float3 diff = a - b;
    return length(diff);
}

__device__ __host__ __forceinline__
float distance_sq(const float3& a, const float3& b) {
    float3 diff = a - b;
    return dot(diff, diff);
}



// ============================================================================
// Trigonometric Utilities
// ============================================================================

__device__ __host__ __forceinline__
float deg2rad(float deg) {
    return deg * PI / 180.0f;
}

__device__ __host__ __forceinline__
float rad2deg(float rad) {
    return rad * 180.0f / PI;
}


__device__ __host__ __forceinline__
float3 spherical_to_cartesian(float theta, float phi) {
    theta = deg2rad(theta);
    phi = deg2rad(phi);

    float3 result;
    result.x = sinf(theta) * cosf(phi);
    result.y = sinf(theta) * sinf(phi);
    result.z = cosf(theta);
    return result;
}



__device__ __host__ __forceinline__
float sin2cos(float sin_theta) {
    return sqrtf(fmaxf(0.0f, 1.0f - sin_theta * sin_theta));
}

__device__ __host__ __forceinline__
float cos2sin(float cos_theta) {
    return sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
}

} // namespace phonder