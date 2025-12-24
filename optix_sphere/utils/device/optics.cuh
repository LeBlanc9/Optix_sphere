#pragma once
#include <curand_kernel.h>
#include "./math.cuh"

namespace phonder {
namespace optics {

// ============================================================================
// Fresnel Reflection/Refraction
// ============================================================================

/**
 * @brief 计算菲涅尔反射系数
 * 
 * @param n1 入射介质折射率
 * @param n2 折射介质折射率  
 * @param cos_i 入射角余弦值
 * @return 反射系数 [0, 1]
 */
__device__ __forceinline__ float fresnel_reflectance(float n1, float n2, float cos_i) {
    const float eta = __fdividef(n1, n2);
    const float cos_i2 = cos_i * cos_i;
    const float sin2_t = eta * eta * (1.0f - cos_i2);
    
    // Total internal reflection
    if (sin2_t >= 1.0f) {
        return 1.0f;
    }
    
    const float cos_t = sqrtf(1.0f - sin2_t);
    
    // Fresnel coefficients for s and p polarization
    const float rs = __fdividef(n1 * cos_i - n2 * cos_t, n1 * cos_i + n2 * cos_t);
    const float rp = __fdividef(n1 * cos_t - n2 * cos_i, n1 * cos_t + n2 * cos_i);
    
    return 0.5f * (rs * rs + rp * rp);
}

/**
 * @brief 计算反射方向 - 就地修改版本
 * 
 * @param [in,out] direction 入射光线方向向量
 * @param [in] normal 界面法线向量（单位向量）
 */
__device__ __host__ __forceinline__ 
void reflect(float3& direction, const float3& normal) {
    direction = direction - 2.0f * dot(direction, normal) * normal;
    normalize(direction);
}

/**
 * @brief 计算反射方向 - 返回新值版本
 * 
 * @param [in] direction 入射光线方向向量
 * @param [in] normal 界面法线向量（单位向量）
 * @return 反射后的方向向量
 */
__device__ __host__ __forceinline__ 
float3 reflected(const float3& direction, const float3& normal) {
    float3 result = direction - 2.0f * dot(direction, normal) * normal;
    return normalized(result);
}


/**
 * @brief 通用折射函数 - 就地修改版本，适用于任意法线
 * 
 * @param [in,out] direction 入射光线方向向量
 * @param [in] normal 界面法线向量（单位向量）
 * @param [in] ni 入射介质折射率
 * @param [in] nt 折射介质折射率
 */
__device__ __forceinline__ void refract(float3& direction, const float3& normal, float ni, float nt) {
    float cos_theta_i = -dot(direction, normal);
    float eta = __fdividef(ni, nt);
    
    float sin_theta_i_squared = 1.0f - cos_theta_i * cos_theta_i;
    float sin_theta_t_squared = eta * eta * sin_theta_i_squared;
    
    if (sin_theta_t_squared > 1.0f) {
        // Total internal reflection
        direction = direction - 2.0f * dot(direction, normal) * normal;
        normalize(direction);
        return;
    }
    
    float cos_theta_t = sqrtf(1.0f - sin_theta_t_squared);
    direction = (direction - normal * cos_theta_i) * eta - normal * cos_theta_t;
    normalize(direction);
}

/**
 * @brief 通用折射函数 - 返回新值版本，适用于任意法线
 * 
 * @param [in] direction 入射光线方向向量
 * @param [in] normal 界面法线向量（单位向量）
 * @param [in] ni 入射介质折射率
 * @param [in] nt 折射介质折射率
 * @return 折射后的方向向量（全反射时返回反射方向）
 */
__device__ __forceinline__ float3 refracted(const float3& direction, const float3& normal, float ni, float nt) {
    float cos_theta_i = -dot(direction, normal);
    float eta = __fdividef(ni, nt);
    
    float sin_theta_i_squared = 1.0f - cos_theta_i * cos_theta_i;
    float sin_theta_t_squared = eta * eta * sin_theta_i_squared;
    
    float3 result;
    if (sin_theta_t_squared > 1.0f) {
        // Total internal reflection
        result = direction - 2.0f * dot(direction, normal) * normal;
    } else {
        float cos_theta_t = sqrtf(1.0f - sin_theta_t_squared);
        result = (direction - normal * cos_theta_i) * eta - normal * cos_theta_t;
    }
    
    return normalized(result);
}



/**
 * @brief Z轴法线专用折射函数
 * 
 * @param[in,out] direction 光线方向
 * @param n1 入射介质折射率
 * @param n2 折射介质折射率
 * 
 * @note 仅适用于界面法线为±z方向的情况
 */
__device__ inline void refract_z_axis(float3& direction, float n1, float n2) {
    float eta = n1 / n2;
    float cos_i = fabsf(direction.z);
    float sin2_i = 1.0f - cos_i * cos_i;
    float sin2_t = eta * eta * sin2_i;
    
    if (sin2_t <= 1.0f) {
        float cos_t = sqrtf(1.0f - sin2_t);
        direction.x *= eta;
        direction.y *= eta;
        direction.z = copysignf(cos_t, direction.z);
        normalize(direction);
    }
}

// ============================================================================
// Scattering Phase Functions
// ============================================================================

/**
 * @brief Henyey-Greenstein 相位函数采样
 * 
 * @param g 各向异性参数 [-1, 1]
 * @param rng_state 随机数生成器状态
 * @return 散射角余弦值
 */
__device__ inline float henyey_greenstein(float g, curandState* rng_state) {
    float rand = curand_uniform(rng_state);
    
    if (fabsf(g) < EPSILON) {
        // 各向同性散射
        return 2.0f * rand - 1.0f;
    }
    
    float g2 = g * g;
    float temp = __fdividef(1.0f - g2, 1.0f - g + 2.0f * g * rand);
    return __fdividef(1.0f + g2 - temp * temp, 2.0f * g);
}

/**
 * @brief 更新散射方向 - 就地修改版本
 * 
 * @param[in,out] direction 光子方向（会被修改）
 * @param g 各向异性参数
 * @param rng_state 随机数生成器状态
 */
__device__ inline void scatter_direction(float3& direction, float g, curandState* rng_state) {
    // Random azimuthal angle
    float phi = TWO_PI * curand_uniform(rng_state);
    float sin_phi, cos_phi;
    __sincosf(phi, &sin_phi, &cos_phi);
    
    // Random deflection angle using Henyey-Greenstein
    float cos_theta = henyey_greenstein(g, rng_state);
    float sin_theta = cos2sin(cos_theta);
    
    // Update direction using Malley's method
    float temp = sqrtf(1.0f - direction.z * direction.z);
    
    if (temp == 0.0f) {
        // Special case: direction is along z-axis
        direction.x = sin_theta * cos_phi;
        direction.y = sin_theta * sin_phi;
        direction.z = copysignf(cos_theta, direction.z * cos_theta);
    } else {
        float temp_dx = direction.x;
        direction.x = __fdividef(sin_theta * (direction.x * direction.z * cos_phi - direction.y * sin_phi), temp) + direction.x * cos_theta;
        direction.y = __fdividef(sin_theta * (direction.y * direction.z * cos_phi + temp_dx * sin_phi), temp) + direction.y * cos_theta;
        direction.z = -sin_theta * cos_phi * temp + direction.z * cos_theta;
    }
    
    normalize(direction);
}

/**
 * @brief 计算散射方向 - 返回新值版本
 * 
 * @param[in] direction 入射光子方向
 * @param g 各向异性参数
 * @param rng_state 随机数生成器状态
 * @return 散射后的方向向量
 */
__device__ inline float3 scattered_direction(const float3& direction, float g, curandState* rng_state) {
    float3 result = direction;
    scatter_direction(result, g, rng_state);
    return result;
}

// ============================================================================
// Random Direction Sampling
// ============================================================================

/**
 * @brief 生成朗伯表面的随机反射方向
 * 
 * @param normal 表面法线
 * @param rng_state 随机数生成器状态
 * @return 反射方向
 */
__device__ inline float3 random_lambertian_direction(const float3& normal, curandState* rng_state) {
    float u1 = curand_uniform(rng_state);
    float u2 = curand_uniform(rng_state);
    
    // Cosine-weighted hemisphere sampling
    float cos_theta = sqrtf(u1);
    float sin_theta = sqrtf(1.0f - u1);
    float phi = TWO_PI * u2;
    
    float sin_phi, cos_phi;
    __sincosf(phi, &sin_phi, &cos_phi);
    
    // Local coordinates
    float3 local_dir = make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
    
    // Transform to world coordinates
    // Create orthonormal basis with normal as z-axis
    float3 w = normal;
    float3 u = (fabsf(w.x) > 0.1f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    u = normalized(cross(u, w));
    float3 v = normalized(cross(w, u));
    
    return local_dir.x * u + local_dir.y * v + local_dir.z * w;
}

} // namespace optics
} // namespace phonder