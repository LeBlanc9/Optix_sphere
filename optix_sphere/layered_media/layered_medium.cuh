#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>


#define BLOCK_SIZE 256
#define MAX_LAYERS 10


namespace phonder {


struct Layer {
    float n;        // 折射率
    float mua;      // 吸收系数 (mm^-1)
    float mus;      // 散射系数 (mm^-1)
    float g;        // 各向异性因子 [-1, 1]
    float d;        // 层厚度 (mm)
    float inv_mus;  // 1/mus 预计算

    Layer() = default;
    __host__ __device__ Layer(float n, float mua, float mus, float g, float d)
        : n(n), mua(mua), mus(mus), g(g), d(d), inv_mus(1.0f / mus) {}
};


struct LayeredMedium {
    float ambient_n;                      // 背景折射率
    Layer layers[MAX_LAYERS];             // 组织层
    float layer_boundaries[MAX_LAYERS][2];
    int num_layers;
    float width;
    float total_thickness;

    LayeredMedium() = default;

    // Builder 模式构造
    __host__ __device__ LayeredMedium(float ambient_n, float width = 100.0f)
        : ambient_n(ambient_n), num_layers(0), width(width), total_thickness(0.0f) {}

    __host__ __device__ LayeredMedium& add_layer(float n, float mua, float mus, float g, float d) {
        if (num_layers >= MAX_LAYERS) return *this;
        if (mus <= 0.0f || mua < 0.0f || d <= 0.0f) return *this;  // 参数检查

        layers[num_layers] = Layer(n, mua, mus, g, d);
        layer_boundaries[num_layers][0] = total_thickness;
        total_thickness += d;
        layer_boundaries[num_layers][1] = total_thickness;
        num_layers++;
        return *this;
    }

    __host__ __device__ LayeredMedium& set_width(float w) {
        width = w;
        return *this;
    }

    __host__ __device__ LayeredMedium& set_ambient_n(float n) {
        ambient_n = n;
        return *this;
    }

    __device__ int get_layer_index(float z) const {
        if (z < 0.0f || z >= total_thickness) return -1;
        for (int i = 0; i < num_layers; i++) {
            if (z >= layer_boundaries[i][0] && z < layer_boundaries[i][1]) {
                return i;
            }
        }
        return -1;
    }

    __device__ bool intersect(const float3& origin, const float3& direction,
                            float& t, float3& normal) const {
        int layer_idx = get_layer_index(origin.z);
        if (layer_idx < 0) return false;

        float3 box_min = make_float3(-width/2, -width/2, layer_boundaries[layer_idx][0]);
        float3 box_max = make_float3(width/2, width/2, layer_boundaries[layer_idx][1]);
        float3 inv_dir = make_float3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);

        float tmax_x = fmaxf((box_min.x - origin.x) * inv_dir.x, (box_max.x - origin.x) * inv_dir.x);
        float tmax_y = fmaxf((box_min.y - origin.y) * inv_dir.y, (box_max.y - origin.y) * inv_dir.y);
        float tmax_z = fmaxf((box_min.z - origin.z) * inv_dir.z, (box_max.z - origin.z) * inv_dir.z);

        float t_exit = fminf(fminf(tmax_x, tmax_y), tmax_z);
        if (t_exit < 0.0f) return false;

        t = t_exit;
        if (fabsf(t - tmax_x) < 1e-5f) {
            normal = make_float3(direction.x > 0 ? -1.0f : 1.0f, 0.0f, 0.0f);
        } else if (fabsf(t - tmax_y) < 1e-5f) {
            normal = make_float3(0.0f, direction.y > 0 ? -1.0f : 1.0f, 0.0f);
        } else {
            normal = make_float3(0.0f, 0.0f, direction.z > 0 ? -1.0f : 1.0f);
        }
        return true;
    }
};


__device__ void lm_step(
    const LayeredMedium& medium, 
    float3& position,
    float3& direction,
    double& weight,
    int& trans_type,
    curandState* state
);


} // namespace phonder