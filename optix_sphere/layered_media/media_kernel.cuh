#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "layered_medium.cuh"
#include "photon/batch.cuh"
#include "utils/device/optics.cuh"
#include "utils/device/math.cuh"

namespace phonder {

__device__ void lm_step(
    const LayeredMedium& medium,
    float3& position,
    float3& direction,
    double& weight,
    int& trans_type,
    curandState* state
);



/**
 * @brief Parameters for media simulation kernel
 */
struct MediaKernelParams {
    const LayeredMedium* medium;
    PhotonBatchView      input_batch;
    // --- Direct writable pointers for output ---
    float3*              reflected_positions;
    float3*              reflected_directions;
    double*              reflected_weights;
    float3*              transmitted_positions;
    float3*              transmitted_directions;
    double*              transmitted_weights;
    double*              specular_reflection_weight;
    // -----------------------------------------
    int*                 reflected_counter;
    int*                 transmitted_counter;
    int                  input_batch_size;
    int                  output_buffer_capacity;
    unsigned long long   seed;
    // --- Filter parameters ---
    float                reflected_radius;    // Maximum radius for reflected photons (-1 for no filter)
    float                transmitted_radius;  // Maximum radius for transmitted photons (-1 for no filter)
};


/**
 * @brief Record reflected photon with radius filtering
 */
__device__ void record_reflected_photon(
    const MediaKernelParams* params,
    const float3& pos,
    const float3& dir,
    double& weight
) {
    float radius_sq = distance_sq(pos, make_float3(0.0f, 0.0f, 0.0f));
    if (params->reflected_radius < 0 || radius_sq < params->reflected_radius * params->reflected_radius) {
        int index = atomicAdd(params->reflected_counter, 1);
        if (index < params->output_buffer_capacity) {
            params->reflected_positions[index] = pos;
            params->reflected_directions[index] = dir;
            params->reflected_weights[index] = weight;
        }
    }
}

/**
 * @brief Record transmitted photon with radius filtering
 */
__device__ void record_transmitted_photon(
    const MediaKernelParams* params,
    const float3& pos,
    const float3& dir,
    double& weight
) {
    const float bottom_z = params->medium->layer_boundaries[params->medium->num_layers - 1][1];
    const float3 boundary_pos = make_float3(0.0f, 0.0f, bottom_z);
    const float radius_sq = distance_sq(pos, boundary_pos);
    if (params->transmitted_radius < 0 || radius_sq < params->transmitted_radius * params->transmitted_radius) {
        int index = atomicAdd(params->transmitted_counter, 1);
        if (index < params->output_buffer_capacity) {
            params->transmitted_positions[index] = pos;
            params->transmitted_directions[index] = dir;
            params->transmitted_weights[index] = weight;
        }
    }
}

/**
 * @brief Main kernel for media simulation
 * Processes photon transport through layered media
 */
static __global__ void media_simulation_kernel(const MediaKernelParams* params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    curandState state;
    curand_init(params->seed + idx * 97, idx, 0, &state);

    for (int i = idx; i < params->input_batch_size; i += stride) {
        float3 pos = params->input_batch.positions[i];
        float3 dir = params->input_batch.directions[i];
        double weight = params->input_batch.weights[i];

        if (dir.z <= 0.0f) {
            continue;
        };

        // 确保光子从表面入射（z=0）
        const float3 surface_pos = (pos.z < 0.0f) ? pos + (-pos.z / dir.z) * dir : pos;

        const float n_ambient = params->medium->ambient_n;
        const float n_tissue = params->medium->layers[0].n;
        const float R = optics::fresnel_reflectance(n_ambient, n_tissue, dir.z);

        if (curand_uniform(&state) > R) {
            optics::refract_z_axis(dir, n_ambient, n_tissue);
            pos = surface_pos + dir * 1e-6f;
        } else {
            dir.z = -dir.z;
            atomicAdd(params->specular_reflection_weight, weight);
            record_reflected_photon(params, surface_pos, dir, weight);
            continue;
        }

        int trans_type = 0;
        while (weight > 0.0f) {

            // Russian Roulette
            if (weight < 0.1f) {
                constexpr float survival_probability = 0.1f;
                if (curand_uniform(&state) > survival_probability) {
                    weight = 0.0f;
                    break;
                }
                weight *= 10.0f;
            }


            lm_step(*(params->medium), pos, dir, weight, trans_type, &state);

            // Record photon
            switch (trans_type) {
                case 1: // side
                    weight = 0;
                    break;
                case 2: // transmitted
                    record_transmitted_photon(params, pos, dir, weight);
                    weight = 0.0;
                    break;
                case 3: // reflected
                    record_reflected_photon(params, pos, dir, weight);
                    weight = 0.0;
                    break;

                default:
                    break;
            };

            
        }
    }
}


__device__ void lm_step(
    const LayeredMedium& medium,
    float3& position,
    float3& direction,
    double& weight,
    int& trans_type,
    curandState* state
) {
    // 获取当前层索引（0 = 第一层组织）
    int layer_idx = medium.get_layer_index(position.z);
    if (layer_idx < 0) {
        // 不在任何层内，终止光子
        weight = 0;
        trans_type = 0;
        return;
    }

    const Layer& current_layer = medium.layers[layer_idx];

    // 采样自由程（使用预计算的 inv_mus）
    float step = -logf(curand_uniform(state)) * current_layer.inv_mus;

    // 计算到边界的距离
    float t;
    float3 normal;
    if (!medium.intersect(position, direction, t, normal)) {
        weight = 0;
        trans_type = 0;
        return;
    }

    if (step >= t) {
        step = t;
        position = position + step * direction;

        float absorption_factor = __expf(-current_layer.mua * step);
        weight *= absorption_factor;

        float n1 = current_layer.n;  // 当前层
        float n2;

        bool is_upper_boundary = (normal.z == 1.0f);   // z=0，组织顶部
        bool is_lower_boundary = (normal.z == -1.0f);  // 组织底部或层间下边界

        if (is_upper_boundary) {
            if (layer_idx == 0) {
                n2 = medium.ambient_n;
            } else {
                n2 = medium.layers[layer_idx - 1].n;
            }
        } else if (is_lower_boundary) {
            if (layer_idx == medium.num_layers - 1) {
                n2 = medium.ambient_n;
            } else {
                n2 = medium.layers[layer_idx + 1].n;
            }
        } else {
            n2 = medium.ambient_n;
        }

        // 计算菲涅尔反射系数
        float cos_i = fabsf(dot(direction, normal));
        float r_fresnel = optics::fresnel_reflectance(n1, n2, cos_i);

        if (curand_uniform(state) < r_fresnel) {
            optics::reflect(direction, normal);
            position = position + 1e-4f * normal;
        } else {
            optics::refract_z_axis(direction, n1, n2);
            position = position - 1e-4f * normal;

            if (normal.z == 0.0f) {
                // 从侧边离开
                trans_type = 1;  // side
                return;
            }

            if ((layer_idx == 0 && is_upper_boundary) ||
                (layer_idx == medium.num_layers - 1 && is_lower_boundary)) {
                // 从顶部或底部离开介质
                trans_type = is_upper_boundary ? 3 : 2;  // 3=reflected, 2=transmitted
                return;
            }
        }
    } else {
        // 散射事件
        position = position + step * direction;

        float absorption_factor = __expf(-current_layer.mua * step);
        weight *= absorption_factor;

        optics::scatter_direction(direction, current_layer.g, state);
    }
}



} // namespace phonder