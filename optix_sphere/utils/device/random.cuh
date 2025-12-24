#pragma once
#include <curand_kernel.h>

/* 1. CUDA default random number generator */
namespace rng {

    __device__ __inline__ 
    void init_rng_state(curandState* state, unsigned int seed) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(seed, idx, 0, state);
    }

    __device__ __inline__ 
    float rand_uniform01(curandState* state) {
        return curand_uniform(state);
    }

    __device__ __forceinline__ float random_float(curandState* state) {
        // curand_uniform 生成 (0,1) 范围的均匀分布随机数
        return curand_uniform(state);
    }
    
    __device__ __forceinline__ float random_float(float min, float max, curandState* state) {
        // 生成 [min, max] 范围的均匀分布随机数
        return min + (max - min) * curand_uniform(state);
    }    


    __device__ __inline__ 
    float rand_log(curandState* state) {
        float rand = rand_uniform01(state);
        return logf(rand);        
    }

}; // end namespace rng


/* 2. XOROSHIRO128+ random number generator */

#include <cstdint> // c++11 for coss-platform compatibility

namespace xoroshiro128p {

    typedef uint64_t RandState;

    __device__ __inline__
    void xoroshiro128p_seed(RandState state[2], unsigned int seed[4]) {
        state[0] = (uint64_t)seed[0] << 32 | seed[1];
        state[1] = (uint64_t)seed[2] << 32 | seed[3];
    }


    __device__ __inline__
    uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }


    __device__ __inline__
    float xoroshiro128p_nextf(RandState state[2]) {
        union {
            uint64_t i;
            float f[2];
            unsigned int  u[2];
        } result;
        const uint64_t s0 = state[0];
        uint64_t s1 = state[1];
        result.i = s0 + s1;

        s1 ^= s0;
        state[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
        state[1] = rotl(s1, 36); // c
        result.u[0] = 0x3F800000U | (result.u[0] >> 9);

        return result.f[0] - 1.0f;
    }


    // Interface functions
    __device__ __inline__
    void init_rng_state(RandState state[2], unsigned int seed) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        seed = seed ^ idx;
        unsigned int n_seed[4] = {seed, seed, seed, seed};
        xoroshiro128p_seed(state, n_seed);
    }

    __device__ __inline__
    float rand_uniform01(RandState state[2]) {
        return xoroshiro128p_nextf(state);
    }


}; // end namespace xoroshiro128p