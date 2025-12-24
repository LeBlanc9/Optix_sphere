#pragma once
#include <stdio.h>

namespace phonder {

// ============================================================================
// CUDA Error Handling
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(err); \
        } \
    } while (0)


// ============================================================================
// Monte Carlo Simulation Constants
// ============================================================================

// Russian Roulette parameters
constexpr double WEIGHT_THRESHOLD = 0.1;
constexpr float SURVIVAL_CHANCE = 0.1f;
constexpr float INV_SURVIVAL_CHANCE = 1.0f / SURVIVAL_CHANCE;



}; // namespace phonder