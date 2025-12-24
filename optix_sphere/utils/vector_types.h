#pragma once
#include <vector_types.h> // Provides the official float3 from CUDA

// This header is a PURE C++ header.
// It relies on the CUDA toolkit's <vector_types.h> to define vector types
// in a way that is compatible with both host and device compilers.
// It also defines shared mathematical constants.

// ============================================================================
// Mathematical Constants
// ============================================================================
static const float C_PI = 3.1415926535897932f;
static const float C_M_PI = 3.1415926535897932f;
static const float C_TWO_PI = 2.0f * C_PI;
static const float C_EPSILON = 1e-6f;
