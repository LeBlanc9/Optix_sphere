#pragma once

/**
 * @file math_constants.h
 * @brief Unified mathematical constants for the entire project
 *
 * This is the ONLY place where mathematical constants should be defined.
 * All other files should include this header.
 *
 * Design principles:
 * - Use constexpr (modern C++, type-safe, no macro pollution)
 * - Use lowercase names (avoiding conflicts with system macros like M_PI)
 * - Define in phonder namespace
 * - Single source of truth
 */

namespace phonder {

// ============================================================================
// Mathematical Constants
// ============================================================================

/// Pi (π) - ratio of circle's circumference to diameter
constexpr float pi = 3.1415926535897932f;

/// 2π - full circle in radians
constexpr float two_pi = 2.0f * pi;

/// π/2 - quarter circle in radians
constexpr float half_pi = pi / 2.0f;

/// 1/π
constexpr float inv_pi = 1.0f / pi;

/// 1/(2π)
constexpr float inv_two_pi = 1.0f / two_pi;

/// Numerical epsilon for floating point comparisons
constexpr float epsilon = 1e-6f;

/// Speed of light in vacuum (mm/ps)
constexpr float speed_of_light = 0.299792458f;

// ============================================================================
// Double precision variants (if needed)
// ============================================================================

constexpr double pi_d = 3.1415926535897932;
constexpr double two_pi_d = 2.0 * pi_d;
constexpr double epsilon_d = 1e-12;

} // namespace phonder
