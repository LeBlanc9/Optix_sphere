#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../voxel_grid.cuh"

namespace phonder::voxel {

/**
 * @brief Calculate Fresnel reflection coefficient
 *
 * Computes R = (Rs + Rp)/2 for unpolarized light at a refractive index boundary.
 *
 * @param v Direction vector
 * @param n1 Refractive index of current medium
 * @param n2 Refractive index of next medium
 * @param flipdir Interface normal (0=x, 1=y, 2=z)
 * @return Reflection coefficient [0,1]
 */
__device__ inline float reflectcoeff(float3* v, float n1, float n2, int flipdir) {
    float Icos = fabsf((flipdir == 0) ? v->x : (flipdir == 1 ? v->y : v->z));
    float tmp0 = n1 * n1;
    float tmp1 = n2 * n2;
    float tmp2 = 1.f - tmp0 / tmp1 * (1.f - Icos * Icos); /** 1-[n1/n2*sin(si)]^2 = cos(ti)^2*/

    if (tmp2 > 0.f) { //< partial reflection
        float Re, Im, Rtotal;
        Re = tmp0 * Icos * Icos + tmp1 * tmp2;
        tmp2 = sqrtf(tmp2); /** to save one sqrt*/
        Im = 2.f * n1 * n2 * Icos * tmp2;
        Rtotal = (Re - Im) / (Re + Im); /** Rp*/
        Re = tmp1 * Icos * Icos + tmp0 * tmp2 * tmp2;
        Rtotal = (Rtotal + (Re - Im) / (Re + Im)) * 0.5f; /** (Rp+Rs)/2*/
        return Rtotal;
    } else { //< total internal reflection
        return 1.f;
    }
}

/**
 * @brief Apply refraction at interface
 *
 * Updates direction vector using Snell's law for voxelized boundaries.
 *
 * @param v Direction vector (modified in place)
 * @param n1 Refractive index of current medium
 * @param n2 Refractive index of next medium
 * @param flipdir Interface normal (0=x, 1=y, 2=z)
 */
__device__ inline void transmit(float3* v, float n1, float n2, int flipdir) {
    float tmp0 = n1 / n2;
    v->x *= tmp0;
    v->y *= tmp0;
    v->z *= tmp0;

    (flipdir == 0) ?
    (v->x = ((tmp0 = v->y * v->y + v->z * v->z) < 1.f) ? sqrtf(1.f - tmp0) * ((v->x > 0.f) - (v->x < 0.f)) : 0.f) :
    ((flipdir == 1) ?
     (v->y = ((tmp0 = v->x * v->x + v->z * v->z) < 1.f) ? sqrtf(1.f - tmp0) * ((v->y > 0.f) - (v->y < 0.f)) : 0.f) :
     (v->z = ((tmp0 = v->x * v->x + v->y * v->y) < 1.f) ? sqrtf(1.f - tmp0) * ((v->z > 0.f) - (v->z < 0.f)) : 0.f));

    tmp0 = rsqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
    v->x *= tmp0;
    v->y *= tmp0;
    v->z *= tmp0;
}

/**
 * @brief Rotate direction vector by spherical angles
 *
 * @param v Direction vector (modified in place)
 * @param stheta sin(theta) - polar angle sine
 * @param ctheta cos(theta) - polar angle cosine
 * @param sphi sin(phi) - azimuthal angle sine
 * @param cphi cos(phi) - azimuthal angle cosine
 */
__device__ inline void rotatevector(float3* v, float stheta, float ctheta, float sphi, float cphi) {
    if (v->z > -1.f + EPS && v->z < 1.f - EPS) {
        float tmp0 = 1.f - v->z * v->z;
        float tmp1 = stheta * rsqrtf(tmp0);
        float new_x = tmp1 * (v->x * v->z * cphi - v->y * sphi) + v->x * ctheta;
        float new_y = tmp1 * (v->y * v->z * cphi + v->x * sphi) + v->y * ctheta;
        float new_z = -tmp1 * tmp0 * cphi + v->z * ctheta;

        v->x = new_x;
        v->y = new_y;
        v->z = new_z;
    } else {
        v->x = stheta * cphi;
        v->y = stheta * sphi;
        v->z = (v->z > 0.f) ? ctheta : -ctheta;
    }

    float tmp0 = rsqrtf(v->x * v->x + v->y * v->y + v->z * v->z);
    v->x *= tmp0;
    v->y *= tmp0;
    v->z *= tmp0;
}

/**
 * @brief Sample scattering angle from Henyey-Greenstein phase function
 *
 * @param g Anisotropy factor [-1, 1]
 * @param rand Random number [0, 1)
 * @return cos(theta) of scattering angle
 */
__device__ inline float hg_sample_costheta(float g, float rand) {
    if (fabsf(g) < EPS) {
        // Isotropic scattering
        return 2.f * rand - 1.f;
    } else {
        // Henyey-Greenstein
        float tmp = (1.f - g * g) / (1.f - g + 2.f * g * rand);
        return (1.f + g * g - tmp * tmp) / (2.f * g);
    }
}

/**
 * @brief Apply scattering to direction vector
 *
 * @param v Direction vector (modified in place)
 * @param g Anisotropy factor
 * @param state Random number generator state (template supports any cuRAND type)
 */
template<typename RNGState>
__device__ __forceinline__ void scatter(float3* v, float g, RNGState* state) {
    float costheta = hg_sample_costheta(g, curand_uniform(state));
    float sintheta = sqrtf(1.f - costheta * costheta);

    float phi = TWO_PI * curand_uniform(state);
    float sphi, cphi;
    sincosf(phi, &sphi, &cphi);

    rotatevector(v, sintheta, costheta, sphi, cphi);
}

/**
 * @brief Sample scattering path length
 *
 * @param state Random number generator state (template supports any cuRAND type)
 * @return Path length in mean free paths
 */
template<typename RNGState>
__device__ inline float sample_scatter_length(RNGState* state) {
    return -logf(curand_uniform(state) + EPS);
}

/**
 * @brief Reflect photon at interface
 *
 * Reverses the perpendicular direction component and updates voxel index.
 *
 * @param dir Direction vector (modified in place)
 * @param voxel_state Voxel indices [0-2] and hit face [3]
 * @param inv_dir Inverse direction (updated)
 */
__device__ __forceinline__ void reflect_at_interface(float3* dir, short voxel_state[4], float3* inv_dir) {
    // Reverse direction component perpendicular to the interface
    // and update voxel index to go back to previous voxel
    if (voxel_state[3] == 0) {
        // Reflected through X face
        dir->x = -dir->x;
        voxel_state[0] += (dir->x > 0.f) ? 1 : -1;
    } else if (voxel_state[3] == 1) {
        // Reflected through Y face
        dir->y = -dir->y;
        voxel_state[1] += (dir->y > 0.f) ? 1 : -1;
    } else if (voxel_state[3] == 2) {
        // Reflected through Z face
        dir->z = -dir->z;
        voxel_state[2] += (dir->z > 0.f) ? 1 : -1;
    }

    // Update inverse direction
    inv_dir->x = __fdividef(1.f, dir->x);
    inv_dir->y = __fdividef(1.f, dir->y);
    inv_dir->z = __fdividef(1.f, dir->z);
}

} // namespace phonder::voxel