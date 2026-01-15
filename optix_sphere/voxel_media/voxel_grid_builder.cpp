#include "voxel_grid_builder.h"
#include <algorithm>
#include <cmath>

namespace phonder::voxel {

GridBuilder::GridBuilder(int nx, int ny, int nz)
    : nx_(nx), ny_(ny), nz_(nz)
{
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::invalid_argument("Grid dimensions must be positive");
    }

    // Allocate grid - initialize all to material 0
    grid_.resize(nx * ny * nz, 0);
}

int GridBuilder::get_index(int x, int y, int z) const {
    if (x < 0 || x >= nx_ || y < 0 || y >= ny_ || z < 0 || z >= nz_) {
        throw std::out_of_range("Voxel index out of bounds");
    }
    return x * (ny_ * nz_) + y * nz_ + z;
}

void GridBuilder::set_voxel(int x, int y, int z, int material_id) {
    if (material_id < 0 || material_id > 255) {
        throw std::out_of_range("Material ID must be 0-255");
    }
    int idx = get_index(x, y, z);
    grid_[idx] = static_cast<unsigned char>(material_id);
}

void GridBuilder::fill_uniform(int material_id) {
    if (material_id < 0 || material_id > 255) {
        throw std::out_of_range("Material ID must be 0-255");
    }
    std::fill(grid_.begin(), grid_.end(), static_cast<unsigned char>(material_id));
}

void GridBuilder::fill_region(int x0, int x1, int y0, int y1, int z0, int z1, int material_id) {
    if (material_id < 0 || material_id > 255) {
        throw std::out_of_range("Material ID must be 0-255");
    }

    // Clamp to valid range
    x0 = std::max(0, std::min(x0, nx_));
    x1 = std::max(0, std::min(x1, nx_));
    y0 = std::max(0, std::min(y0, ny_));
    y1 = std::max(0, std::min(y1, ny_));
    z0 = std::max(0, std::min(z0, nz_));
    z1 = std::max(0, std::min(z1, nz_));

    unsigned char mid = static_cast<unsigned char>(material_id);
    for (int x = x0; x < x1; ++x) {
        for (int y = y0; y < y1; ++y) {
            for (int z = z0; z < z1; ++z) {
                int idx = get_index(x, y, z);
                grid_[idx] = mid;
            }
        }
    }
}

void GridBuilder::fill_sphere(int cx, int cy, int cz, float radius, int material_id) {
    if (material_id < 0 || material_id > 255) {
        throw std::out_of_range("Material ID must be 0-255");
    }

    unsigned char mid = static_cast<unsigned char>(material_id);
    float r2 = radius * radius;

    // Bounding box
    int x0 = std::max(0, static_cast<int>(std::floor(cx - radius)));
    int x1 = std::min(nx_, static_cast<int>(std::ceil(cx + radius)) + 1);
    int y0 = std::max(0, static_cast<int>(std::floor(cy - radius)));
    int y1 = std::min(ny_, static_cast<int>(std::ceil(cy + radius)) + 1);
    int z0 = std::max(0, static_cast<int>(std::floor(cz - radius)));
    int z1 = std::min(nz_, static_cast<int>(std::ceil(cz + radius)) + 1);

    for (int x = x0; x < x1; ++x) {
        for (int y = y0; y < y1; ++y) {
            for (int z = z0; z < z1; ++z) {
                float dx = x - cx;
                float dy = y - cy;
                float dz = z - cz;
                if (dx*dx + dy*dy + dz*dz <= r2) {
                    int idx = get_index(x, y, z);
                    grid_[idx] = mid;
                }
            }
        }
    }
}

void GridBuilder::fill_cylinder_z(int cx, int cy, float radius, int z0, int z1, int material_id) {
    if (material_id < 0 || material_id > 255) {
        throw std::out_of_range("Material ID must be 0-255");
    }

    unsigned char mid = static_cast<unsigned char>(material_id);
    float r2 = radius * radius;

    // Clamp z range
    z0 = std::max(0, z0);
    z1 = std::min(nz_, z1);

    // Bounding box in XY
    int x0 = std::max(0, static_cast<int>(std::floor(cx - radius)));
    int x1 = std::min(nx_, static_cast<int>(std::ceil(cx + radius)) + 1);
    int y0 = std::max(0, static_cast<int>(std::floor(cy - radius)));
    int y1 = std::min(ny_, static_cast<int>(std::ceil(cy + radius)) + 1);

    for (int x = x0; x < x1; ++x) {
        for (int y = y0; y < y1; ++y) {
            float dx = x - cx;
            float dy = y - cy;
            if (dx*dx + dy*dy <= r2) {
                for (int z = z0; z < z1; ++z) {
                    int idx = get_index(x, y, z);
                    grid_[idx] = mid;
                }
            }
        }
    }
}

} // namespace phonder::voxel
