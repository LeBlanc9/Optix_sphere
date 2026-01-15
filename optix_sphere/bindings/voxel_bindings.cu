#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "voxel_media/voxel_grid_builder.h"
#include "voxel_media/voxel_sim_config.h"
#include "voxel_media/voxel_simulator.h"
#include "photon/photon_batch.h"

namespace py = pybind11;
using namespace phonder;
using namespace phonder::voxel;

void bind_voxel(py::module_ &m) {
    py::module_ voxel_module = m.def_submodule("voxel", "Voxel-based media simulation functions");

    // GridBuilder - for constructing material ID grids
    py::class_<GridBuilder>(voxel_module, "GridBuilder")
        .def(py::init<int, int, int>(),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             "Create a grid builder with specified dimensions")
        .def(py::init([](py::array_t<uint8_t> grid_array) {
            // Create GridBuilder from numpy array
            py::buffer_info buf = grid_array.request();

            if (buf.ndim != 3) {
                throw std::runtime_error("Grid must be 3D array");
            }

            int nx = buf.shape[0];
            int ny = buf.shape[1];
            int nz = buf.shape[2];

            auto builder = std::make_unique<GridBuilder>(nx, ny, nz);

            // Copy data from numpy array
            auto grid_ptr = static_cast<uint8_t*>(buf.ptr);
            for (int x = 0; x < nx; x++) {
                for (int y = 0; y < ny; y++) {
                    for (int z = 0; z < nz; z++) {
                        int idx = x * (ny * nz) + y * nz + z;
                        builder->set_voxel(x, y, z, grid_ptr[idx]);
                    }
                }
            }

            return builder;
        }), py::arg("grid"),
            "Create GridBuilder from numpy array (nx, ny, nz) of uint8 material IDs")
        .def("set_voxel", &GridBuilder::set_voxel,
             py::arg("x"), py::arg("y"), py::arg("z"), py::arg("material_id"),
             "Set material ID for a single voxel")
        .def("fill_uniform", &GridBuilder::fill_uniform,
             py::arg("material_id"),
             "Fill all voxels with a material ID")
        .def("fill_region", &GridBuilder::fill_region,
             py::arg("x0"), py::arg("x1"),
             py::arg("y0"), py::arg("y1"),
             py::arg("z0"), py::arg("z1"),
             py::arg("material_id"),
             "Fill a rectangular region with a material ID")
        .def("fill_sphere", &GridBuilder::fill_sphere,
             py::arg("cx"), py::arg("cy"), py::arg("cz"),
             py::arg("radius"), py::arg("material_id"),
             "Fill a sphere with a material ID")
        .def("fill_cylinder_z", &GridBuilder::fill_cylinder_z,
             py::arg("cx"), py::arg("cy"), py::arg("radius"),
             py::arg("z0"), py::arg("z1"), py::arg("material_id"),
             "Fill a cylinder along Z axis")
        .def("get_grid", [](const GridBuilder& builder) {
            // Return as numpy array (read-only view)
            int nx = builder.get_nx();
            int ny = builder.get_ny();
            int nz = builder.get_nz();
            auto grid_ptr = builder.get_grid();

            return py::array_t<uint8_t>(
                {nx, ny, nz},  // shape
                {ny * nz, nz, 1},  // strides (C-order)
                grid_ptr,  // data pointer
                py::cast(builder)  // parent object (keep alive)
            );
        }, "Get grid as numpy array (nx, ny, nz)")
        .def_property_readonly("nx", &GridBuilder::get_nx)
        .def_property_readonly("ny", &GridBuilder::get_ny)
        .def_property_readonly("nz", &GridBuilder::get_nz);

    // SimConfig - configuration for voxel simulation
    py::class_<SimConfig>(voxel_module, "SimConfig")
        .def(py::init<>(), "Create an empty simulation configuration")
        .def("set_grid", [](SimConfig& config,
                           py::array_t<uint8_t> grid,
                           const float3& voxel_size) {
            // Accept numpy array for grid
            py::buffer_info buf = grid.request();

            if (buf.ndim != 3) {
                throw std::runtime_error("Grid must be 3D array");
            }

            int nx = buf.shape[0];
            int ny = buf.shape[1];
            int nz = buf.shape[2];

            auto grid_ptr = static_cast<unsigned char*>(buf.ptr);
            config.set_grid(grid_ptr, nx, ny, nz, voxel_size);
        }, py::arg("grid"), py::arg("voxel_size") = make_float3(1.0f, 1.0f, 1.0f),
           "Set grid from numpy array (nx, ny, nz) with voxel size (default: 1x1x1 mm)")
        .def("set_grid_from_builder", [](SimConfig& config,
                                         const GridBuilder& builder,
                                         const float3& voxel_size) {
            config.set_grid(builder.get_grid(),
                          builder.get_nx(),
                          builder.get_ny(),
                          builder.get_nz(),
                          voxel_size);
        }, py::arg("builder"), py::arg("voxel_size") = make_float3(1.0f, 1.0f, 1.0f),
           "Set grid from GridBuilder with voxel size (default: 1x1x1 mm)")
        .def("set_materials", [](SimConfig& config, py::array_t<float> materials) {
            // Accept numpy array for materials (num_materials, 4)
            py::buffer_info buf = materials.request();

            if (buf.ndim != 2 || buf.shape[1] != 4) {
                throw std::runtime_error("Materials must be (num_materials, 4) array");
            }

            int num_materials = buf.shape[0];
            auto materials_ptr = static_cast<float*>(buf.ptr);
            config.set_materials(materials_ptr, num_materials);
        }, py::arg("materials"),
           "Set materials from numpy array (num_materials, 4) [n, mua, mus, g]")
        .def("set_source", &SimConfig::set_source, py::arg("source"),
             "Set photon source")
        .def("set_ambient_n", &SimConfig::set_ambient_n, py::arg("n"),
             "Set ambient refractive index")
        .def("set_exit_boundaries", &SimConfig::set_exit_boundaries,
             py::arg("z_min"), py::arg("z_max"),
             "Set exit boundaries for reflection/transmission")
        .def("set_gpu_id", &SimConfig::set_gpu_id, py::arg("gpu_id"),
             "Set GPU device ID")
        .def("set_seed", &SimConfig::set_seed, py::arg("seed"),
             "Set random seed")
        .def("is_valid", &SimConfig::is_valid,
             "Check if configuration is valid");

    // SimulationResult
    py::class_<SimulationResult>(voxel_module, "SimulationResult")
        .def(py::init<>())
        .def_readonly("specular_batch", &SimulationResult::specular_batch,
                     "Specular reflection batch at entry")
        .def_readonly("reflected_batch", &SimulationResult::reflected_batch,
                     "Diffuse reflection batch from -Z face")
        .def_readonly("transmitted_batch", &SimulationResult::transmitted_batch,
                     "Transmission batch from +Z face")
        .def("to_host", &SimulationResult::to_host,
             "Convert to host result (HostMediaSimulationResult)");

    // Simulator
    py::class_<Simulator>(voxel_module, "Simulator")
        .def(py::init<const SimConfig&>(), py::arg("config"),
             "Create simulator from configuration")
        .def("run", static_cast<SimulationResult (Simulator::*)(int)>(&Simulator::run),
             py::arg("num_photons"),
             "Run simulation with num_photons")
        .def("run", static_cast<SimulationResult (Simulator::*)(const PhotonBatch&)>(&Simulator::run),
             py::arg("input_batch"),
             "Run simulation with input photon batch")
        .def("update_materials", [](Simulator& sim, py::array_t<float> materials) {
            py::buffer_info buf = materials.request();

            if (buf.ndim != 2 || buf.shape[1] != 4) {
                throw std::runtime_error("Materials must be (num_materials, 4) array");
            }

            int num_materials = buf.shape[0];
            auto materials_ptr = static_cast<float*>(buf.ptr);
            sim.update_materials(materials_ptr, num_materials);
        }, py::arg("materials"),
           "Update material properties")
        .def("update_source", &Simulator::update_source, py::arg("source"),
             "Update light source");
}
