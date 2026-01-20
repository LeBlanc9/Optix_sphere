#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "voxel_media/voxel_sim_config.h"
#include "voxel_media/voxel_simulator.h"
#include "voxel_media/boundary_config.h"
#include "photon/photon_batch.h"

namespace py = pybind11;
using namespace phonder;
using namespace phonder::voxel;

void bind_voxel(py::module_ &m) {
    py::module_ voxel_module = m.def_submodule("voxel", "Voxel-based media simulation functions");

    // BoundaryFace enum
    py::enum_<BoundaryFace>(voxel_module, "BoundaryFace")
        .value("NegX", BoundaryFace::NegX, "-X face")
        .value("NegY", BoundaryFace::NegY, "-Y face")
        .value("NegZ", BoundaryFace::NegZ, "-Z face")
        .value("PosX", BoundaryFace::PosX, "+X face")
        .value("PosY", BoundaryFace::PosY, "+Y face")
        .value("PosZ", BoundaryFace::PosZ, "+Z face")
        .export_values();

    // BoundaryCollectionConfig
    py::class_<BoundaryCollectionConfig>(voxel_module, "BoundaryCollectionConfig")
        .def(py::init<>(), "Create default boundary collection config")
        // Radius configuration
        .def_readwrite("collection_radius_negative", &BoundaryCollectionConfig::collection_radius_negative,
                       "Collection radius for negative faces (mm, negative = no limit)")
        .def_readwrite("collection_radius_positive", &BoundaryCollectionConfig::collection_radius_positive,
                       "Collection radius for positive faces (mm, negative = no limit)")
        .def("set_negative_radius", &BoundaryCollectionConfig::set_negative_radius,
             py::arg("radius"),
             "Set collection radius for negative faces")
        .def("set_positive_radius", &BoundaryCollectionConfig::set_positive_radius,
             py::arg("radius"),
             "Set collection radius for positive faces")
        // Center configuration
        .def_readwrite("collection_center", &BoundaryCollectionConfig::collection_center,
                       "Collection center point (float3)")
        .def_readwrite("use_grid_center", &BoundaryCollectionConfig::use_grid_center,
                       "Auto-set collection center to grid center")
        .def("set_center", &BoundaryCollectionConfig::set_center,
             py::arg("x"), py::arg("y"), py::arg("z"),
             "Set collection center point")
        // Enable/disable individual faces
        .def("enable_face", &BoundaryCollectionConfig::enable_face,
             py::arg("face"),
             "Enable collection on a specific face")
        .def("disable_face", &BoundaryCollectionConfig::disable_face,
             py::arg("face"),
             "Disable collection on a specific face")
        // Enable collection by axis
        .def("enable_x_faces", &BoundaryCollectionConfig::enable_x_faces,
             "Enable collection on -X and +X faces")
        .def("enable_y_faces", &BoundaryCollectionConfig::enable_y_faces,
             "Enable collection on -Y and +Y faces")
        .def("enable_z_faces", &BoundaryCollectionConfig::enable_z_faces,
             "Enable collection on -Z and +Z faces")
        .def("enable_all_faces", &BoundaryCollectionConfig::enable_all_faces,
             "Enable collection on all 6 faces")
        .def("enable_positive_faces", &BoundaryCollectionConfig::enable_positive_faces,
             "Enable collection on +X, +Y, +Z faces")
        .def("enable_negative_faces", &BoundaryCollectionConfig::enable_negative_faces,
             "Enable collection on -X, -Y, -Z faces")
        // Disable collection
        .def("disable_all_faces", &BoundaryCollectionConfig::disable_all_faces,
             "Disable collection on all faces")
        // Query methods
        .def("is_collecting", &BoundaryCollectionConfig::is_collecting,
             py::arg("face"),
             "Check if a specific face collects photons")
        .def("has_positive_collection", &BoundaryCollectionConfig::has_positive_collection,
             "Check if any positive face collects photons")
        .def("has_negative_collection", &BoundaryCollectionConfig::has_negative_collection,
             "Check if any negative face collects photons")
        // String representation
        .def("to_string", &BoundaryCollectionConfig::to_string,
             "Get human-readable description")
        .def("__str__", &BoundaryCollectionConfig::to_string,
             "String representation")
        .def("__repr__", &BoundaryCollectionConfig::to_string,
             "String representation");

    // SimConfig - public members with nested vectors
    py::class_<SimConfig>(voxel_module, "SimConfig")
        .def(py::init<>(), "Create empty SimConfig")
        // 3D grid (pybind11 auto-converts nested list/numpy → nested vector)
        .def_readwrite("grid", &SimConfig::grid,
                       "3D Material ID grid [x][y][z] (nested list or numpy array)")
        .def_readwrite("voxel_size", &SimConfig::voxel_size,
                       "Voxel size in mm (tuple/list/float3)")
        // 2D materials (pybind11 auto-converts nested list → nested vector)
        .def_readwrite("materials", &SimConfig::materials,
                       "Material properties [i][j] where i=material_id, j=[n,mua,mus,g]. "
                       "Note: materials[0] is the ambient medium (typically air)")
        .def_readwrite("source", &SimConfig::source,
                       "Light source")
        .def_readwrite("boundary_collection", &SimConfig::boundary_collection,
                       "Boundary collection configuration")
        .def_readwrite("enable_specular", &SimConfig::enable_specular,
                       "Enable specular reflection calculation")
        .def_readwrite("merge_specular", &SimConfig::merge_specular,
                       "Merge specular into boundary batches (saves memory, specular_batch will be empty)")
        .def_readwrite("gpu_id", &SimConfig::gpu_id,
                       "GPU device ID")
        .def_readwrite("seed", &SimConfig::seed,
                       "Random seed (0 = auto)")
        // Helper methods
        .def("get_dims", &SimConfig::get_dims,
             "Get grid dimensions (int3)")
        .def("get_num_materials", &SimConfig::get_num_materials,
             "Get number of materials")
        .def("get_ambient_n", &SimConfig::get_ambient_n,
             "Get ambient refractive index (from materials[0])")
        .def("get_center_x", &SimConfig::get_center_x,
             "Get grid center X coordinate (mm)")
        .def("get_center_y", &SimConfig::get_center_y,
             "Get grid center Y coordinate (mm)")
        .def("is_valid", &SimConfig::is_valid,
             "Validate configuration");

    // Simulator
    py::class_<Simulator>(voxel_module, "Simulator")
        .def(py::init<const SimConfig&>(),
             py::arg("config"),
             "Create simulator from SimConfig")
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

    // SimulationResult
    py::class_<SimulationResult>(voxel_module, "SimulationResult")
        .def_readonly("specular_batch", &SimulationResult::specular_batch,
                      "Specular reflection photons (mirror-like reflection at entry)")
        .def_readonly("negative_boundary_batch", &SimulationResult::negative_boundary_batch,
                      "Photons exiting from negative faces (-X, -Y, -Z)")
        .def_readonly("positive_boundary_batch", &SimulationResult::positive_boundary_batch,
                      "Photons exiting from positive faces (+X, +Y, +Z)")
        .def("to_host", &SimulationResult::to_host,
             "Copy results to host memory");
}
