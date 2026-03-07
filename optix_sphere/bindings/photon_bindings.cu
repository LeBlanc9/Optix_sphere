#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "photon/sources.h"
#include "photon/photon_batch.h"
#include "photon/photon_transform.cuh"

namespace py = pybind11;
using namespace phonder;

void bind_photon(py::module_ &m) {
    // PhotonBatch
    py::class_<PhotonBatch>(m, "PhotonBatch")
        .def(py::init<>(), "Create empty PhotonBatch")
        .def(py::init<int>(), py::arg("size"), "Create PhotonBatch with size")
        .def("size", &PhotonBatch::size, "Get number of photons")
        .def("empty", &PhotonBatch::empty, "Check if batch is empty")
        .def("resize", &PhotonBatch::resize, py::arg("new_size"), "Resize batch")
        .def("clear", &PhotonBatch::clear, "Clear batch")
        .def("total_weight", &PhotonBatch::total_weight, "Get total weight of all photons")
        .def("append", &PhotonBatch::append, py::arg("other"),
             "Append another batch to this one (GPU memory copy)")
        .def("swap", &PhotonBatch::swap, py::arg("other"),
             "Swap data with another batch (zero-copy)")
        .def_static("merge", &PhotonBatch::merge, py::arg("batches"),
             "Merge multiple batches into a new batch")
        .def("to_host", &PhotonBatch::to_host, "Copy batch to host memory");

    // HostPhotonBatch
    py::class_<HostPhotonBatch>(m, "HostPhotonBatch")
        .def(py::init<>(), "Create empty HostPhotonBatch")
        .def_readonly("positions", &HostPhotonBatch::positions, "Photon positions")
        .def_readonly("directions", &HostPhotonBatch::directions, "Photon directions")
        .def_readonly("weights", &HostPhotonBatch::weights, "Photon weights");

    // Abstract base class
    py::class_<PhotonSource, std::shared_ptr<PhotonSource>>(m, "PhotonSource")
        .def("generate", &PhotonSource::generate,
             py::arg("batch"), py::arg("num_photons"), py::arg("seed") = 42,
             "Generate photons into a PhotonBatch")
        .def_readwrite("weight", &PhotonSource::weight,
             "Source weight/power");

    // IsotropicPointSource
    py::class_<IsotropicPointSource, PhotonSource, std::shared_ptr<IsotropicPointSource>>(m, "IsotropicPointSource")
        .def(py::init<>())
        .def_readwrite("position", &IsotropicPointSource::position,
                       "Source position (float3)")
        .def_readwrite("weight", &IsotropicPointSource::weight,
                       "Photon weight");

    // CollimatedBeamSource
    py::class_<CollimatedBeamSource, PhotonSource, std::shared_ptr<CollimatedBeamSource>>(m, "CollimatedBeamSource")
        .def(py::init<>())
        .def_readwrite("position", &CollimatedBeamSource::position,
                       "Source position (float3)")
        .def_readwrite("direction", &CollimatedBeamSource::direction,
                       "Beam direction (float3)")
        .def_readwrite("weight", &CollimatedBeamSource::weight,
                       "Photon weight");

    // SpotSource
    py::class_<SpotSource, PhotonSource, std::shared_ptr<SpotSource>>(m, "SpotSource")
        .def(py::init<>())
        .def_readwrite("center_position", &SpotSource::center_position,
                       "Disk center position (float3)")
        .def_readwrite("disk_normal", &SpotSource::disk_normal,
                       "Disk normal vector (float3)")
        .def_readwrite("direction", &SpotSource::direction,
                       "Beam direction (float3)")
        .def_readwrite("radius", &SpotSource::radius,
                       "Disk radius (mm)")
        .def_readwrite("weight", &SpotSource::weight,
                       "Photon weight");

    // GaussianBeamSource
    py::class_<GaussianBeamSource, PhotonSource, std::shared_ptr<GaussianBeamSource>>(m, "GaussianBeamSource")
        .def(py::init<>())
        .def_readwrite("center_position", &GaussianBeamSource::center_position,
                       "Beam center position (float3)")
        .def_readwrite("direction", &GaussianBeamSource::direction,
                       "Beam direction (float3)")
        .def_readwrite("beam_waist", &GaussianBeamSource::beam_waist,
                       "Beam waist radius (mm)")
        .def_readwrite("weight", &GaussianBeamSource::weight,
                       "Photon weight");

    // FocusedSpotSource
    py::class_<FocusedSpotSource, PhotonSource, std::shared_ptr<FocusedSpotSource>>(m, "FocusedSpotSource")
        .def(py::init<>())
        .def_readwrite("spot_center", &FocusedSpotSource::spot_center,
                       "Disk center position (float3)")
        .def_readwrite("spot_radius", &FocusedSpotSource::spot_radius,
                       "Disk radius (mm)")
        .def_readwrite("disk_normal", &FocusedSpotSource::disk_normal,
                       "Disk normal direction (float3)")
        .def_readwrite("convergence_half_angle_rad", &FocusedSpotSource::convergence_half_angle_rad,
                       "Cone half angle for direction distribution (radians)")
        .def_readwrite("main_axis", &FocusedSpotSource::main_axis,
                       "Main direction axis for cone distribution (float3)")
        .def_readwrite("weight", &FocusedSpotSource::weight,
                       "Photon weight");

    // LambertianDiskSource
    py::class_<LambertianDiskSource, PhotonSource, std::shared_ptr<LambertianDiskSource>>(m, "LambertianDiskSource")
        .def(py::init<>())
        .def_readwrite("center_position", &LambertianDiskSource::center_position,
                       "Disk center position (float3)")
        .def_readwrite("disk_normal", &LambertianDiskSource::disk_normal,
                       "Disk normal / hemisphere orientation (float3)")
        .def_readwrite("radius", &LambertianDiskSource::radius,
                       "Disk radius (mm)")
        .def_readwrite("weight", &LambertianDiskSource::weight,
                       "Photon weight");

    // Convenience function for generating photons
    m.def("generate_photons",
          [](PhotonSource& source, int num_photons, unsigned long long seed = 42) {
              PhotonBatch batch;
              source.generate(batch, num_photons, seed);
              return batch;
          },
          py::arg("source"), py::arg("num_photons"), py::arg("seed") = 42,
          "Generate photons from any source type\n\n"
          "Args:\n"
          "    source: Any PhotonSource object\n"
          "    num_photons (int): Number of photons to generate\n"
          "    seed (int, optional): Random seed. Defaults to 42.\n\n"
          "Returns:\n"
          "    PhotonBatch: Generated photons on GPU\n\n"
          "Example:\n"
          "    >>> source = osg.CollimatedBeamSource()\n"
          "    >>> source.position = (0, 0, -1)\n"
          "    >>> source.direction = (0, 0, 1)\n"
          "    >>> batch = osg.generate_photons(source, 10000)\n");

    m.def("translate_photons", &translate_photons,
          py::arg("input_batch"), py::arg("offset"));

    m.def("translate_photons_inplace", &translate_photons_inplace,
          py::arg("batch"), py::arg("offset"));

}

