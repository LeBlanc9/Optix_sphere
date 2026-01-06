#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "simulator.h"
#include "theory/theory.h"
#include "material.h"
#include "photon/sources.h"
#include "photon/photon_batch.h"
#include "photon/photon_transform.cuh"
#include <spdlog/spdlog.h>

namespace py = pybind11;
using namespace phonder;
using namespace theory;

// Implementation for the missing set_log_level function
void set_log_level(int level) {
    spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "OptiX Sphere - Monte Carlo simulation for integrating spheres";
    m.attr("__version__") = "0.1.0";

    // Bind spdlog's level enum to Python
    py::enum_<spdlog::level::level_enum>(m, "LogLevel")
        .value("TRACE", spdlog::level::trace)
        .value("DEBUG", spdlog::level::debug)
        .value("INFO", spdlog::level::info)
        .value("WARN", spdlog::level::warn)
        .value("ERROR", spdlog::level::err)
        .value("CRITICAL", spdlog::level::critical)
        .value("OFF", spdlog::level::off)
        .export_values();


    py::class_<float3>(m, "float3")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &float3::x)
        .def_readwrite("y", &float3::y)
        .def_readwrite("z", &float3::z);

    py::class_<MeshSceneConfig>(m, "MeshSceneConfig")
        .def(py::init<>());

    py::class_<SimConfig>(m, "SimConfig")
        .def(py::init<>())
        .def_readwrite("num_rays", &SimConfig::num_rays)
        .def_readwrite("max_bounces", &SimConfig::max_bounces)
        .def_readwrite("use_nee", &SimConfig::use_nee)
        .def_readwrite("random_seed", &SimConfig::random_seed);

    py::class_<SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readonly("detected_flux", &SimulationResult::detected_flux)
        .def_readonly("irradiance", &SimulationResult::irradiance)
        .def_readonly("total_rays", &SimulationResult::total_rays)
        .def_readonly("detected_rays", &SimulationResult::detected_rays)
        .def_readonly("avg_bounces", &SimulationResult::avg_bounces);

    py::class_<TheoryResult>(m, "TheoryResult")
        .def(py::init<>())
        .def_readonly("avg_irradiance", &TheoryResult::avg_irradiance)
        .def_readonly("sphere_area", &TheoryResult::sphere_area)
        .def_readonly("total_flux_in_sphere", &TheoryResult::total_flux_in_sphere);

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<>(), "Initializes the OptiX Simulator.")
        .def("build_scene_from_file",
             static_cast<void (Simulator::*)(const std::string&, const MeshSceneConfig&)>(&Simulator::build_scene_from_file),
             py::arg("file_path"), py::arg("config"))
        .def("build_scene_from_file",
             static_cast<void (Simulator::*)(const std::string&, const std::map<std::string, MaterialFactory>&, const MeshSceneConfig&)>(&Simulator::build_scene_from_file),
             py::arg("file_path"), py::arg("materials"), py::arg("config"))
        .def("run", static_cast<SimulationResult (Simulator::*)(const phonder::PhotonSource&, const SimConfig&)>(&Simulator::run),
             py::arg("photon_source"), py::arg("config"))
        .def("run", static_cast<SimulationResult (Simulator::*)(const phonder::PhotonBatch&, const SimConfig&)>(&Simulator::run),
             py::arg("source_batch"), py::arg("config"))
        .def("get_detector_total_area", &Simulator::get_detector_total_area);

    py::class_<IsotropicPointSource>(m, "IsotropicPointSource")
        .def(py::init<>())
        .def_readwrite("position", &IsotropicPointSource::position)
        .def_readwrite("weight", &IsotropicPointSource::weight);

    py::class_<CollimatedBeamSource>(m, "CollimatedBeamSource")
        .def(py::init<>())
        .def_readwrite("position", &CollimatedBeamSource::position)
        .def_readwrite("direction", &CollimatedBeamSource::direction)
        .def_readwrite("weight", &CollimatedBeamSource::weight);

    // Bind HostPhotonBatch to expose data to Python as numpy arrays
    py::class_<HostPhotonBatch>(m, "HostPhotonBatch")
        .def(py::init<>())
        .def("size", &HostPhotonBatch::size)
        .def_property_readonly("positions", [](const HostPhotonBatch &b) {
            std::vector<py::ssize_t> shape = { (py::ssize_t)b.size(), 3 };
            return py::array_t<float>(shape, reinterpret_cast<const float*>(b.positions.data()));
        })
        .def_property_readonly("directions", [](const HostPhotonBatch &b) {
            std::vector<py::ssize_t> shape = { (py::ssize_t)b.size(), 3 };
            return py::array_t<float>(shape, reinterpret_cast<const float*>(b.directions.data()));
        })
        .def_property_readonly("weights", [](const HostPhotonBatch &b) {
            return py::array_t<double>(b.weights.size(), b.weights.data());
        });

    // Bind the new device-centric PhotonBatch
    py::class_<PhotonBatch>(m, "PhotonBatch")
        .def(py::init<>())
        .def("size", &PhotonBatch::size)
        .def("empty", &PhotonBatch::empty)
        .def("clear", &PhotonBatch::clear)
        .def("to_host", &PhotonBatch::to_host,
             "Transfer photon batch from GPU to CPU memory, returning a HostPhotonBatch.");

    m.def("translate_photons", &translate_photons,
          py::arg("input_batch"), py::arg("offset"),
          "Translate photon positions by a fixed offset.");

    m.def("set_log_level", &set_log_level,
          py::arg("level"),
          "Set global logging level (0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical, 6=off).");

    py::module_ material_module = m.def_submodule("material", "Material factory functions");
    material_module.def("lambertian", &material::lambertian, py::arg("reflectance"));
    material_module.def("mixed", &material::mixed, py::arg("diffuse_ratio"), py::arg("specular_ratio"), py::arg("reflectance"));
    material_module.def("detector", &material::detector);
    material_module.def("absorber", &material::absorber);
    material_module.def("spherical_lambertian", &material::spherical_lambertian, py::arg("reflectance"));
    material_module.def("spherical_mixed", &material::spherical_mixed, py::arg("diffuse_ratio"), py::arg("specular_ratio"), py::arg("reflectance"));
}