#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h> // Add spdlog header for log level control

#include "simulator.h" // New unified Simulator API
#include "theory/theory.h"
#include "constants.h"
#include "photon/sources.h"     // New data-only source structs
#include "photon/batch.h"       // For HostPhotonBatch

namespace py = pybind11;
using namespace phonder; // For PhotonSource, IsotropicPointSource etc.

// C++ function to set spdlog level
void set_log_level(const std::string& level_name) {
    spdlog::level::level_enum level;
    if (level_name == "trace") {
        level = spdlog::level::trace;
    } else if (level_name == "debug") {
        level = spdlog::level::debug;
    } else if (level_name == "info") {
        level = spdlog::level::info;
    } else if (level_name == "warn") {
        level = spdlog::level::warn;
    } else if (level_name == "error") {
        level = spdlog::level::err);
    } else if (level_name == "critical") {
        level = spdlog::level::critical;
    } else if (level_name == "off") {
        level = spdlog::level::off;
    } else {
        spdlog::warn("Invalid log level: '{}'. Defaulting to 'info'.", level_name);
        level = spdlog::level::info;
    }
    spdlog::set_level(level);
    spdlog::info("Global log level set to '{}'.", level_name);
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "OptiX Sphere - Monte Carlo simulation for integrating spheres";
    m.attr("__version__") = "0.1.0";

    // Bind common vector types
    py::class_<float3>(m, "float3")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &float3::x)
        .def_readwrite("y", &float3::y)
        .def_readwrite("z", &float3::z);

    // Bind existing scene types
    py::class_<Sphere>(m, "Sphere")
        .def(py::init<>())
        .def_readwrite("center", &Sphere::center)
        .def_readwrite("radius", &Sphere::radius)
        .def_readwrite("reflectance", &Sphere::reflectance);

    py::class_<Detector>(m, "Detector")
        .def(py::init<>())
        .def_readwrite("position", &Detector::position)
        .def_readwrite("normal", &Detector::normal)
        .def_readwrite("radius", &Detector::radius);

    // Bind new scene configuration structs
    py::class_<MeshSceneConfig>(m, "MeshSceneConfig")
        .def(py::init<>())
        .def_readwrite("default_reflectance", &MeshSceneConfig::default_reflectance);

    // Bind simulation configuration
    py::class_<SimConfig>(m, "SimConfig")
        .def(py::init<>())
        .def_readwrite("num_rays", &SimConfig::num_rays)
        .def_readwrite("max_bounces", &SimConfig::max_bounces)
        .def_readwrite("use_nee", &SimConfig::use_nee)
        .def_readwrite("random_seed", &SimConfig::random_seed);

    // Bind simulation results
    py::class_<SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readonly("detected_flux", &SimulationResult::detected_flux)
        .def_readonly("irradiance", &SimulationResult::irradiance)
        .def_readonly("total_rays", &SimulationResult::total_rays)
        .def_readonly("detected_rays", &SimulationResult::detected_rays)
        .def_readonly("avg_bounces", &SimulationResult::avg_bounces);

    // Bind theory results
    py::class_<TheoryResult>(m, "TheoryResult")
        .def(py::init<>())
        .def_readonly("avg_irradiance", &TheoryResult::avg_irradiance)
        .def_readonly("detected_flux", &TheoryResult::detected_flux)
        .def_readonly("sphere_area", &TheoryResult::sphere_area)
        .def_readonly("total_flux_in_sphere", &TheoryResult::total_flux_in_sphere);

    // Bind the unified Simulator class
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<>(), "Initializes the OptiX Simulator.")
        .def("build_scene_from_file", &Simulator::build_scene_from_file,
             py::arg("file_path"), py::arg("config"),
             "Builds the scene from an OBJ file using the provided mesh configuration. "
             "The 'file_path' should be an absolute path to the .obj file.")
        .def("run", &Simulator::run,
             py::arg("photon_source"), py::arg("config"),
             "Runs the Monte Carlo simulation with the given photon source and simulation configuration.")
        .def("get_detector_total_area", &Simulator::get_detector_total_area,
             "Returns the total area of the detector in the currently built scene (mm^2).");

    // Bind the data-only source structs
    py::class_<IsotropicPointSource>(m, "IsotropicPointSource")
        .def(py::init<>())
        .def_readwrite("position", &IsotropicPointSource::position)
        .def_readwrite("weight", &IsotropicPointSource::weight);

    py::class_<CollimatedBeamSource>(m, "CollimatedBeamSource")
        .def(py::init<>())
        .def_readwrite("position", &CollimatedBeamSource::position)
        .def_readwrite("direction", &CollimatedBeamSource::direction)
        .def_readwrite("weight", &CollimatedBeamSource::weight);

    py::class_<SpotSource>(m, "SpotSource")
        .def(py::init<>())
        .def_readwrite("center_position", &SpotSource::center_position)
        .def_readwrite("direction", &SpotSource::direction)
        .def_readwrite("radius", &SpotSource::radius)
        .def_readwrite("weight", &SpotSource::weight);
    
    py::class_<GaussianBeamSource>(m, "GaussianBeamSource")
        .def(py::init<>())
        .def_readwrite("center_position", &GaussianBeamSource::center_position)
        .def_readwrite("direction", &GaussianBeamSource::direction)
        .def_readwrite("beam_waist", &GaussianBeamSource::beam_waist)
        .def_readwrite("weight", &GaussianBeamSource::weight);

    py::class_<FocusedSpotSource>(m, "FocusedSpotSource")
        .def(py::init<>())
        .def_readwrite("spot_center", &FocusedSpotSource::spot_center)
        .def_readwrite("spot_radius", &FocusedSpotSource::spot_radius)
        .def_readwrite("convergence_half_angle_rad", &FocusedSpotSource::convergence_half_angle_rad)
        .def_readwrite("main_axis", &FocusedSpotSource::main_axis)
        .def_readwrite("source_distance", &FocusedSpotSource::source_distance)
        .def_readwrite("weight", &FocusedSpotSource::weight);

    // HostPhotonBatch
    py::class_<HostPhotonBatch>(m, "HostPhotonBatch")
        .def(py::init<>())
        .def("size", &HostPhotonBatch::size)
        .def_property_readonly("positions", [](const HostPhotonBatch &b) { return py::array_t<float>(b.positions.size() * 3, reinterpret_cast<const float*>(b.positions.data())); })
        .def_property_readonly("directions", [](const HostPhotonBatch &b) { return py::array_t<float>(b.directions.size() * 3, reinterpret_cast<const float*>(b.directions.data())); })
        .def_property_readonly("weights", [](const HostPhotonBatch &b) { return py::array_t<double>(b.weights.size(), b.weights.data()); });


    m.def("configure_detector_chord", &configure_detector_chord,
          py::arg("detector"), py::arg("sphere"), py::arg("port_hole_radius_mm"),
          "Configure detector position for chord surface geometry");

    m.def("calculate_theory", &TheoryCalculator::calculateWithPorts,
          py::arg("radius"), py::arg("reflectance"),
          py::arg("incident_power"), py::arg("port_area"),
          "Calculate theoretical result using Goebel formula");
          

    // 常量
    m.attr("PI") = PI;
}