#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h> // For spdlog::set_level
#include <spdlog/common.h> // For spdlog::level::level_enum and spdlog::to_string_view

#include "simulator.h" // New unified Simulator API
#include "theory/theory.h" // New theory API
#include "constants.h"
#include "photon/sources.h"     // New data-only source structs
#include "photon/batch.h"       // For HostPhotonBatch

namespace py = pybind11;
using namespace phonder; // For PhotonSource, IsotropicPointSource etc.
using namespace theory;  // For TheoryCalculator, TheoreticalIntegratingSphere, Port

// C++ function to set spdlog level
// Now directly accepts spdlog::level::level_enum
void set_log_level(spdlog::level::level_enum level) {
    spdlog::set_level(level);
    spdlog::info("Global log level set to {}.", spdlog::to_string_view(level));
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "OptiX Sphere - Monte Carlo simulation for integrating spheres";
    m.attr("__version__") = "0.1.0";

    // Bind spdlog::level::level_enum for Python control
    py::enum_<spdlog::level::level_enum>(m, "LogLevel", "Global logging levels for spdlog.")
        .value("TRACE", spdlog::level::trace)
        .value("DEBUG", spdlog::level::debug)
        .value("INFO", spdlog::level::info)
        .value("WARN", spdlog::level::warn)
        .value("ERROR", spdlog::level::err)
        .value("CRITICAL", spdlog::level::critical)
        .value("OFF", spdlog::level::off)
        .export_values(); // Exports values directly into the module (e.g., osg.INFO)


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

    // Bind new theoretical model classes
    py::class_<Port>(m, "Port")
        .def(py::init<>(), "Default constructor.")
        .def(py::init<float, float>(), py::arg("radius"), py::arg("reflectance"), "Constructs a Port with given radius and reflectance.")
        .def_readwrite("radius", &Port::radius)
        .def_readwrite("reflectance", &Port::reflectance);

    py::class_<TheoreticalIntegratingSphere>(m, "TheoreticalIntegratingSphere")
        .def(py::init<float, float>(), py::arg("radius"), py::arg("wall_reflectance"),
             "Constructs a TheoreticalIntegratingSphere with internal radius and wall reflectance.")
        .def("add_port", &TheoreticalIntegratingSphere::add_port, py::arg("radius"), py::arg("reflectance"),
             "Adds a port to the sphere model with specified radius and reflectance.")
        .def("get_radius", &TheoreticalIntegratingSphere::get_radius)
        .def("get_wall_reflectance", &TheoreticalIntegratingSphere::get_wall_reflectance)
        .def("get_total_sphere_area", &TheoreticalIntegratingSphere::get_total_sphere_area)
        .def("get_effective_wall_reflectance", &TheoreticalIntegratingSphere::get_effective_wall_reflectance);
    
    // Bind TheoryCalculator class (static methods)
    py::class_<TheoryCalculator>(m, "TheoryCalculator")
        .def(py::init<>(), "Placeholder constructor to allow class instantiation in Python (optional for static methods).")
        .def_static("calculate", &TheoryCalculator::calculate,
                    py::arg("sphere_model"), py::arg("incident_power"),
                    "Calculates the theoretical performance of an integrating sphere model.");

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
          
    // Bind the set_log_level function
    m.def("set_log_level", &set_log_level,
          py::arg("level"),
          "Sets the global logging level. Use LogLevel enum (e.g., osg.LogLevel.INFO).");


    // 常量
    m.attr("PI") = PI;
}