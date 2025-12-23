#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "core/optix_context.h"
#include "scene/scene.h"
#include "simulation/path_tracer.h"
#include "theory/theory.h"
#include "scene/scene_types.h"
#include "simulation_result.h"
#include "constants.h"
#include "embedded_ptx.h"  // 嵌入的 PTX 代码

namespace py = pybind11;

// Python包装器类，简化使用
class Simulator {
public:
    Simulator() : context_(), scene_(context_) {
        // OptiX 初始化在 OptixContext 构造函数中完成
    }

    void setup_scene(const Sphere& sphere, const Detector& detector) {
        sphere_ = sphere;
        detector_ = detector;
        scene_.build_scene(sphere, detector);

        // 创建 PathTracer (使用嵌入的 PTX 代码)
        tracer_ = std::make_unique<PathTracer>(context_, scene_, embedded::g_forward_tracer_ptx, true);
    }

    SimulationResult run(const SimConfig& config, const LightSource& light) {
        if (!tracer_) {
            throw std::runtime_error("Scene not set up. Call setup_scene() first.");
        }
        return tracer_->launch(config, light, detector_);
    }

private:
    OptixContext context_;
    Scene scene_;
    std::unique_ptr<PathTracer> tracer_;
    Sphere sphere_;
    Detector detector_;
};

// Python 绑定模块
PYBIND11_MODULE(_core, m) {
    m.doc() = "OptiX Sphere - Monte Carlo simulation for integrating spheres";
    m.attr("__version__") = "0.1.0";

    // Sphere 类
    py::class_<Sphere>(m, "Sphere")
        .def(py::init<>())
        .def_readwrite("center", &Sphere::center)
        .def_readwrite("radius", &Sphere::radius)
        .def_readwrite("reflectance", &Sphere::reflectance)
        .def("__repr__", [](const Sphere& s) {
            return "<Sphere radius=" + std::to_string(s.radius) +
                   " reflectance=" + std::to_string(s.reflectance) + ">";
        });

    // LightSource 类
    py::class_<LightSource>(m, "LightSource")
        .def(py::init<>())
        .def_readwrite("position", &LightSource::position)
        .def_readwrite("power", &LightSource::power)
        .def("__repr__", [](const LightSource& l) {
            return "<LightSource power=" + std::to_string(l.power) + "W>";
        });

    // Detector 类
    py::class_<Detector>(m, "Detector")
        .def(py::init<>())
        .def_readwrite("position", &Detector::position)
        .def_readwrite("normal", &Detector::normal)
        .def_readwrite("radius", &Detector::radius)
        .def("__repr__", [](const Detector& d) {
            return "<Detector radius=" + std::to_string(d.radius) + "mm>";
        });

    // SimConfig 类
    py::class_<SimConfig>(m, "SimConfig")
        .def(py::init<>())
        .def_readwrite("num_rays", &SimConfig::num_rays)
        .def_readwrite("max_bounces", &SimConfig::max_bounces)
        .def_readwrite("use_nee", &SimConfig::use_nee)
        .def_readwrite("random_seed", &SimConfig::random_seed)
        .def("__repr__", [](const SimConfig& c) {
            return "<SimConfig num_rays=" + std::to_string(c.num_rays) +
                   " max_bounces=" + std::to_string(c.max_bounces) + ">";
        });

    // SimulationResult 类
    py::class_<SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readonly("detected_flux", &SimulationResult::detected_flux)
        .def_readonly("irradiance", &SimulationResult::irradiance)
        .def_readonly("total_rays", &SimulationResult::total_rays)
        .def_readonly("detected_rays", &SimulationResult::detected_rays)
        .def_readonly("avg_bounces", &SimulationResult::avg_bounces)
        .def("__repr__", [](const SimulationResult& r) {
            return "<SimulationResult detected_flux=" + std::to_string(r.detected_flux) +
                   "W irradiance=" + std::to_string(r.irradiance) + "W/mm²>";
        });

    // TheoryResult 类
    py::class_<TheoryResult>(m, "TheoryResult")
        .def(py::init<>())
        .def_readonly("avg_irradiance", &TheoryResult::avg_irradiance)
        .def_readonly("detected_flux", &TheoryResult::detected_flux)
        .def_readonly("sphere_area", &TheoryResult::sphere_area)
        .def_readonly("total_flux_in_sphere", &TheoryResult::total_flux_in_sphere)
        .def("__repr__", [](const TheoryResult& r) {
            return "<TheoryResult avg_irradiance=" + std::to_string(r.avg_irradiance) +
                   "W/mm²>";
        });

    // Simulator 类（主接口）
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<>())
        .def("setup_scene", &Simulator::setup_scene,
             py::arg("sphere"), py::arg("detector"),
             "Setup the simulation scene with sphere and detector")
        .def("run", &Simulator::run,
             py::arg("config"), py::arg("light"),
             "Run the Monte Carlo simulation")
        .def("__repr__", [](const Simulator&) {
            return "<Simulator>";
        });

    // 辅助函数
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
