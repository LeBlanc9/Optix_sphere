#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "layered_media/layered_medium.cuh"
#include "layered_media/media_simulator.cuh"
#include "photon/photon_batch.h"

namespace py = pybind11;

void bind_media(py::module_ &m) {
    py::module_ media_module = m.def_submodule("media", "Layered media simulation functions");

    py::class_<Layer>(media_module, "Layer")
        .def(py::init<float, float, float, float, float>(),
             py::arg("n"), py::arg("mua"), py::arg("mus"), py::arg("g"), py::arg("d"))
        .def_readwrite("n", &Layer::n)
        .def_readwrite("mua", &Layer::mua)
        .def_readwrite("mus", &Layer::mus)
        .def_readwrite("g", &Layer::g)
        .def_readwrite("d", &Layer::d);

    py::class_<LayeredMedium>(media_module, "LayeredMedium")
        .def(py::init<float, float>(), py::arg("ambient_n"), py::arg("width") = 100.0f)
        .def("add_layer", &LayeredMedium::add_layer,
             py::arg("n"), py::arg("mua"), py::arg("mus"), py::arg("g"), py::arg("d"),
             py::return_value_policy::reference_internal)
        .def("set_width", &LayeredMedium::set_width,
             py::arg("w"), py::return_value_policy::reference_internal)
        .def("set_ambient_n", &LayeredMedium::set_ambient_n,
             py::arg("n"), py::return_value_policy::reference_internal)
        .def_readonly("ambient_n", &LayeredMedium::ambient_n)
        .def_readonly("num_layers", &LayeredMedium::num_layers)
        .def_readonly("width", &LayeredMedium::width)
        .def_readonly("total_thickness", &LayeredMedium::total_thickness);

    py::class_<MediaSimConfig>(media_module, "MediaSimConfig")
        .def(py::init<>())
        .def_readwrite("medium", &MediaSimConfig::medium)
        .def_readwrite("source", &MediaSimConfig::source)
        .def_readwrite("gpu_id", &MediaSimConfig::gpu_id)
        .def_readwrite("reflected_radius", &MediaSimConfig::reflected_radius)
        .def_readwrite("transmitted_radius", &MediaSimConfig::transmitted_radius);

    py::class_<HostMediaSimulationResult>(media_module, "HostMediaSimulationResult")
        .def(py::init<>())
        .def_readonly("reflected_batch", &HostMediaSimulationResult::reflected_batch)
        .def_readonly("transmitted_batch", &HostMediaSimulationResult::transmitted_batch)
        .def_readonly("specular_reflection_weight", &HostMediaSimulationResult::specular_reflection_weight);

    py::class_<MediaSimulationResult>(media_module, "MediaSimulationResult")
        .def(py::init<>())
        .def_readonly("reflected_batch", &MediaSimulationResult::reflected_batch)
        .def_readonly("transmitted_batch", &MediaSimulationResult::transmitted_batch)
        .def_readonly("specular_reflection_weight", &MediaSimulationResult::specular_reflection_weight)
        .def("to_host", &MediaSimulationResult::to_host);

    py::class_<MediaSimulator>(media_module, "MediaSimulator")
        .def(py::init<const MediaSimConfig&>(), py::arg("config"))
        .def("run", static_cast<MediaSimulationResult (MediaSimulator::*)(int)>(&MediaSimulator::run),
             py::arg("num_photons"))
        .def("run", static_cast<MediaSimulationResult (MediaSimulator::*)(const PhotonBatch&)>(&MediaSimulator::run),
             py::arg("input_batch"))
        .def("get_medium", &MediaSimulator::get_medium, py::return_value_policy::reference)
        .def("update_medium", &MediaSimulator::update_medium, py::arg("new_medium"));
}
