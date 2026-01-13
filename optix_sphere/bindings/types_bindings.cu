#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

namespace py = pybind11;

void bind_types(py::module_ &m) {
    py::class_<float3>(m, "float3")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &float3::x)
        .def_readwrite("y", &float3::y)
        .def_readwrite("z", &float3::z);
}
