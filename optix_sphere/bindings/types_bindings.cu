#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

namespace py = pybind11;

void bind_types(py::module_ &m) {
    // float3
    py::class_<float3>(m, "float3")
        .def(py::init<float, float, float>())
        .def(py::init([](py::tuple t) {
            if (t.size() != 3)
                throw std::runtime_error("float3 requires tuple of size 3");
            return make_float3(t[0].cast<float>(), t[1].cast<float>(), t[2].cast<float>());
        }))
        .def(py::init([](py::list l) {
            if (l.size() != 3)
                throw std::runtime_error("float3 requires list of size 3");
            return make_float3(l[0].cast<float>(), l[1].cast<float>(), l[2].cast<float>());
        }))
        .def_readwrite("x", &float3::x)
        .def_readwrite("y", &float3::y)
        .def_readwrite("z", &float3::z)
        .def("__repr__", [](const float3 &v) {
            return "float3(" + std::to_string(v.x) + ", " +
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

    py::implicitly_convertible<py::tuple, float3>();
    py::implicitly_convertible<py::list, float3>();

    // int3
    py::class_<int3>(m, "int3")
        .def(py::init<int, int, int>())
        .def(py::init([](py::tuple t) {
            if (t.size() != 3)
                throw std::runtime_error("int3 requires tuple of size 3");
            return make_int3(t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>());
        }))
        .def(py::init([](py::list l) {
            if (l.size() != 3)
                throw std::runtime_error("int3 requires list of size 3");
            return make_int3(l[0].cast<int>(), l[1].cast<int>(), l[2].cast<int>());
        }))
        .def_readwrite("x", &int3::x)
        .def_readwrite("y", &int3::y)
        .def_readwrite("z", &int3::z)
        .def("__repr__", [](const int3 &v) {
            return "int3(" + std::to_string(v.x) + ", " +
                   std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
        });

    py::implicitly_convertible<py::tuple, int3>();
    py::implicitly_convertible<py::list, int3>();
}
