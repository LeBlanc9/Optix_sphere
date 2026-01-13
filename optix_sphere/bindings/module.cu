#include <pybind11/pybind11.h>

void bind_logging(pybind11::module_ &m);
void bind_types(pybind11::module_ &m);
void bind_sim(pybind11::module_ &m);
void bind_photon(pybind11::module_ &m);
void bind_material(pybind11::module_ &m);
void bind_media(pybind11::module_ &m);

PYBIND11_MODULE(_core, m) {
    m.doc() = "OptiX Sphere - Monte Carlo simulation for integrating spheres";
    m.attr("__version__") = "0.1.0";

    bind_logging(m);
    bind_types(m);
    bind_sim(m);
    bind_photon(m);
    bind_material(m);
    bind_media(m);
}
