#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "material.h"

namespace py = pybind11;

void bind_material(py::module_ &m) {
    py::class_<Material, std::shared_ptr<Material>>(m, "Material")
        .def("get_kernel_name", &Material::get_kernel_name,
             "Returns the OptiX kernel name for this material");

    py::class_<LambertianMaterial, Material, std::shared_ptr<LambertianMaterial>>(m, "LambertianMaterial");
    py::class_<MixedMaterial, Material, std::shared_ptr<MixedMaterial>>(m, "MixedMaterial");
    py::class_<DetectorMaterial, Material, std::shared_ptr<DetectorMaterial>>(m, "DetectorMaterial");
    py::class_<AbsorberMaterial, Material, std::shared_ptr<AbsorberMaterial>>(m, "AbsorberMaterial");

    py::module_ material_module = m.def_submodule("material", "Material factory functions for creating custom materials");

    material_module.def("lambertian", &material::lambertian, py::arg("reflectance"),
                        "Create a Lambertian (purely diffuse) material factory.");

    material_module.def("mixed", &material::mixed,
                        py::arg("diffuse_ratio"),
                        py::arg("specular_ratio"),
                        py::arg("reflectance"),
                        "Create a mixed (diffuse + specular) material factory.");

    material_module.def("detector", &material::detector,
                        "Create a detector material factory.");

    material_module.def("absorber", &material::absorber,
                        "Create an absorber (perfect black body) material factory.");

    material_module.def("spherical_lambertian", &material::spherical_lambertian,
                        py::arg("reflectance"),
                        py::arg("center"),
                        "Create a spherical Lambertian material factory (optimized for spheres).");

    material_module.def("spherical_mixed", &material::spherical_mixed,
                        py::arg("diffuse_ratio"),
                        py::arg("specular_ratio"),
                        py::arg("reflectance"),
                        py::arg("center"),
                        "Create a spherical mixed material factory (optimized for spheres).");

    material_module.def("get_default_materials", &material::get_default_materials,
                        "Get default material factory mapping.");
}
