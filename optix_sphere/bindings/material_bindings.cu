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

    // Python 侧直接返回材质实例（自动调用 C++ factory）
    material_module.def("lambertian",
        [](float reflectance) {
            return material::lambertian(reflectance)();  // 调用 factory 得到实例
        },
        py::arg("reflectance"),
        "Create a Lambertian (purely diffuse) material.\n\n"
        "Args:\n"
        "    reflectance (float): Surface reflectance [0, 1]\n\n"
        "Returns:\n"
        "    Material: Lambertian material instance\n\n"
        "Example:\n"
        "    >>> mat = osg.material.lambertian(0.98)\n"
        "    >>> sim.material_pool.append(mat)\n");

    material_module.def("mixed",
        [](float diffuse_ratio, float specular_ratio, float reflectance) {
            return material::mixed(diffuse_ratio, specular_ratio, reflectance)();
        },
        py::arg("diffuse_ratio"),
        py::arg("specular_ratio"),
        py::arg("reflectance"),
        "Create a mixed (diffuse + specular) material.\n\n"
        "Args:\n"
        "    diffuse_ratio (float): Proportion of diffuse reflection [0, 1]\n"
        "    specular_ratio (float): Proportion of specular reflection [0, 1]\n"
        "    reflectance (float): Total surface reflectance [0, 1]\n\n"
        "Returns:\n"
        "    Material: Mixed material instance\n\n"
        "Example:\n"
        "    >>> mat = osg.material.mixed(0.7, 0.3, 0.99)\n"
        "    >>> sim.material_pool.append(mat)\n");

    material_module.def("detector",
        [](float na, float n) {
            return material::detector(na, n)();
        },
        py::arg("na") = 1.0f,
        py::arg("n") = 1.0f,
        "Create a detector material.\n\n"
        "Args:\n"
        "    na (float): Numerical Aperture [0, 1], default 1.0 (no angle limit)\n"
        "    n (float): Refractive index of medium, default 1.0 (air)\n\n"
        "Returns:\n"
        "    Material: Detector material instance\n\n"
        "Example:\n"
        "    >>> detector_mat = osg.material.detector()  # No NA limit\n"
        "    >>> detector_na05 = osg.material.detector(na=0.5)  # NA=0.5\n"
        "    >>> sim.material_pool.append(detector_mat)\n");

    material_module.def("absorber",
        []() {
            return material::absorber()();
        },
        "Create an absorber (perfect black body) material.\n\n"
        "Returns:\n"
        "    Material: Absorber material instance\n\n"
        "Example:\n"
        "    >>> absorber_mat = osg.material.absorber()\n"
        "    >>> sim.material_pool.append(absorber_mat)\n");

    material_module.def("spherical_lambertian",
        [](float reflectance, float3 center) {
            return material::spherical_lambertian(reflectance, center)();
        },
        py::arg("reflectance"),
        py::arg("center"),
        "Create a spherical Lambertian material (optimized for spheres).\n\n"
        "Args:\n"
        "    reflectance (float): Surface reflectance [0, 1]\n"
        "    center (float3): Sphere center position\n\n"
        "Returns:\n"
        "    Material: Spherical Lambertian material instance\n\n"
        "Example:\n"
        "    >>> mat = osg.material.spherical_lambertian(0.98, osg.float3(0, 0, 0))\n"
        "    >>> sim.material_pool.append(mat)\n");

    material_module.def("spherical_mixed",
        [](float diffuse_ratio, float specular_ratio, float reflectance, float3 center) {
            return material::spherical_mixed(diffuse_ratio, specular_ratio, reflectance, center)();
        },
        py::arg("diffuse_ratio"),
        py::arg("specular_ratio"),
        py::arg("reflectance"),
        py::arg("center"),
        "Create a spherical mixed material (optimized for spheres).\n\n"
        "Args:\n"
        "    diffuse_ratio (float): Proportion of diffuse reflection [0, 1]\n"
        "    specular_ratio (float): Proportion of specular reflection [0, 1]\n"
        "    reflectance (float): Total surface reflectance [0, 1]\n"
        "    center (float3): Sphere center position\n\n"
        "Returns:\n"
        "    Material: Spherical mixed material instance\n\n"
        "Example:\n"
        "    >>> mat = osg.material.spherical_mixed(0.7, 0.3, 0.99, osg.float3(0, 0, 0))\n"
        "    >>> sim.material_pool.append(mat)\n");

    material_module.def("get_default_materials", &material::get_default_materials,
                        "Get default material factory mapping.");
}
