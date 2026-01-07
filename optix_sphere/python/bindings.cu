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
#include "layered_media/layered_medium.cuh"
#include "layered_media/media_simulator.cuh"
#include "layered_media/layered_medium.cuh"
#include "layered_media/media_simulator.cuh"
#include <spdlog/spdlog.h>

namespace py = pybind11;
using namespace phonder;
using namespace theory;

PYBIND11_MODULE(_core, m) {
    m.doc() = "OptiX Sphere - Monte Carlo simulation for integrating spheres";
    m.attr("__version__") = "0.1.0";

    py::enum_<spdlog::level::level_enum>(m, "LogLevel")
        .value("TRACE", spdlog::level::trace)
        .value("DEBUG", spdlog::level::debug)
        .value("INFO", spdlog::level::info)
        .value("WARN", spdlog::level::warn)
        .value("ERROR", spdlog::level::err)
        .value("CRITICAL", spdlog::level::critical)
        .value("OFF", spdlog::level::off)
        .export_values();

    m.def("set_log_level", [](spdlog::level::level_enum level) {
        spdlog::set_level(level);
    }, py::arg("level"));


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
        .def("total_weight", &PhotonBatch::total_weight)
        .def("to_host", &PhotonBatch::to_host,
             "Transfer photon batch from GPU to CPU memory, returning a HostPhotonBatch.");


    m.def("translate_photons", &translate_photons,
          py::arg("input_batch"), py::arg("offset"),
          "Translate photon positions by a fixed offset (creates new batch).\n\n"
          "Args:\n"
          "    input_batch (PhotonBatch): Input photon batch on GPU\n"
          "    offset (float3): Translation vector (mm)\n\n"
          "Returns:\n"
          "    PhotonBatch: New batch with translated positions\n\n"
          "Example:\n"
          "    >>> batch = osg.generate_photons(source, 10000)\n"
          "    >>> translated = osg.translate_photons(batch, osg.float3(0, 0, -12.7))\n"
          "    >>> # Original batch is unchanged\n");

    m.def("translate_photons_inplace", &translate_photons_inplace,
          py::arg("batch"), py::arg("offset"),
          "Translate photon positions in-place (modifies batch directly).\n\n"
          "More efficient than translate_photons() as it avoids memory allocation.\n"
          "Use this for large batches to save memory.\n\n"
          "Args:\n"
          "    batch (PhotonBatch): Photon batch to modify (will be changed!)\n"
          "    offset (float3): Translation vector (mm)\n\n"
          "Returns:\n"
          "    None (modifies batch in-place)\n\n"
          "Example:\n"
          "    >>> batch = osg.generate_photons(source, 10000)\n"
          "    >>> osg.translate_photons_inplace(batch, osg.float3(0, 0, -12.7))\n"
          "    >>> # batch has been modified\n");

    py::class_<IsotropicPointSource>(m, "IsotropicPointSource")
        .def(py::init<>())
        .def_readwrite("position", &IsotropicPointSource::position)
        .def_readwrite("weight", &IsotropicPointSource::weight);

    py::class_<CollimatedBeamSource>(m, "CollimatedBeamSource")
        .def(py::init<>())
        .def_readwrite("position", &CollimatedBeamSource::position)
        .def_readwrite("direction", &CollimatedBeamSource::direction)
        .def_readwrite("weight", &CollimatedBeamSource::weight);

    // ============================================
    // Photon Generation Convenience Functions
    // ============================================

    // Unified photon generation function (works with any source type)
    m.def("generate_photons",
          [](const PhotonSource& source, int num_photons, unsigned long long seed = 42) {
              PhotonBatch batch;
              generate_photons_on_device(source, batch, num_photons, seed);
              return batch;
          },
          py::arg("source"), py::arg("num_photons"), py::arg("seed") = 42,
          "Generate photons from any source type on the GPU.\n\n"
          "This is the unified interface that works with all source types:\n"
          "  - IsotropicPointSource\n"
          "  - CollimatedBeamSource\n"
          "  - SpotSource\n"
          "  - GaussianBeamSource\n"
          "  - FocusedSpotSource\n\n"
          "Args:\n"
          "    source: Any PhotonSource type (see list above)\n"
          "    num_photons (int): Number of photons to generate\n"
          "    seed (int, optional): Random seed for generation. Defaults to 42.\n\n"
          "Returns:\n"
          "    PhotonBatch: A batch of photons on the GPU\n\n"
          "Example:\n"
          "    >>> # Collimated beam\n"
          "    >>> source = osg.CollimatedBeamSource()\n"
          "    >>> source.position = osg.float3(0, 0, -1)\n"
          "    >>> source.direction = osg.float3(0, 0, 1)\n"
          "    >>> batch = osg.generate_photons(source, 10000)\n"
          "    >>> \n"
          "    >>> # Isotropic point source\n"
          "    >>> source = osg.IsotropicPointSource()\n"
          "    >>> source.position = osg.float3(0, 0, 0)\n"
          "    >>> batch = osg.generate_photons(source, 10000)\n");

    // ============================================
    // Material System
    // ============================================
    // Bind Material base class with shared_ptr as holder
    // This is required for MaterialFactory functions to work with pybind11
    py::class_<Material, std::shared_ptr<Material>>(m, "Material")
        .def("get_kernel_name", &Material::get_kernel_name,
             "Returns the OptiX kernel name for this material");

    // Also bind concrete material types (as opaque types)
    py::class_<LambertianMaterial, Material, std::shared_ptr<LambertianMaterial>>(m, "LambertianMaterial");
    py::class_<MixedMaterial, Material, std::shared_ptr<MixedMaterial>>(m, "MixedMaterial");
    py::class_<DetectorMaterial, Material, std::shared_ptr<DetectorMaterial>>(m, "DetectorMaterial");
    py::class_<AbsorberMaterial, Material, std::shared_ptr<AbsorberMaterial>>(m, "AbsorberMaterial");

    // Create a submodule for material factory functions
    py::module_ material_module = m.def_submodule("material", "Material factory functions for creating custom materials");

    // Bind material factory functions
    material_module.def("lambertian", &material::lambertian,
                       py::arg("reflectance"),
                       "Create a Lambertian (purely diffuse) material factory.\n\n"
                       "Args:\n"
                       "    reflectance (float): Surface reflectance (0-1)\n\n"
                       "Returns:\n"
                       "    MaterialFactory: A factory function for creating Lambertian materials\n\n"
                       "Example:\n"
                       "    >>> materials = {}\n"
                       "    >>> materials['wall'] = material.lambertian(0.98)\n");

    material_module.def("mixed", &material::mixed,
                       py::arg("diffuse_ratio"),
                       py::arg("specular_ratio"),
                       py::arg("reflectance"),
                       "Create a mixed (diffuse + specular) material factory.\n\n"
                       "Args:\n"
                       "    diffuse_ratio (float): Fraction using Lambertian scattering (0-1)\n"
                       "    specular_ratio (float): Fraction using specular reflection (0-1)\n"
                       "    reflectance (float): Total reflectance (0-1)\n\n"
                       "Note:\n"
                       "    diffuse_ratio + specular_ratio should equal 1.0\n\n"
                       "Returns:\n"
                       "    MaterialFactory: A factory function for creating mixed materials\n\n"
                       "Example:\n"
                       "    >>> materials = {}\n"
                       "    >>> materials['wall'] = material.mixed(0.7, 0.3, 0.98)  # 70% diffuse, 30% specular\n");

    material_module.def("detector", &material::detector,
                       "Create a detector material factory.\n\n"
                       "Returns:\n"
                       "    MaterialFactory: A factory function for creating detector materials\n\n"
                       "Example:\n"
                       "    >>> materials = {}\n"
                       "    >>> materials['detector'] = material.detector()\n");

    material_module.def("absorber", &material::absorber,
                       "Create an absorber (perfect black body) material factory.\n\n"
                       "Returns:\n"
                       "    MaterialFactory: A factory function for creating absorber materials\n\n"
                       "Example:\n"
                       "    >>> materials = {}\n"
                       "    >>> materials['porthole'] = material.absorber()\n");

    material_module.def("spherical_lambertian", &material::spherical_lambertian,
                       py::arg("reflectance"),
                       "Create a spherical Lambertian material factory (optimized for spheres).\n\n"
                       "Uses spherical normal calculation - ~3-5x faster than triangle normals.\n"
                       "ONLY use for perfectly spherical surfaces without baffles.\n\n"
                       "Args:\n"
                       "    reflectance (float): Surface reflectance (0-1)\n");

    material_module.def("spherical_mixed", &material::spherical_mixed,
                       py::arg("diffuse_ratio"),
                       py::arg("specular_ratio"),
                       py::arg("reflectance"),
                       "Create a spherical mixed material factory (optimized for spheres).\n\n"
                       "Uses spherical normal calculation - ~3-5x faster than triangle normals.\n"
                       "ONLY use for perfectly spherical surfaces without baffles.\n\n"
                       "Args:\n"
                       "    diffuse_ratio (float): Fraction using Lambertian scattering (0-1)\n"
                       "    specular_ratio (float): Fraction using specular reflection (0-1)\n"
                       "    reflectance (float): Total reflectance (0-1)\n");

    material_module.def("get_default_materials", &material::get_default_materials,
                       "Get default material factory mapping.\n\n"
                       "Returns:\n"
                       "    dict: A dictionary mapping common OBJ material names to default factories\n\n"
                       "Example:\n"
                       "    >>> materials = material.get_default_materials()\n"
                       "    >>> # Modify specific materials as needed\n"
                       "    >>> materials['wall_material'] = material.lambertian(0.99)\n");

    // ============================================
    // Layered Media Simulation
    // ============================================
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