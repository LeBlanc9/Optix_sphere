#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>  // For std::function binding
#include <pybind11/numpy.h>

#include "simulator.h" // New unified Simulator API
#include "theory/theory.h" // New theory API
#include "constants.h"
#include "photon/sources.h"     // New data-only source structs
#include "photon/batch.h"       // For HostPhotonBatch
#include "photon/batch.cuh"     // For DevicePhotonBatch
#include "material.h"           // For material factory functions
#include "photon/photon_transform.cuh"  // For translate_photons

namespace py = pybind11;
using namespace phonder; // For PhotonSource, IsotropicPointSource etc.
using namespace theory;  // For TheoryCalculator, TheoreticalIntegratingSphere, Port


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
        .def(py::init<>());
        // Reserved for future extensions (e.g., global scaling, coordinate transforms)

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

    // Bind TheoryResult class
    py::class_<TheoryResult>(m, "TheoryResult")
        .def(py::init<>()) // Add default constructor for Python
        .def_readonly("avg_irradiance", &TheoryResult::avg_irradiance)
        .def_readonly("sphere_area", &TheoryResult::sphere_area)
        .def_readonly("total_flux_in_sphere", &TheoryResult::total_flux_in_sphere);

    // Bind the unified Simulator class
    py::class_<Simulator>(m, "Simulator")
        .def(py::init<>(), "Initializes the OptiX Simulator.")
        .def("build_scene_from_file",
             static_cast<void (Simulator::*)(const std::string&, const MeshSceneConfig&)>(&Simulator::build_scene_from_file),
             py::arg("file_path"), py::arg("config"),
             "Builds the scene from an OBJ file using the provided mesh configuration. "
             "The 'file_path' should be an absolute path to the .obj file.")
        .def("build_scene_from_file",
             static_cast<void (Simulator::*)(const std::string&, const std::map<std::string, MaterialFactory>&, const MeshSceneConfig&)>(&Simulator::build_scene_from_file),
             py::arg("file_path"), py::arg("materials"), py::arg("config"),
             "Builds the scene from an OBJ file with custom material factories. "
             "The 'materials' parameter is a dict mapping material names to factory functions.")
        .def("run", static_cast<SimulationResult (Simulator::*)(const phonder::PhotonSource&, const SimConfig&)>(&Simulator::run),
             py::arg("photon_source"), py::arg("config"),
             "Runs the Monte Carlo simulation with the given photon source and simulation configuration.")
        .def("run", static_cast<SimulationResult (Simulator::*)(const phonder::DevicePhotonBatch&, const SimConfig&)>(&Simulator::run),
             py::arg("source_batch"), py::arg("config"),
             "Runs the Monte Carlo simulation with a pre-generated GPU photon batch (e.g., from MediaSimulator).")
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

    // DevicePhotonBatch (GPU-resident photon batch)
    py::class_<DevicePhotonBatch>(m, "DevicePhotonBatch")
        .def(py::init<>())
        .def("size", &DevicePhotonBatch::size)
        .def("empty", &DevicePhotonBatch::empty)
        .def("clear", &DevicePhotonBatch::clear)
        .def("to_host", &DevicePhotonBatch::to_host,
             "Transfer photon batch from GPU to CPU memory");


    m.def("configure_detector_chord", &configure_detector_chord,
          py::arg("detector"), py::arg("sphere"), py::arg("port_hole_radius_mm"),
          "Configure detector position for chord surface geometry");

    m.def("translate_photons", &translate_photons,
          py::arg("input_batch"), py::arg("offset"),
          "Translate photon positions by a fixed offset.\n\n"
          "This function shifts all photon positions in a batch by a constant 3D offset.\n"
          "Useful for moving photons from MediaSimulator output to integrating sphere ports.\n\n"
          "Args:\n"
          "    input_batch (DevicePhotonBatch): Input photon batch on GPU\n"
          "    offset (float3): 3D translation vector (mm)\n\n"
          "Returns:\n"
          "    DevicePhotonBatch: New batch with translated positions (directions unchanged)\n\n"
          "Example:\n"
          "    >>> # Move reflected photons to sphere port at x=25mm\n"
          "    >>> aligned_batch = osg.translate_photons(\n"
          "    ...     media_result.reflected_batch,\n"
          "    ...     osg.float3(25.0, 0.0, 0.0)\n"
          "    ... )\n");

    m.def("set_log_level", &set_log_level,
          py::arg("level"),
          "Set global logging level.\n\n"
          "Args:\n"
          "    level (int): Log level (0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical, 6=off)\n\n"
          "Example:\n"
          "    >>> import optix_sphere as osg\n"
          "    >>> osg.set_log_level(2)  # Set to INFO level\n");

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
    py::class_<SphericalLambertianMaterial, Material, std::shared_ptr<SphericalLambertianMaterial>>(m, "SphericalLambertianMaterial");
    py::class_<SphericalMixedMaterial, Material, std::shared_ptr<SphericalMixedMaterial>>(m, "SphericalMixedMaterial");

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
                       "Create a spherical Lambertian material factory (OPTIMIZED for spheres).\n\n"
                       "Uses spherical normal calculation - ~3-5x faster than triangle normals.\n"
                       "ONLY use for perfectly spherical surfaces without baffles.\n\n"
                       "Args:\n"
                       "    reflectance (float): Surface reflectance (0-1)\n\n"
                       "Returns:\n"
                       "    MaterialFactory: A factory function for creating spherical Lambertian materials\n\n"
                       "Example:\n"
                       "    >>> # For perfect sphere (faster)\n"
                       "    >>> materials = {}\n"
                       "    >>> materials['wall'] = material.spherical_lambertian(0.98)\n");

    material_module.def("spherical_mixed", &material::spherical_mixed,
                       py::arg("diffuse_ratio"),
                       py::arg("specular_ratio"),
                       py::arg("reflectance"),
                       "Create a spherical mixed material factory (OPTIMIZED for spheres).\n\n"
                       "Uses spherical normal calculation - ~3-5x faster than triangle normals.\n"
                       "ONLY use for perfectly spherical surfaces without baffles.\n\n"
                       "Args:\n"
                       "    diffuse_ratio (float): Fraction using Lambertian scattering (0-1)\n"
                       "    specular_ratio (float): Fraction using specular reflection (0-1)\n"
                       "    reflectance (float): Total reflectance (0-1)\n\n"
                       "Note:\n"
                       "    diffuse_ratio + specular_ratio should equal 1.0\n\n"
                       "Returns:\n"
                       "    MaterialFactory: A factory function for creating spherical mixed materials\n\n"
                       "Example:\n"
                       "    >>> # For perfect sphere with realistic surface (faster)\n"
                       "    >>> materials = {}\n"
                       "    >>> materials['wall'] = material.spherical_mixed(0.7, 0.3, 0.98)\n");

    material_module.def("get_default_materials", &material::get_default_materials,
                       "Get default material factory mapping.\n\n"
                       "Returns:\n"
                       "    dict: A dictionary mapping common OBJ material names to default factories\n\n"
                       "Example:\n"
                       "    >>> materials = material.get_default_materials()\n"
                       "    >>> # Modify specific materials as needed\n"
                       "    >>> materials['wall_material'] = material.lambertian(0.99)\n");

    // 常量
    m.attr("PI") = PI;
}