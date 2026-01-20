#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "simulator.h"
#include "theory/theory.h"
#include "material.h"
#include "scene/scene.h"
#include "photon/sources.h"
#include "photon/photon_batch.h"

namespace py = pybind11;
using namespace phonder;
using namespace theory;

void bind_sim(py::module_ &m) {
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

    py::class_<Scene>(m, "Scene")
        .def_static("from_obj", &Scene::from_obj,
             py::arg("file_path"),
             "Load scene from OBJ file (static factory method).\n\n"
             "This is the primary way to create a Scene. It loads geometry\n"
             "and material names from the OBJ file without binding material instances.\n\n"
             "Args:\n"
             "    file_path (str): Path to OBJ file\n\n"
             "Returns:\n"
             "    Scene: Loaded scene data\n\n"
             "Example:\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> print(scene.get_material_names())\n"
             "    ['wall', 'detector', 'port']")
        .def_static("from_mesh", &Scene::from_mesh,
             py::arg("mesh"),
             "Create scene from existing mesh data (static factory method).\n\n"
             "Args:\n"
             "    mesh (Mesh): Mesh data to wrap in a Scene\n\n"
             "Returns:\n"
             "    Scene: Scene object containing the mesh\n\n"
             "Example:\n"
             "    >>> mesh = create_procedural_mesh()\n"
             "    >>> scene = Scene.from_mesh(mesh)\n")
        .def("get_material_names", &Scene::get_material_names,
             "Get all material names found in the mesh.\n\n"
             "Returns:\n"
             "    list[str]: List of material names from OBJ file")
        .def("get_vertex_count", &Scene::get_vertex_count,
             "Get number of vertices in the mesh.\n\n"
             "Returns:\n"
             "    int: Vertex count")
        .def("get_triangle_count", &Scene::get_triangle_count,
             "Get number of triangles in the mesh.\n\n"
             "Returns:\n"
             "    int: Triangle count")
        .def("get_material_count", &Scene::get_material_count,
             "Get number of materials in the mesh.\n\n"
             "Returns:\n"
             "    int: Material count")
        .def("get_bounds", &Scene::get_bounds,
             "Get axis-aligned bounding box of the mesh.\n\n"
             "Returns:\n"
             "    tuple[float3, float3]: (min_corner, max_corner) in millimeters");

    py::class_<Simulator>(m, "Simulator")
        .def(py::init<int>(), py::arg("device_id") = 0,
             "Initializes the OptiX Simulator.\n\n"
             "Args:\n"
             "    device_id (int, optional): CUDA device ID. Defaults to 0.\n\n"
             "Example:\n"
             "    >>> # Use default GPU 0\n"
             "    >>> sim = Simulator()\n"
             "    >>> \n"
             "    >>> # Use GPU 1\n"
             "    >>> sim = Simulator(device_id=1)\n")
        .def("build_scene",
             &Simulator::build_scene,
             py::arg("cpu_scene"), py::arg("materials"),
             "Build GPU scene from Scene.\n\n"
             "Converts an independently created Scene into GPU acceleration structures.\n\n"
             "Args:\n"
             "    cpu_scene (Scene): CPU-side scene data\n"
             "    materials (dict[str, MaterialFactory]): Material name to factory mapping\n\n"
             "Example:\n"
             "    >>> from optix_sphere import Scene, Simulator, material\n"
             "    >>> \n"
             "    >>> # Load scene independently\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> print(scene.get_material_names())\n"
             "    >>> \n"
             "    >>> # Prepare materials\n"
             "    >>> materials = {\n"
             "    ...     'wall': material.lambertian(0.98),\n"
             "    ...     'detector': material.detector()\n"
             "    ... }\n"
             "    >>> \n"
             "    >>> # Build GPU scene\n"
             "    >>> sim = Simulator()\n"
             "    >>> sim.build_scene(scene, materials)\n"
             "    >>> result = sim.run(source, config)")
        .def("run", static_cast<SimulationResult (Simulator::*)(phonder::PhotonSource&, const SimConfig&)>(&Simulator::run),
             py::arg("photon_source"), py::arg("config"))
        .def("run", static_cast<SimulationResult (Simulator::*)(const phonder::PhotonBatch&, const SimConfig&)>(&Simulator::run),
             py::arg("source_batch"), py::arg("config"))
        .def("get_detector_total_area", &Simulator::get_detector_total_area)
        .def("update_material", &Simulator::update_material,
             py::arg("name"), py::arg("factory"),
             "Update a single material without rebuilding the scene geometry.\n\n"
             "This is a fast operation that only updates material parameters.\n"
             "Changes take effect on the next run().\n\n"
             "Args:\n"
             "    name (str): Material name to update (must exist in scene)\n"
             "    factory (MaterialFactory): Factory function to create new material\n\n"
             "Raises:\n"
             "    RuntimeError: If material name not found or scene not built\n\n"
             "Example:\n"
             "    >>> from optix_sphere import Simulator, material\n"
             "    >>> sim = Simulator()\n"
             "    >>> sim.build_scene_from_file('sphere.obj', config)\n"
             "    >>> \n"
             "    >>> # Update wall material reflectance\n"
             "    >>> sim.update_material('wall_material', material.lambertian(0.95))\n"
             "    >>> result = sim.run(source, config)  # Uses new material\n"
             "    >>> \n"
             "    >>> # Change to mixed material\n"
             "    >>> sim.update_material('wall_material', material.mixed(0.7, 0.3, 0.98))\n"
             "    >>> result = sim.run(source, config)  # Uses updated material\n");

}
