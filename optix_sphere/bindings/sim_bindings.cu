#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "simulator.h"
#include "theory/theory.h"
#include "material.h"
#include "scene/scene.h"
#include "geometry/obj_loader.h"
#include "photon/sources.h"
#include "photon/photon_batch.h"

namespace py = pybind11;
using namespace phonder;
using namespace theory;

void bind_sim(py::module_ &m) {
    // Bind Mesh structure
    py::class_<Mesh>(m, "Mesh")
        .def(py::init<>())
        .def_readonly("vertices", &Mesh::vertices,
             "List of vertex positions (float3)")
        .def_readonly("normals", &Mesh::normals,
             "List of vertex normals (float3)")
        .def_readonly("indices", &Mesh::indices,
             "List of triangle indices (uint3)")
        .def_readonly("triangle_materials", &Mesh::triangle_materials,
             "List of material indices per triangle (int)")
        .def_readonly("material_names", &Mesh::material_names,
             "List of material names (str)")
        .def("get_triangle_count", &Mesh::get_triangle_count,
             "Get number of triangles in the mesh")
        .def("get_vertex_count", &Mesh::get_vertex_count,
             "Get number of vertices in the mesh")
        .def("get_material_count", &Mesh::get_material_count,
             "Get number of materials in the mesh")
        .def("get_triangle_count_by_material", &Mesh::get_triangle_count_by_material,
             py::arg("material_name"),
             "Get number of triangles with a specific material")
        .def("get_vertex_count_by_material", &Mesh::get_vertex_count_by_material,
             py::arg("material_name"),
             "Get number of unique vertices used by a specific material")
        .def("get_triangle_indices_by_material", &Mesh::get_triangle_indices_by_material,
             py::arg("material_name"),
             "Get all triangle indices that use a specific material");

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
             "    tuple[float3, float3]: (min_corner, max_corner) in millimeters")
        .def("get_mesh", &Scene::get_mesh, py::return_value_policy::reference_internal,
             "Get the underlying mesh data.\n\n"
             "Returns the Mesh object containing all geometry data (vertices, normals, indices, etc.).\n"
             "The returned Mesh reflects any transformations applied to the Scene.\n\n"
             "Returns:\n"
             "    Mesh: Reference to the mesh data\n\n"
             "Example:\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> scene.rotate_y(180.0)\n"
             "    >>> mesh = scene.get_mesh()\n"
             "    >>> print(f'Vertices: {len(mesh.vertices)}')\n"
             "    >>> print(f'First vertex: ({mesh.vertices[0].x}, {mesh.vertices[0].y}, {mesh.vertices[0].z})')\n")
        .def("rotate_y", &Scene::rotate_y,
             py::arg("angle_degrees"),
             "Rotate the entire scene around Y-axis.\n\n"
             "This modifies the scene in-place by transforming all vertices and normals.\n"
             "Useful for aligning scene geometry or flipping coordinate systems.\n\n"
             "Args:\n"
             "    angle_degrees (float): Rotation angle in degrees\n"
             "                          (positive = counter-clockwise from top view)\n\n"
             "Example:\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> scene.rotate_y(180.0)  # Flip 180 degrees\n"
             "    >>> scene.rotate_y(90.0)   # Rotate 90 degrees CCW\n")
        .def("translate", &Scene::translate,
             py::arg("offset"),
             "Translate the entire scene by a given offset.\n\n"
             "This modifies the scene in-place by adding the offset to all vertices.\n"
             "Useful for centering the scene or aligning with photon batches.\n\n"
             "Args:\n"
             "    offset (float3): Translation vector (x, y, z) in millimeters\n\n"
             "Example:\n"
             "    >>> import optix_sphere._core as osg\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> \n"
             "    >>> # Move 10mm in X direction\n"
             "    >>> scene.translate(osg.float3(10.0, 0.0, 0.0))\n"
             "    >>> \n"
             "    >>> # Center at origin\n"
             "    >>> min_b, max_b = scene.get_bounds()\n"
             "    >>> center = osg.float3(\n"
             "    ...     (min_b.x + max_b.x) / 2,\n"
             "    ...     (min_b.y + max_b.y) / 2,\n"
             "    ...     (min_b.z + max_b.z) / 2\n"
             "    ... )\n"
             "    >>> scene.translate(osg.float3(-center.x, -center.y, -center.z))\n")
        .def("scale", &Scene::scale,
             py::arg("scale_factor"),
             "Scale the entire scene uniformly.\n\n"
             "This modifies the scene in-place by multiplying all vertex positions\n"
             "by the scale factor. Normals remain unit length.\n\n"
             "Args:\n"
             "    scale_factor (float): Scaling factor (1.0 = no change, 2.0 = double size)\n\n"
             "Example:\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> scene.scale(2.0)   # Double the size\n"
             "    >>> scene.scale(0.5)   # Halve the size\n")
        .def("get_triangle_count_by_material", &Scene::get_triangle_count_by_material,
             py::arg("material_name"),
             "Get number of triangles with a specific material.\n\n"
             "Args:\n"
             "    material_name (str): Material name to query\n\n"
             "Returns:\n"
             "    int: Number of triangles using this material, or 0 if material not found\n\n"
             "Example:\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> wall_tris = scene.get_triangle_count_by_material('wall_material')\n"
             "    >>> detector_tris = scene.get_triangle_count_by_material('detector_material')\n"
             "    >>> print(f'Wall: {wall_tris}, Detector: {detector_tris}')\n")
        .def("get_vertex_count_by_material", &Scene::get_vertex_count_by_material,
             py::arg("material_name"),
             "Get number of unique vertices used by triangles with a specific material.\n\n"
             "This counts unique vertex indices referenced by triangles of this material.\n"
             "A vertex shared between multiple materials will be counted for each material.\n\n"
             "Args:\n"
             "    material_name (str): Material name to query\n\n"
             "Returns:\n"
             "    int: Number of unique vertices used by this material, or 0 if material not found\n\n"
             "Example:\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> detector_verts = scene.get_vertex_count_by_material('detector_material')\n"
             "    >>> print(f'Detector uses {detector_verts} unique vertices')\n")
        .def("get_triangle_indices_by_material", &Scene::get_triangle_indices_by_material,
             py::arg("material_name"),
             "Get all triangle indices that use a specific material.\n\n"
             "Args:\n"
             "    material_name (str): Material name to query\n\n"
             "Returns:\n"
             "    list[int]: List of triangle indices (0-based), empty if material not found\n\n"
             "Example:\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> detector_tri_ids = scene.get_triangle_indices_by_material('detector_material')\n"
             "    >>> print(f'First detector triangle: {detector_tri_ids[0]}')\n");

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
