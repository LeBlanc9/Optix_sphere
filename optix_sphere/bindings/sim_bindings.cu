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
        .def_static("from_obj", &Mesh::from_obj,
             py::arg("file_path"),
             "Load mesh from OBJ file.\n\n"
             "Args:\n"
             "    file_path (str): Path to OBJ file\n\n"
             "Returns:\n"
             "    Mesh: Loaded mesh object\n\n"
             "Example:\n"
             "    >>> mesh = Mesh.from_obj('sphere.obj')\n"
             "    >>> print(mesh.get_triangle_count())\n")
        .def("to_obj", &Mesh::to_obj,
             py::arg("obj_file_path"),
             py::arg("mtl_file_path") = "",
             "Export mesh to OBJ and MTL files.\n\n"
             "Args:\n"
             "    obj_file_path (str): Output OBJ path\n"
             "    mtl_file_path (str, optional): Output MTL path. If empty,\n"
             "        writes <obj_basename>.mtl in the same directory.\n\n"
             "Example:\n"
             "    >>> mesh = Mesh.from_obj('sphere.obj')\n"
             "    >>> mesh.to_obj('sphere_copy.obj')\n")
        .def("copy", [](const Mesh& self) { return Mesh(self); },
             "Create a deep copy of the mesh.\n\n"
             "Returns:\n"
             "    Mesh: Independent copy of the mesh\n\n"
             "Example:\n"
             "    >>> mesh = Mesh.from_obj('sphere.obj')\n"
             "    >>> mesh_copy = mesh.copy()\n"
             "    >>> mesh_copy.remove_mesh_by_material('wall')\n"
             "    >>> # Original mesh is unchanged\n"
             "    >>> print(mesh.get_material_count())  # Still has 'wall'\n")
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
        .def("get_bounds", &Mesh::get_bounds,
             "Get axis-aligned bounding box of the mesh.\n\n"
             "Returns:\n"
             "    tuple[float3, float3]: (min_corner, max_corner) in millimeters\n\n"
             "Example:\n"
             "    >>> mesh = osg.Mesh.from_obj('sphere.obj')\n"
             "    >>> min_pt, max_pt = mesh.get_bounds()\n"
             "    >>> print(f'X: [{min_pt.x}, {max_pt.x}]')\n"
             "    >>> print(f'Y: [{min_pt.y}, {max_pt.y}]')\n"
             "    >>> print(f'Z: [{min_pt.z}, {max_pt.z}]')\n")
        .def("get_triangle_count_by_material", &Mesh::get_triangle_count_by_material,
             py::arg("material_name"),
             "Get number of triangles with a specific material")
        .def("get_vertex_count_by_material", &Mesh::get_vertex_count_by_material,
             py::arg("material_name"),
             "Get number of unique vertices used by a specific material")
        .def("split_by_material", &Mesh::split_by_material,
             "Split mesh into separate meshes by material.\n\n"
             "Returns:\n"
             "    dict[str, Mesh]: Map of material_name -> Mesh\n\n"
             "Example:\n"
             "    >>> mesh = Mesh.from_obj('sphere.obj')\n"
             "    >>> parts = mesh.split_by_material()\n"
             "    >>> wall_mesh = parts['wall']\n"
             "    >>> detector_mesh = parts['detector']\n")
        .def("extract_mesh_by_material", &Mesh::extract_mesh_by_material,
             py::arg("material_name"),
             "Extract a new mesh containing only the specified material.\n\n"
             "Args:\n"
             "    material_name (str): Material name to extract\n\n"
             "Returns:\n"
             "    Mesh: New mesh with only triangles of this material\n\n"
             "Example:\n"
             "    >>> mesh = Mesh.from_obj('sphere.obj')\n"
             "    >>> wall_mesh = mesh.extract_mesh_by_material('wall')\n"
             "    >>> print(wall_mesh.get_triangle_count())\n")
        .def("remove_mesh_by_material", &Mesh::remove_mesh_by_material,
             py::arg("material_name"),
             "Remove triangles with the specified material (in-place modification).\n\n"
             "Args:\n"
             "    material_name (str): Material name to remove\n\n"
             "Note:\n"
             "    This modifies the mesh in-place. The specified material and its\n"
             "    triangles are removed from this mesh object.\n\n"
             "Example:\n"
             "    >>> mesh = Mesh.from_obj('sphere.obj')\n"
             "    >>> print(mesh.get_material_count())  # e.g., 3\n"
             "    >>> mesh.remove_mesh_by_material('sample_holder')\n"
             "    >>> print(mesh.get_material_count())  # e.g., 2\n")
        .def("info", &Mesh::info,
             "Print detailed mesh statistics.\n\n"
             "Displays:\n"
             "  - Total vertices and triangles\n"
             "  - Materials with triangle/vertex counts and percentages\n\n"
             "Example:\n"
             "    >>> mesh = Mesh.from_obj('sphere.obj')\n"
             "    >>> mesh.info()\n"
             "    ========================================\n"
             "    Mesh Statistics\n"
             "    ========================================\n"
             "    Vertices: 642\n"
             "    Triangles: 1280\n"
             "    Materials: 2\n"
             "    ...\n")
        .def("translate", &Mesh::translate,
             py::arg("offset"),
             "Translate the mesh by a given offset (in-place).\n\n"
             "Args:\n"
             "    offset (float3): Translation vector (x, y, z) in millimeters\n\n"
             "Example:\n"
             "    >>> import optix_sphere as osg\n"
             "    >>> mesh = osg.Mesh.from_obj('sphere.obj')\n"
             "    >>> mesh.translate(osg.float3(10.0, 0.0, 0.0))  # Move 10mm in X\n")
        .def("rotate_y", &Mesh::rotate_y,
             py::arg("angle_degrees"),
             "Rotate the mesh around Y-axis (in-place).\n\n"
             "Args:\n"
             "    angle_degrees (float): Rotation angle in degrees\n"
             "                          (positive = counter-clockwise from top view)\n\n"
             "Example:\n"
             "    >>> mesh = osg.Mesh.from_obj('sphere.obj')\n"
             "    >>> mesh.rotate_y(180.0)  # Flip 180 degrees\n")
        .def("scale", &Mesh::scale,
             py::arg("scale_factor"),
             "Scale the mesh uniformly (in-place).\n\n"
             "Args:\n"
             "    scale_factor (float): Scaling factor (1.0 = no change, 2.0 = double size)\n\n"
             "Example:\n"
             "    >>> mesh = osg.Mesh.from_obj('sphere.obj')\n"
             "    >>> mesh.scale(2.0)   # Double the size\n"
             "    >>> mesh.scale(0.5)   # Halve the size\n");

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
        .def("info", &Scene::info,
             "Print detailed scene statistics and node information.\n\n"
             "Displays:\n"
             "  - Total nodes (enabled/disabled)\n"
             "  - Total triangles and vertices\n"
             "  - Per-node mesh information\n"
             "  - Materials and triangle counts per material\n"
             "  - Transform information (if non-identity)\n\n"
             "Example:\n"
             "    >>> scene = Scene.from_obj('sphere.obj')\n"
             "    >>> scene.info()\n"
             "    ========================================\n"
             "    Scene Statistics\n"
             "    ========================================\n"
             "    Total Nodes: 1\n"
             "      Enabled: 1\n"
             "      Disabled: 0\n"
             "    ...\n");

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
        .def_readwrite("config", &Simulator::config,
             "Simulation configuration parameters.\n\n"
             "Can be modified directly before calling run().\n\n"
             "Example:\n"
             "    >>> sim = Simulator()\n"
             "    >>> sim.config.num_rays = 2000000\n"
             "    >>> sim.config.max_bounces = 500\n"
             "    >>> sim.config.use_nee = True\n"
             "    >>> sim.config.random_seed = 12345\n"
             "    >>> result = sim.run(source)\n")
        .def("get_material_count", &Simulator::get_material_count,
             "Get the number of materials registered in the simulator.\n\n"
             "Returns:\n"
             "    int: Number of materials\n")
        .def("get_material", &Simulator::get_material,
             py::arg("material_name"),
             "Get material instance by name.\n\n"
             "Args:\n"
             "    material_name (str): Material name\n\n"
             "Returns:\n"
             "    Material: Material instance\n\n"
             "Example:\n"
             "    >>> wall_mat = sim.get_material('wall')\n"
             "    >>> wall_mat.reflectance = 0.99\n")
        .def("set_material", &Simulator::set_material,
             py::arg("material_name"), py::arg("material"), py::arg("sync_to_gpu") = false,
             "Set material for a specific name.\n\n"
             "Args:\n"
             "    material_name (str): Name of material to set\n"
             "    material (Material): New material instance\n"
             "    sync_to_gpu (bool, optional): Whether to sync to GPU immediately. Defaults to False.\n\n"
             "Example:\n"
             "    >>> # Update and sync in one go\n"
             "    >>> sim.set_material('wall', material.lambertian(0.95), sync_to_gpu=True)\n")
        .def("get_material_names", &Simulator::get_material_names,
             "Get all registered material names.\n\n"
             "Returns:\n"
             "    list[str]: List of material names\n")
        .def("clear_materials", &Simulator::clear_materials,
             "Clear all materials and mappings.\n\n"
             "Example:\n"
             "    >>> sim.clear_materials()\n")
        .def("build_scene",
             static_cast<void (Simulator::*)(
                 const Scene&,
                 const std::map<std::string, size_t>&,
                 bool
             )>(&Simulator::build_scene),
             py::arg("cpu_scene"), py::arg("material_mapping"), py::arg("flip_detector_normal") = false,
             "Build GPU scene from Scene using material pool indices.\n\n"
             "Converts an independently created Scene into GPU acceleration structures.\n\n"
             "Args:\n"
             "    cpu_scene (Scene): CPU-side scene data\n"
             "    material_mapping (dict[str, int]): Material name to pool index mapping\n"
             "    flip_detector_normal (bool, optional): Flip detector normal direction. Defaults to False.\n\n"
             "Example:\n"
             "    >>> sim = Simulator()\n"
             "    >>> idx0 = sim.add_material(material.lambertian(0.98))\n"
             "    >>> idx1 = sim.add_material(material.detector())\n"
             "    >>> mapping = {'wall': idx0, 'detector': idx1}\n"
             "    >>> sim.build_scene(scene, mapping)\n")
        .def("build_scene",
             static_cast<void (Simulator::*)(
                 const Scene&,
                 const std::map<std::string, std::shared_ptr<Material>>&,
                 bool
             )>(&Simulator::build_scene),
             py::arg("cpu_scene"),
             py::arg("materials"),
             py::arg("flip_detector_normal") = false,
             "Build GPU scene with direct material mapping (recommended).\n\n"
             "Args:\n"
             "    cpu_scene (Scene): CPU-side scene data\n"
             "    materials (dict[str, Material]): Material name to Material object\n"
             "    flip_detector_normal (bool): Flip detector normal if needed\n\n"
             "Example:\n"
             "    >>> materials = {\n"
             "    ...     'wall': osg.material.lambertian(0.98),\n"
             "    ...     'detector': osg.material.detector(na=0.37)\n"
             "    ... }\n"
             "    >>> sim.build_scene(scene, materials)\n"
             "    >>> \n"
             "    >>> # Get material index for dynamic updates\n"
             "    >>> wall_idx = sim.get_material_index('wall')\n"
             "    >>> sim.set_material(wall_idx, osg.material.lambertian(0.99))\n"
             "    >>> sim.update_materials()\n")
        .def("run", static_cast<SimulationResult (Simulator::*)(phonder::PhotonSource&)>(&Simulator::run),
             py::arg("photon_source"),
             "Run Monte Carlo simulation using a procedural photon source.\n\n"
             "Uses simulator.config for configuration.\n\n"
             "Args:\n"
             "    photon_source: Photon source object (e.g., SpotSource, IsotropicPointSource)\n\n"
             "Returns:\n"
             "    SimulationResult: Simulation results\n\n"
             "Example:\n"
             "    >>> sim = Simulator()\n"
             "    >>> sim.config.num_rays = 1000000\n"
             "    >>> sim.config.use_nee = True\n"
             "    >>> source = SpotSource()\n"
             "    >>> source.center_position = (0, 0, 10)\n"
             "    >>> result = sim.run(source)\n")
        .def("run", static_cast<SimulationResult (Simulator::*)(const phonder::PhotonBatch&)>(&Simulator::run),
             py::arg("source_batch"),
             "Run Monte Carlo simulation using a pre-generated photon batch.\n\n"
             "Uses simulator.config for configuration.\n\n"
             "Args:\n"
             "    source_batch: Pre-generated photon batch\n\n"
             "Returns:\n"
             "    SimulationResult: Simulation results\n\n"
             "Example:\n"
             "    >>> sim = Simulator()\n"
             "    >>> sim.config.max_bounces = 500\n"
             "    >>> batch = generate_photons(source, 1000000)\n"
             "    >>> result = sim.run(batch)\n")
        .def("get_detector_total_area", &Simulator::get_detector_total_area)
        .def("sync_to_gpu", &Simulator::sync_to_gpu, "Sync materials from CPU to GPU")
        .def("get_material_index", &Simulator::get_material_index,
             py::arg("material_name"),
             "Get material pool index by name.\n\n"
             "Args:\n"
             "    material_name (str): Material name\n\n"
             "Returns:\n"
             "    int: Material pool index\n\n"
             "Example:\n"
             "    >>> wall_idx = sim.get_material_index('wall')\n"
             "    >>> sim.set_material(wall_idx, osg.material.lambertian(0.99))\n")
        .def("get_material_indices", &Simulator::get_material_indices,
             "Get all material name to index mappings.\n\n"
             "Returns:\n"
             "    dict[str, int]: Material name to pool index mapping\n\n"
             "Example:\n"
             "    >>> indices = sim.get_material_indices()\n"
             "    >>> print(indices)  # {'wall': 0, 'detector': 1}\n");

}
