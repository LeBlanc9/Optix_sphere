#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "photon/sources.h"
#include "photon/photon_batch.h"
#include "photon/photon_transform.cuh"

namespace py = pybind11;
using namespace phonder;

void bind_photon(py::module_ &m) {
    py::class_<HostPhotonBatch>(m, "HostPhotonBatch")
        .def(py::init<>())
        .def("size", &HostPhotonBatch::size)
        .def_property_readonly("positions", [](const HostPhotonBatch &b) {
            std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(b.size()), 3};
            return py::array_t<float>(shape, reinterpret_cast<const float*>(b.positions.data()));
        })
        .def_property_readonly("directions", [](const HostPhotonBatch &b) {
            std::vector<py::ssize_t> shape = {static_cast<py::ssize_t>(b.size()), 3};
            return py::array_t<float>(shape, reinterpret_cast<const float*>(b.directions.data()));
        })
        .def_property_readonly("weights", [](const HostPhotonBatch &b) {
            return py::array_t<double>(b.weights.size(), b.weights.data());
        });

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
          "Translate photon positions in-place (modifies batch directly)."
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
}

