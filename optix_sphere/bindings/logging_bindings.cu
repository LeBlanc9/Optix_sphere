#include <pybind11/pybind11.h>
#include <spdlog/spdlog.h>

namespace py = pybind11;

void bind_logging(py::module_ &m) {
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
}
