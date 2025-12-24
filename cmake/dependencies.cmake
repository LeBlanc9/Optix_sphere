# External dependencies management (via Git Submodules)
# This file adds the dependencies stored as Git submodules in the 'vendor' directory.

message(STATUS "Loading dependencies from 'vendor' directory...")

# Google Test
# Disable unnecessary options before adding the subdirectory.
set(BUILD_GMOCK OFF CACHE BOOL "")
set(INSTALL_GTEST OFF CACHE BOOL "")
add_subdirectory(vendor/googletest)
message(STATUS "  -> Loaded googletest")

# spdlog
add_subdirectory(vendor/spdlog)
message(STATUS "  -> Loaded spdlog")

# tinyobjloader
# The CMakeLists for tinyobjloader allows disabling tests, which we don't need.
set(TINYOBJLOADER_BUILD_TEST OFF CACHE BOOL "")
add_subdirectory(vendor/tinyobjloader)
message(STATUS "  -> Loaded tinyobjloader")

# nlohmann/json
# This is a header-only library, but add_subdirectory is the standard way
# to make its INTERFACE target `nlohmann_json::nlohmann_json` available.
set(JSON_BuildTests OFF CACHE BOOL "")
add_subdirectory(vendor/json)
message(STATUS "  -> Loaded nlohmann_json")

message(STATUS "✅ All external dependencies loaded from submodules.")
