#pragma once

#include "mesh.h"
#include "../material.h"
#include <string>
#include <map>

/**
 * @brief OBJ file loader
 *
 * Loads triangle mesh geometry and material names from Wavefront OBJ files.
 */
class ObjLoader {
public:
    /**
     * @brief Load mesh geometry and material names from OBJ file
     * @param filepath Path to OBJ file
     * @return Mesh with geometry and material name references
     */
    static Mesh load_obj(const std::string& filepath);

    /**
     * @brief Get default material factories
     * @return Map from material names to factory functions
     */
    static std::map<std::string, MaterialFactory> get_default_materials();
};
