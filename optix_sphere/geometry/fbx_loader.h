#pragma once

#include <string>
#include <vector>
#include <assimp/scene.h>

namespace fbx_loader {

/**
 * @brief 使用 Assimp 加载 3D 模型文件。
 * 
 * 这个函数会加载指定的模型文件（如 .fbx, .gltf, .obj 等）并返回
 * Assimp 场景对象的常量指针。Assimp 负责管理返回场景的内存。
 * 
 * @param file_path 模型的路径。
 * @return const aiScene* 如果加载成功，返回一个指向 Assimp 场景对象的指针；
 *         如果失败，则返回 nullptr。
 */
const aiScene* load_scene(const std::string& file_path);

// 未来可以添加更多函数，例如：
// - 从 aiScene 中提取特定节点（如 _locator_detector）的变换矩阵。
// - 将 aiMesh 转换为我们项目内部的顶点/索引格式。

} // namespace fbx_loader
