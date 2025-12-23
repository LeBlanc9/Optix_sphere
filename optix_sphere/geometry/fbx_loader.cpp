#include "geometry/fbx_loader.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <spdlog/spdlog.h>

#include <map>
#include <memory>

namespace fbx_loader {

// 使用静态变量来管理 Assimp Importer 和加载的场景。
// 这确保了 aiScene 对象在程序运行期间一直有效。
static std::map<std::string, std::unique_ptr<Assimp::Importer>> g_importers;

const aiScene* load_scene(const std::string& file_path) {
    if (g_importers.count(file_path)) {
        spdlog::info("FBX Loader: Found cached scene for '{}'", file_path);
        return g_importers[file_path]->GetScene();
    }

    spdlog::info("FBX Loader: Loading scene '{}' for the first time...", file_path);

    auto importer = std::make_unique<Assimp::Importer>();
    
    const aiScene* scene = importer->ReadFile(
        file_path,
        aiProcess_Triangulate |           // 将所有图元转换为三角形
        aiProcess_JoinIdenticalVertices | // 合并相同的顶点
        aiProcess_SortByPType |           // 按图元类型排序
        ai_Process_GenNormals |           // 如果模型没有法线则生成
        aiProcess_CalcTangentSpace        // 计算切线和副切线
    );

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        spdlog::error("FBX Loader: Failed to load scene '{}'. Error: {}", file_path, importer->GetErrorString());
        return nullptr;
    }

    spdlog::info("FBX Loader: Successfully loaded scene '{}'", file_path);
    
    // 存储 importer，它的所有权被转移到 map 中。
    // importer 析构时会自动释放其加载的 scene。
    g_importers[file_path] = std::move(importer);

    return scene;
}

} // namespace fbx_loader
