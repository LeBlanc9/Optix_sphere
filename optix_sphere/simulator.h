#pragma once

#include <string>
#include <memory>
#include <map>
#include "scene/scene_types.h" // Contains config structs
#include "photon/sources.h"    // Data-only source definitions
#include "photon/photon_batch.h"      // For PhotonBatch
#include "photon/launchers.h"  // For generate_photons_on_device in implementation
#include "simulation/simulation_result.h"
#include "material.h"          // For MaterialDescriptor


/**
 * @brief 统一的高级仿真器接口 (API v2).
 *
 * 封装了OptiX上下文、场景和路径追踪器。
 * 支持从文件加载场景或程序化创建理想场景。
 */
class Simulator {
public:
    /**
     * @brief 构造函数，初始化仿真器核心组件。
     * @param device_id CUDA 设备 ID (默认 0)。
     */
    explicit Simulator(int device_id = 0);

    /**
     * @brief 析构函数。
     */
    ~Simulator();

    // --- Scene Building Methods ---

    /**
     * @brief 从.obj文件构建一个基于网格的场景。
     * @param file_path .obj文件的路径。
     * @param config 场景的物理和材质配置。
     */
    void build_scene_from_file(const std::string& file_path, const MeshSceneConfig& config);

    /**
     * @brief 从.obj文件构建一个基于网格的场景，使用自定义材质工厂。
     * @param file_path .obj文件的路径。
     * @param material_factories 材质名称到 MaterialFactory 的映射（OBJ 材质名 -> 材质工厂函数）。
     * @param config 场景的物理配置。
     *
     * @example
     * ```cpp
     * // 创建自定义材质
     * using namespace material;
     * std::map<std::string, MaterialFactory> materials;
     *
     * materials["wall_material"] = mixed(0.7, 0.3, 0.98);
     * materials["detector_material"] = detector();
     *
     * // 对于球面材质，需要指定球心
     * materials["sphere_wall"] = spherical_mixed(0.7, 0.3, 0.98, make_float3(0, 0, 0));
     *
     * // 使用自定义材质构建场景
     * MeshSceneConfig config;
     * simulator.build_scene_from_file(mesh_path, materials, config);
     * ```
     */
    void build_scene_from_file(
        const std::string& file_path,
        const std::map<std::string, MaterialFactory>& material_factories,
        const MeshSceneConfig& config
    );


    // --- Simulation Execution ---

    /**
     * @brief 运行蒙特卡洛仿真。
     *        光子源通过数学描述在GPU上生成。
     * @param procedural_source 光子源的数学描述 (例如 IsotropicPointSource, CollimatedBeamSource等)。
     * @param config 通用的仿真运行配置 (光线数、反弹次数等)。
     * @return 仿真结果。
     */
    SimulationResult run(const phonder::PhotonSource& procedural_source, const SimConfig& config);

    /**
     * @brief 运行蒙特卡洛仿真。
     *        使用一个预先在GPU上生成的光子批次作为输入。
     * @param source_batch 预先在GPU上的光子批次。
     * @param config 通用的仿真运行配置 (光线数、反弹次数等)。
     * @return 仿真结果。
     */
    SimulationResult run(const phonder::PhotonBatch& source_batch, const SimConfig& config);


    /**
     * @brief 获取当前场景中探测器的总面积 (mm²)。
     * @return 探测器面积。如果场景未构建则抛出异常。
     */
    float get_detector_total_area() const;

    /**
     * @brief 更新单个材质参数（不重建几何结构）
     *
     * 快速操作，仅更新材质定义。下次调用 run() 时会自动使用新材质。
     * 不会重建耗时的 BVH 几何加速结构。
     *
     * @param name 要更新的材质名称（必须在场景中已存在）
     * @param factory MaterialFactory 函数，用于创建新的材质实例
     * @throws std::runtime_error 如果材质名称未找到或场景未构建
     *
     * @example
     * ```cpp
     * // 更新墙面材质的反射率
     * simulator.update_material("wall_material", material::lambertian(0.95));
     * simulator.run(source, config);  // 自动使用新材质
     * ```
     */
    void update_material(const std::string& name, const MaterialFactory& factory);

private:
    class Pimpl;
    std::unique_ptr<Pimpl> pimpl_;
};