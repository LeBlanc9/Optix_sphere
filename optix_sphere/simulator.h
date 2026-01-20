#pragma once

#include <string>
#include <memory>
#include <map>
#include "scene/scene.h"   // For Scene
#include "photon/sources.h"    // Data-only source definitions
#include "photon/photon_batch.h"      // For PhotonBatch
#include "simulation/simulation_result.h"
#include "material.h"          // For MaterialDescriptor


/**
 * @brief 仿真运行配置参数
 */
struct SimConfig {
    int num_rays = 1'000'000;   ///< 要追踪的光线数量
    int max_bounces = 50;       ///< 最大反弹次数
    bool use_nee = false;       ///< 是否启用Next Event Estimation (默认关闭)
    unsigned int random_seed = 0; ///< 随机数种子 (0 = 使用时间，非0 = 固定种子)
};


/**
 * @brief 统一的仿真器接口 (API v2).
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
     * @brief 从Scene构建GPU场景
     *
     * 将独立创建的Scene构建为GPU加速结构。
     *
     * @param scene CPU端场景数据
     * @param material_factories 材质名称到 MaterialFactory 的映射
     *
     * @example
     * ```cpp
     * // 创建CPU场景
     * Scene scene = Scene::from_obj("sphere.obj");
     *
     * // 准备材质工厂
     * std::map<std::string, MaterialFactory> materials;
     * materials["wall"] = material::lambertian(0.98);
     * materials["detector"] = material::detector();
     *
     * // 构建GPU场景
     * Simulator sim;
     * sim.build_scene(scene, materials);
     * ```
     */
    void build_scene(
        const Scene& scene,
        const std::map<std::string, MaterialFactory>& material_factories
    );


    // --- Simulation Execution ---

    /**
     * @brief 运行蒙特卡洛仿真。
     *        光子源通过数学描述在GPU上生成。
     * @param procedural_source 光子源的数学描述 (例如 IsotropicPointSource, CollimatedBeamSource等)。
     * @param config 通用的仿真运行配置 (光线数、反弹次数等)。
     * @return 仿真结果。
     */
    SimulationResult run(phonder::PhotonSource& procedural_source, const SimConfig& config);

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