#pragma once

#include <string>
#include <memory>
#include <map>
#include <vector>
#include "scene/scene.h"
#include "photon/sources.h"
#include "photon/photon_batch.h"
#include "simulation/simulation_result.h"
#include "material.h"

/**
 * @brief 仿真运行配置参数
 */
struct SimConfig {
    int num_rays = 1'000'000;
    int max_bounces = 50;
    bool use_nee = false;
    unsigned int random_seed = 0;
};

/**
 * @brief 统一的仿真器接口
 */
class Simulator {
public:
    /**
     * @brief 构造函数
     * @param device_id CUDA 设备 ID (默认 0)
     */
    explicit Simulator(int device_id = 0);
    ~Simulator();

    /**
     * @brief 仿真配置参数（可直接修改）
     */
    SimConfig config;

    /**
     * @brief 添加材质到材质池
     * @param material 材质实例
     * @return 材质在池中的索引
     */
    size_t add_material(std::shared_ptr<Material> material);

    /**
     * @brief 设置指定索引的材质
     * @param index 材质索引
     * @param material 新的材质实例
     */
    void set_material(size_t index, std::shared_ptr<Material> material);

    /**
     * @brief 获取材质池大小
     */
    size_t get_material_pool_size() const;

    /**
     * @brief 获取指定索引的材质
     */
    std::shared_ptr<Material> get_material(size_t index) const;

    /**
     * @brief 清空材质池
     */
    void clear_materials();

    /**
     * @brief 从Scene构建GPU场景（使用材质池）
     *
     * @param scene CPU端场景数据
     * @param material_mapping mesh材质名称 -> material_pool索引的映射
     * @param flip_detector_normal 是否翻转探测器法向 (默认 false)
     *
     * @example
     * ```cpp
     * Simulator sim;
     *
     * // 设置材质池
     * sim.material_pool.push_back(material::lambertian(0.98));  // 0
     * sim.material_pool.push_back(material::detector());        // 1
     *
     * // 材质名称映射到池索引
     * std::map<std::string, size_t> mapping;
     * mapping["wall_left"] = 0;
     * mapping["wall_right"] = 0;
     * mapping["detector"] = 1;
     *
     * sim.build_scene(scene, mapping);
     * ```
     */
    void build_scene(
        const Scene& scene,
        const std::map<std::string, size_t>& material_mapping,
        bool flip_detector_normal = false
    );

    /**
     * @brief 更新材质到GPU（不重建几何结构）
     *
     * 在修改 material_pool 后调用此方法同步到 GPU
     *
     * @example
     * ```cpp
     * sim.material_pool[0] = material::lambertian(0.95);
     * sim.update_materials();  // 同步到 GPU
     * ```
     */
    void update_materials();

    /**
     * @brief 运行蒙特卡洛仿真
     */
    SimulationResult run(phonder::PhotonSource& procedural_source);
    SimulationResult run(const phonder::PhotonBatch& source_batch);

    /**
     * @brief 获取探测器总面积 (mm²)
     */
    float get_detector_total_area() const;

private:
    class Pimpl;
    std::unique_ptr<Pimpl> pimpl_;

    // 材质池（私有）
    std::vector<std::shared_ptr<Material>> material_pool_;
};
