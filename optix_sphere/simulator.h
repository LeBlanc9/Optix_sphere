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
     * @brief 获取场景中注册的材质总数
     */
    size_t get_material_count() const;

    /**
     * @brief 通过名称获取材质实例
     * @param material_name 材质名称
     * @return 材质实例
     * @throws std::runtime_error 如果名称不存在
     */
    std::shared_ptr<Material> get_material(const std::string& material_name) const;

    /**
     * @brief 清空所有材质和映射关系
     */
    void clear_materials();

    /**
     * @brief 设置指定名称的材质参数
     * 
     * 注意：只能设置 build_scene 时已存在的材质名称。
     * 
     * @param material_name 材质名称
     * @param material 新的材质实例
     * @param sync_to_gpu 是否立即同步到 GPU (默认 false)
     * @throws std::runtime_error 如果 material_name 在场景中不存在
     */
    void set_material(const std::string& material_name, std::shared_ptr<Material> material, bool sync_to_gpu = false);

    /**
     * @brief 获取当前场景中所有的材质名称
     */
    std::vector<std::string> get_material_names() const;

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
     * @brief 从Scene构建GPU场景（直接传入材质对象）
     *
     * @param scene CPU端场景数据
     * @param materials mesh材质名称 -> Material对象的映射
     * @param flip_detector_normal 是否翻转探测器法向 (默认 false)
     *
     * @example
     * ```cpp
     * Simulator sim;
     *
     * // 直接传入材质字典
     * std::map<std::string, std::shared_ptr<Material>> materials;
     * materials["wall"] = material::lambertian(0.98);
     * materials["detector"] = material::detector();
     *
     * sim.build_scene(scene, materials);
     *
     * // 获取材质索引用于后续更新
     * size_t wall_idx = sim.get_material_index("wall");
     * sim.set_material(wall_idx, material::lambertian(0.99));
     * sim.update_materials();
     * ```
     */
    void build_scene(
        const Scene& scene,
        const std::map<std::string, std::shared_ptr<Material>>& materials,
        bool flip_detector_normal = false
    );

    /**
     * @brief 根据材质名称获取材质池索引
     * @param material_name 材质名称
     * @return 材质在池中的索引
     * @throws std::runtime_error 如果材质名称不存在
     */
    size_t get_material_index(const std::string& material_name) const;

    /**
     * @brief 获取所有材质名称到索引的映射
     * @return 材质名称 -> 池索引的映射
     */
    std::map<std::string, size_t> get_material_indices() const;

    /**
     * @brief 更新材质到GPU（不重建几何结构）
     *
     * 在修改材质后调用此方法同步到 GPU
     *
     * @example
     * ```cpp
     * sim.set_material("wall", material::lambertian(0.95));
     * sim.sync_to_gpu();  // 同步到 GPU
     * ```
     */
    void sync_to_gpu();

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

    // 材质名称到索引的映射（用于 get_material_index）
    std::map<std::string, size_t> material_name_to_index_;
};
