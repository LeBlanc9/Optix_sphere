#pragma once

#include <string>
#include <memory>
#include "scene/scene_types.h" // Contains config structs
#include "photon/sources.h"    // Data-only source definitions
#include "simulation/simulation_result.h"


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
     */
    Simulator();

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


    // --- Simulation Execution ---

    /**
     * @brief 运行蒙特卡洛仿真。
     * @param source 光子源 (例如 IsotropicPointSource, CollimatedBeamSource等)。
     * @param config 通用的仿真运行配置 (光线数、反弹次数等)。
     * @return 仿真结果。
     */
    SimulationResult run(phonder::PhotonSource& source, const SimConfig& config);

    /**
     * @brief 获取当前场景中探测器的总面积 (mm²)。
     * @return 探测器面积。如果场景未构建则抛出异常。
     */
    float get_detector_total_area() const;

private:
    // 使用PIMPL模式（指向实现的指针）来隐藏内部实现细节，
    // 避免在头文件中暴露OptiX等底层库的头文件，降低编译依赖。
    class Pimpl;
    std::unique_ptr<Pimpl> pimpl_;
};
