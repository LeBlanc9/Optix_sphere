#pragma once

#include <cmath>
#include "simulation/simulation_result.h"
#include "constants.h"


/**
 * 积分球理论计算器
 *
 * 基于 Goebel 公式计算理想积分球的辐照度分布
 */
class TheoryCalculator {
public:
    /**
     * 计算理想积分球的平均辐照度
     *
     * 公式：E_avg = (ρ * Φ) / (A_s * (1 - ρ))
     *
     * 其中：
     *   E_avg: 平均辐照度 (W/m²)
     *   ρ: 壁面反射率
     *   Φ: 入射功率 (W)
     *   A_s: 球面面积 (m²)
     *   (1-ρ): 考虑无限次反射的能量损失
     *
     * @param radius 球半径 (m)
     * @param reflectance 壁面反射率 [0, 1]
     * @param incident_power 入射光功率 (W)
     * @return TheoryResult 理论结果
     */
    static TheoryResult calculateIdealSphere(
        float radius,
        float reflectance,
        float incident_power
    );

    /**
     * 计算考虑端口损失的积分球辐照度
     *
     * 当积分球有端口时，端口会导致光线逃逸，需要修正
     *
     * @param radius 球半径
     * @param reflectance 壁面反射率
     * @param incident_power 入射功率
     * @param port_area 端口总面积
     * @return TheoryResult 修正后的理论结果
     */
    static TheoryResult calculateWithPorts(
        float radius,
        float reflectance,
        float incident_power,
        float port_area
    );

    /**
     * 计算相对误差
     *
     * @param simulation 模拟结果
     * @param theory 理论结果
     * @return 相对误差百分比 (double精度)
     */
    static double calculateRelativeError(
        const SimulationResult& simulation,
        const TheoryResult& theory
    );

    /**
     * 打印对比报告
     */
    static void printComparison(
        const SimulationResult& simulation,
        const TheoryResult& theory
    );
};
