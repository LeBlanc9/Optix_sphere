#pragma once

#include <cmath>
#include <vector>
#include "simulation/simulation_result.h"
#include "constants.h"

namespace theory {

/**
 * @brief 代表理论积分球上的一个端口。
 */
struct Port {
    float radius = 0.0f;        // 端口半径 (mm)
    float reflectance = 0.0f;   // 端口表面的反射率 [0, 1] (0代表开放的孔)
};

/**
 * @brief 代表一个理论积分球的模型 (TheoreticalIntegratingSphere)。
 *
 * 这个类用于构建一个包含主体和多个端口的经典积分球模型，
 * 并作为理论计算器(TheoryCalculator)的输入。
 */
class TheoreticalIntegratingSphere {
public:
    /**
     * @brief 构造一个理论积分球。
     * @param radius 球的内半径 (mm)。
     * @param wall_reflectance 球体内壁的反射率 [0, 1]。
     */
    TheoreticalIntegratingSphere(float radius, float wall_reflectance)
        : radius_(radius), wall_reflectance_(wall_reflectance) {}

    /**
     * @brief 向球体模型添加一个端口。
     * @param radius 端口的半径 (mm)。
     * @param reflectance 端口表面的反射率 (例如，0代表开放的孔)。
     */
    void add_port(float radius, float reflectance) {
        ports_.push_back({radius, reflectance});
    }

    // --- 用于计算的Getter方法 ---
    float get_radius() const { return radius_; }
    float get_wall_reflectance() const { return wall_reflectance_; }
    const std::vector<Port>& get_ports() const { return ports_; }

    /**
     * @brief 计算理想球体的总表面积 (4 * PI * r^2)。
     */
    double get_total_sphere_area() const {
        return 4.0 * PI * radius_ * radius_;
    }
    
    /**
     * @brief 计算包含所有端口在内的球壁的面积加权平均反射率。
     * 这是在理论计算中使用的“有效反射率”。
     */
    double get_effective_wall_reflectance() const {
        double total_area = get_total_sphere_area();
        if (total_area == 0) return 0;

        double wall_area = total_area;
        double weighted_reflectance_sum = 0.0;

        for (const auto& port : ports_) {
            double port_area = PI * port.radius * port.radius;
            wall_area -= port_area;
            weighted_reflectance_sum += port_area * port.reflectance;
        }

        weighted_reflectance_sum += wall_area * wall_reflectance_;

        return weighted_reflectance_sum / total_area;
    }

private:
    float radius_;
    float wall_reflectance_;
    std::vector<Port> ports_;
};


/**
 * @brief 为积分球模型执行理论计算。
 */
class TheoryCalculator {
public:
    /**
     * @brief 计算一个积分球模型的理论性能。
     *
     * 此计算基于Goebel公式，并为包含多个不同反射率端口的球体进行了适配。
     *
     * @param sphere 配置好的理论积分球模型。
     * @param incident_power 入射到球体内的总光功率 (W)。
     * @return 一个包含计算结果的TheoryResult结构体。
     */
    static TheoryResult calculate(
        const TheoreticalIntegratingSphere& sphere,
        float incident_power
    );
};

} // namespace theory