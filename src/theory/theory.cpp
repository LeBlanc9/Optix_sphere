#include "theory.h"
#include "constants.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

// 单位约定: radius in mm, power in W, result irradiance in W/mm²
TheoryResult TheoryCalculator::calculateIdealSphere(
    float radius,
    float reflectance,
    float incident_power
) {
    TheoryResult result;

    // 球面面积：A = 4πr² (mm²)
    result.sphere_area = 4.0 * PI * radius * radius;

    // 平均辐照度（Goebel 公式） (W/mm²)
    // E = (ρ * Φ) / (A * (1 - ρ))
    //
    // 推导：
    // 第一次入射：Φ₀ = Φ
    // 第一次反射：Φ₁ = ρ * Φ₀
    // 第二次反射：Φ₂ = ρ * Φ₁ = ρ² * Φ₀
    // ...
    // 总通量：Φ_total = Φ₀ * (1 + ρ + ρ² + ...) = Φ₀ / (1 - ρ)
    // 但每次反射后，通量分布在整个球面上
    // 所以平均辐照度：E = (ρ * Φ) / (A * (1 - ρ))

    if (reflectance >= 1.0) {
        std::cerr << "Warning: reflectance >= 1.0 leads to infinite irradiance" << std::endl;
        result.avg_irradiance = INFINITY;
    } else {
        result.avg_irradiance = (reflectance * incident_power) /
                                (result.sphere_area * (1.0 - reflectance));
    }

    // 球内总通量（考虑多次反射）
    result.total_flux_in_sphere = incident_power / (1.0 - reflectance);

    return result;
}

// 单位约定: radius in mm, port_area in mm², power in W, result in W/mm²
TheoryResult TheoryCalculator::calculateWithPorts(
    float radius,
    float reflectance,
    float incident_power,
    float port_area
) {
    TheoryResult result;

    result.sphere_area = 4.0 * PI * radius * radius;  // mm²

    // 等效反射率（考虑端口损失）
    // ρ_eff = ρ * (1 - A_port / A_sphere)
    // 因为端口区域不反射光
    double port_ratio = port_area / result.sphere_area;
    double effective_reflectance = reflectance * (1.0 - port_ratio);

    if (effective_reflectance >= 1.0) {
        result.avg_irradiance = INFINITY;
        result.detected_flux = INFINITY;
    } else {
        // 总平均辐照度: E = P_incident / (A_sphere * (1 - ρ_eff)) [W/mm²]
        result.avg_irradiance = incident_power /
                                (result.sphere_area * (1.0 - effective_reflectance));
        result.detected_flux = result.avg_irradiance * port_area;  // W
    }

    result.total_flux_in_sphere = incident_power / (1.0 - effective_reflectance);  // W

    return result;
}

double TheoryCalculator::calculateRelativeError(
    const SimulationResult& simulation,
    const TheoryResult& theory
) {
    if (theory.avg_irradiance == 0.0) {
        return INFINITY;
    }

    double abs_error = std::abs(simulation.irradiance - theory.avg_irradiance);
    return (abs_error / theory.avg_irradiance) * 100.0;
}

void TheoryCalculator::printComparison(
    const SimulationResult& simulation,
    const TheoryResult& theory
) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║          Simulation vs Theory Comparison              ║\n";
    std::cout << "╠════════════════════════════════════════════════════════╣\n";

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "║ Metric                    │ Simulation  │ Theory      ║\n";
    std::cout << "╟───────────────────────────┼─────────────┼─────────────╢\n";

    std::cout << "║ Irradiance (W/mm²)        │ "
              << std::setw(11) << simulation.irradiance << " │ "
              << std::setw(11) << theory.avg_irradiance << " ║\n";

    std::cout << "║ Detected flux (W)         │ "
              << std::setw(11) << simulation.detected_flux << " │ "
              << std::setw(11) << theory.detected_flux << " ║\n";

    std::cout << "╟───────────────────────────┴─────────────┴─────────────╢\n";

    float error = calculateRelativeError(simulation, theory);
    std::cout << "║ Relative error: " << std::setprecision(3) << error << " %";

    // 填充空格使对齐
    int spaces = 38 - (int)std::to_string((int)error).length();
    for (int i = 0; i < spaces; i++) std::cout << " ";
    std::cout << "║\n";

    std::cout << "╟────────────────────────────────────────────────────────╢\n";
    std::cout << "║ Simulation stats:                                      ║\n";
    std::cout << "║   Total rays: " << std::setw(10) << simulation.total_rays
              << "                                  ║\n";
    std::cout << "║   Detected rays: " << std::setw(10) << simulation.detected_rays
              << "                               ║\n";
    std::cout << "║   Avg bounces: " << std::setprecision(2) << std::setw(10)
              << simulation.avg_bounces << "                                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    std::cout << std::endl;
}
