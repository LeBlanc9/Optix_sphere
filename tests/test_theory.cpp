#include <gtest/gtest.h>
#include "theory.h"
#include <cmath>

// 测试理想积分球的理论计算
TEST(TheoryCalculatorTest, IdealSphereBasic) {
    // 测试参数
    float radius = 0.05f;        // 50mm
    float reflectance = 0.99f;   // 99%
    float power = 1.0f;          // 1W

    auto result = TheoryCalculator::calculateIdealSphere(radius, reflectance, power);

    // 验证球面面积
    float expected_area = 4.0f * PI * radius * radius;
    EXPECT_FLOAT_EQ(result.sphere_area, expected_area);

    // 验证辐照度计算
    // E = (ρ * Φ) / (A * (1 - ρ))
    float expected_irradiance = (reflectance * power) / (expected_area * (1.0f - reflectance));
    EXPECT_FLOAT_EQ(result.avg_irradiance, expected_irradiance);

    // 验证总通量
    float expected_flux = power / (1.0f - reflectance);
    EXPECT_FLOAT_EQ(result.total_flux_in_sphere, expected_flux);
}

// 测试不同反射率的影响
TEST(TheoryCalculatorTest, ReflectanceEffect) {
    float radius = 0.05f;
    float power = 1.0f;

    // 低反射率
    auto result_low = TheoryCalculator::calculateIdealSphere(radius, 0.5f, power);

    // 高反射率
    auto result_high = TheoryCalculator::calculateIdealSphere(radius, 0.99f, power);

    // 高反射率应该导致更高的辐照度（因为多次反射）
    EXPECT_GT(result_high.avg_irradiance, result_low.avg_irradiance);
    EXPECT_GT(result_high.total_flux_in_sphere, result_low.total_flux_in_sphere);
}

// 测试球体大小的影响
TEST(TheoryCalculatorTest, SphereSize) {
    float reflectance = 0.99f;
    float power = 1.0f;

    // 小球
    auto result_small = TheoryCalculator::calculateIdealSphere(0.05f, reflectance, power);

    // 大球
    auto result_large = TheoryCalculator::calculateIdealSphere(0.1f, reflectance, power);

    // 大球表面积是小球的4倍（半径2倍）
    EXPECT_FLOAT_EQ(result_large.sphere_area / result_small.sphere_area, 4.0f);

    // 大球的辐照度应该更低（同样的功率分布在更大的面积上）
    EXPECT_LT(result_large.avg_irradiance, result_small.avg_irradiance);

    // 但总通量应该相同（只取决于反射率）
    EXPECT_FLOAT_EQ(result_large.total_flux_in_sphere, result_small.total_flux_in_sphere);
}

// 测试功率缩放
TEST(TheoryCalculatorTest, PowerScaling) {
    float radius = 0.05f;
    float reflectance = 0.99f;

    auto result_1w = TheoryCalculator::calculateIdealSphere(radius, reflectance, 1.0f);
    auto result_2w = TheoryCalculator::calculateIdealSphere(radius, reflectance, 2.0f);

    // 功率翻倍，辐照度和通量都应该翻倍
    EXPECT_FLOAT_EQ(result_2w.avg_irradiance / result_1w.avg_irradiance, 2.0f);
    EXPECT_FLOAT_EQ(result_2w.total_flux_in_sphere / result_1w.total_flux_in_sphere, 2.0f);
}

// 测试带端口的积分球
TEST(TheoryCalculatorTest, SphereWithPorts) {
    float radius = 0.05f;
    float reflectance = 0.99f;
    float power = 1.0f;

    // 无端口
    auto result_no_port = TheoryCalculator::calculateIdealSphere(radius, reflectance, power);

    // 有端口（10mm² = 0.00001 m²）
    float port_area = 0.00001f;
    auto result_with_port = TheoryCalculator::calculateWithPorts(
        radius, reflectance, power, port_area
    );

    // 端口会降低辐照度
    EXPECT_LT(result_with_port.avg_irradiance, result_no_port.avg_irradiance);

    // 端口会降低球内总通量
    EXPECT_LT(result_with_port.total_flux_in_sphere, result_no_port.total_flux_in_sphere);
}

// 测试极端情况：反射率接近1
TEST(TheoryCalculatorTest, HighReflectance) {
    float radius = 0.05f;
    float power = 1.0f;

    // 反射率 = 0.999
    auto result = TheoryCalculator::calculateIdealSphere(radius, 0.999f, power);

    // 辐照度应该非常高
    EXPECT_GT(result.avg_irradiance, 1000.0f);

    // 总通量也很高
    EXPECT_GT(result.total_flux_in_sphere, 1000.0f);
}

// 测试能量守恒
TEST(TheoryCalculatorTest, EnergyConservation) {
    float radius = 0.05f;
    float reflectance = 0.99f;
    float power = 1.0f;

    auto result = TheoryCalculator::calculateIdealSphere(radius, reflectance, power);

    // 球内总通量应该大于输入功率（因为多次反射）
    EXPECT_GT(result.total_flux_in_sphere, power);

    // 但是有限（不是无穷大）
    EXPECT_LT(result.total_flux_in_sphere, 10000.0f);  // 合理的上界
}

// 测试相对误差计算
TEST(TheoryCalculatorTest, RelativeError) {
    SimulationResult sim;
    sim.irradiance = 100.0f;

    TheoryResult theory;
    theory.avg_irradiance = 110.0f;

    float error = TheoryCalculator::calculateRelativeError(sim, theory);

    // 误差应该约为 9.09%
    EXPECT_NEAR(error, 9.09f, 0.1f);
}

// 测试零反射率（黑体）
TEST(TheoryCalculatorTest, ZeroReflectance) {
    float radius = 0.05f;
    float power = 1.0f;

    auto result = TheoryCalculator::calculateIdealSphere(radius, 0.0f, power);

    // 零反射率意味着零辐照度（光被完全吸收）
    EXPECT_FLOAT_EQ(result.avg_irradiance, 0.0f);

    // 总通量等于输入功率
    EXPECT_FLOAT_EQ(result.total_flux_in_sphere, power);
}

// 主函数（Google Test 会自动提供）
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
