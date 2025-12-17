# 积分球模拟器 - 架构设计

## 模块划分

```
┌─────────────────────────────────────────────┐
│           Main Application                  │
│            (main.cpp)                       │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
┌──────────────┐  ┌──────────────┐
│  Simulator   │  │   Theory     │
│   (CPU)      │  │ Calculator   │
└──────┬───────┘  └──────────────┘
       │
       ├─────────┬─────────┬──────────┐
       │         │         │          │
       ▼         ▼         ▼          ▼
  ┌─────────┐ ┌──────┐ ┌──────┐  ┌──────┐
  │ OptiX   │ │Geom  │ │Light │  │Detect│
  │ Context │ │etry  │ │Source│  │ or   │
  └─────────┘ └──────┘ └──────┘  └──────┘
       │
       ▼
  ┌──────────────┐
  │  GPU Kernels │
  │  (sphere.cu) │
  └──────────────┘
```

## 核心模块

### 1. OptiX Context Manager
**文件**: `optix_context.h/cpp`
**职责**:
- OptiX 环境初始化
- Module 加载（PTX）
- Pipeline 创建
- SBT 管理
- 资源清理

### 2. Geometry Module
**文件**: `geometry.h/cpp`
**职责**:
- 抽象几何接口
- 解析球面实现
- AABB 计算
- 加速结构构建

### 3. Light Source
**文件**: `light_source.h/cpp`
**职责**:
- 光源配置
- 发射方向采样
- 功率分配

### 4. Detector
**文件**: `detector.h/cpp`
**职责**:
- 探测器配置
- 通量收集
- 结果统计

### 5. Theory Calculator
**文件**: `theory.h/cpp`
**职责**:
- Goebel 公式计算
- 理论解
- 误差分析

### 6. Simulator
**文件**: `simulator.h/cpp`
**职责**:
- 协调各模块
- 运行模拟
- 结果输出

### 7. GPU Kernels
**文件**: `sphere.cu`
**职责**:
- 射线相交
- 材质散射
- 能量传递

## 数据流

1. **配置阶段**
   ```
   main → Simulator.configure()
        → Geometry, Light, Detector
   ```

2. **构建阶段**
   ```
   Simulator.build()
        → OptiXContext.createPipeline()
        → Geometry.buildAccelStructure()
   ```

3. **模拟阶段**
   ```
   Simulator.run()
        → OptiXContext.launch()
        → GPU Kernels execute
        → Detector.collect()
   ```

4. **验证阶段**
   ```
   Theory.calculate()
   Simulator.compare(theory_result)
   ```

## 接口设计

### Simulator 接口
```cpp
class Simulator {
public:
    void setGeometry(std::shared_ptr<Geometry> geom);
    void setLightSource(std::shared_ptr<LightSource> light);
    void setDetector(std::shared_ptr<Detector> detector);

    void build();
    SimulationResult run(int num_rays);
    void compare(const TheoryResult& theory);
};
```

### Geometry 接口
```cpp
class Geometry {
public:
    virtual AABB getBounds() const = 0;
    virtual OptixTraversableHandle buildAccelStructure() = 0;
    virtual void getSBTData(void* data) const = 0;
};

class AnalyticalSphere : public Geometry {
    float3 center;
    float radius;
    float reflectance;
};
```

## 配置示例

```cpp
// 创建模拟器
auto sim = std::make_shared<Simulator>();

// 配置几何
auto sphere = std::make_shared<AnalyticalSphere>(
    center: {0, 0, 0},
    radius: 0.05,  // 5cm
    reflectance: 0.99
);
sim->setGeometry(sphere);

// 配置光源
auto light = std::make_shared<IsotropicPointSource>(
    position: {0, 0, 0},
    power: 1.0  // 1W
);
sim->setLightSource(light);

// 配置探测器
auto detector = std::make_shared<AreaDetector>(
    position: {0.049, 0, 0},
    normal: {-1, 0, 0},
    area: 1e-6  // 1mm²
);
sim->setDetector(detector);

// 运行
sim->build();
auto result = sim->run(1000000);

// 对比理论
auto theory = TheoryCalculator::goebel(sphere, light, detector);
sim->compare(theory);
```
