# 数学常量管理架构

## 统一常量定义

所有数学常量现在统一定义在：
```
optix_sphere/core/math_constants.h
```

## 使用方法

### C++ 代码（推荐）
```cpp
#include "core/math_constants.h"

// 在 phonder 命名空间内
float circumference = 2.0f * phonder::pi * radius;

// 或者使用 using 声明
using phonder::pi;
using phonder::two_pi;
float angle = two_pi / 4.0f;
```

### CUDA 代码
```cuda
#include "utils/device/math.cuh"  // 已包含 core/math_constants.h

namespace phonder {
    __device__ void foo() {
        float angle = two_pi * random;  // 可直接使用
    }
}
```

## 可用常量

| 常量名 | 值 | 说明 |
|-------|-----|-----|
| `pi` | 3.14159... | π |
| `two_pi` | 2π | 完整圆周 |
| `half_pi` | π/2 | 直角 |
| `inv_pi` | 1/π | π 的倒数 |
| `epsilon` | 1e-6 | 浮点比较误差 |

## 命名规范

- ✅ 使用小写 + 下划线：`pi`, `two_pi`, `half_pi`
- ✅ 使用 `constexpr`（类型安全，无宏污染）
- ✅ 在 `phonder` 命名空间内
- ❌ 避免使用 `M_PI` 等容易冲突的名字
- ❌ 不要在其他地方重复定义常量

## 废弃文件

以下文件仅保留用于向后兼容，**不应在新代码中使用**：
- `optix_sphere/utils/constant.h` - 已重定向到 core/math_constants.h
- `optix_sphere/constants.h` - 提供宏兼容层（DEPRECATED）

## 跨平台兼容性

这个架构解决了以下问题：
- ✅ 避免与系统头文件的 `M_PI` 宏冲突（Linux/macOS）
- ✅ 单一真实来源（Single Source of Truth）
- ✅ 编译时类型检查
- ✅ 更好的 IDE 支持和代码提示
