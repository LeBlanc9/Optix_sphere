#pragma once

// 数学常量定义
// 这些常量在整个项目中使用（CPU和GPU代码）

#ifndef PI
#define PI 3.14159265358979323846f
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

// 其他常用常量
#define TWO_PI (2.0f * M_PIf)
#define INV_PI (1.0f / M_PIf)
#define INV_TWO_PI (1.0f / TWO_PI)
