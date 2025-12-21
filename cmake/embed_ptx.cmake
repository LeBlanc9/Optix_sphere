# 将 PTX 文件嵌入为 C++ 头文件

if(NOT PTX_FILE)
    message(FATAL_ERROR "PTX_FILE not specified")
endif()

if(NOT OUTPUT_FILE)
    message(FATAL_ERROR "OUTPUT_FILE not specified")
endif()

# 读取 PTX 文件内容为文本
file(READ "${PTX_FILE}" ptx_content)

# 获取文件大小
file(SIZE "${PTX_FILE}" ptx_size)

# 转义特殊字符
string(REPLACE "\\" "\\\\" ptx_content "${ptx_content}")
string(REPLACE "\"" "\\\"" ptx_content "${ptx_content}")
string(REPLACE "\n" "\\n\"\n\"" ptx_content "${ptx_content}")

# 生成 C++ 头文件（使用分段字符串）
file(WRITE "${OUTPUT_FILE}"
"// Auto-generated file - DO NOT EDIT
// Generated from: ${PTX_FILE}
// Size: ${ptx_size} bytes

#pragma once

#include <cstddef>

namespace embedded {

// Embedded PTX code from kernels (split into lines to avoid MSVC limits)
static const char g_forward_tracer_ptx[] =
\"${ptx_content}\";

static const size_t g_forward_tracer_ptx_size = sizeof(g_forward_tracer_ptx) - 1;

} // namespace embedded
")

message(STATUS "Embedded PTX into: ${OUTPUT_FILE} (${ptx_size} bytes)")
