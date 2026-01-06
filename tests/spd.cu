#include <spdlog/spdlog.h>

int main() {
    spdlog::info("SPD log level test - Info level");
    spdlog::warn("SPD log level test - Warn level");
    spdlog::error("SPD log level test - Error level");
    return 0;
}