#pragma once

#include <array>
#include <numeric>

#include "Model.hpp"

namespace {
namespace OFFSET {
inline constexpr auto LAYER = [] {
    std::array<std::size_t, ARCHITECTURE::SIZE.size() + 1> result{};
    std::partial_sum(ARCHITECTURE::SIZE.begin(), ARCHITECTURE::SIZE.end(), result.begin() + 1);
    return result;
}();

inline constexpr auto WEIGHT = [] {
    std::array<std::size_t, ARCHITECTURE::SIZE.size()> result{};

    for (std::size_t i = 1; i < ARCHITECTURE::SIZE.size(); ++i)
        result[i] = result[i - 1] + ARCHITECTURE::SIZE[i - 1] * ARCHITECTURE::SIZE[i];

    return result;
}();
}  // namespace OFFSET
}  // unnamed namespace

namespace propagation {
std::array<int8_t, ARCHITECTURE::SIZE.back()> forward(
    const std::array<int8_t, ARCHITECTURE::SIZE.front()>& input);
float backward(const std::array<float, ARCHITECTURE::SIZE.front()>& input,
               const std::array<float, ARCHITECTURE::SIZE.back()>& target);
}  // namespace propagation
