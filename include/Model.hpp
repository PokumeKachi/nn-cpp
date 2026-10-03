#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <numeric>

#include "_Activation.hpp"

inline constexpr auto LAYER_SIZE = std::to_array<std::size_t>({5, 3, 5});
inline constexpr auto LAYER_ACTIVATION_TYPE = std::to_array<Activation::Tag>(
    {Activation::LeakyRelu{}, Activation::LeakyRelu{}, Activation::Identity{}});

namespace OFFSET {

inline constexpr auto LAYER = [] {
    std::array<std::size_t, LAYER_SIZE.size() + 1> result{};
    std::partial_sum(LAYER_SIZE.begin(), LAYER_SIZE.end(), result.begin() + 1);
    return result;
}();
inline constexpr auto WEIGHT = [] {
    std::array<std::size_t, LAYER_SIZE.size()> result{};
    for (std::size_t i = 1; i < LAYER_SIZE.size(); ++i)
        result[i] = result[i - 1] + LAYER_SIZE[i - 1] * LAYER_SIZE[i];
    return result;
}();
}  // namespace OFFSET

namespace COUNT {
inline constexpr auto LAYER = LAYER_SIZE.size();
inline constexpr auto NODE = OFFSET::LAYER.back();
inline constexpr auto WEIGHT = OFFSET::WEIGHT.back();
inline constexpr auto BIAS = NODE - LAYER_SIZE.front();
}  // namespace COUNT

namespace neural_network {
inline std::array<int8_t, COUNT::WEIGHT> weights{};
inline std::array<int8_t, COUNT::BIAS> biases{};
}  // namespace neural_network
