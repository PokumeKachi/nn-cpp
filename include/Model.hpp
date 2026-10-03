#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "Activation.hpp"

namespace ARCHITECTURE {

inline constexpr auto SIZE = std::to_array<std::size_t>({5, 3, 5});

inline constexpr auto ACTIVATION = std::to_array<Activation::Tag>(
    {Activation::LeakyRelu{}, Activation::LeakyRelu{}, Activation::Identity{}});

}  // namespace ARCHITECTURE

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

namespace COUNT {

inline constexpr auto LAYER = ARCHITECTURE::SIZE.size();
inline constexpr auto NODE = OFFSET::LAYER.back();
inline constexpr auto WEIGHT = OFFSET::WEIGHT.back();
inline constexpr auto BIAS = NODE - ARCHITECTURE::SIZE.front();

}  // namespace COUNT

namespace neural_network {

inline std::array<std::int8_t, COUNT::WEIGHT> weights{};
inline std::array<std::int8_t, COUNT::BIAS> biases{};

}  // namespace neural_network
