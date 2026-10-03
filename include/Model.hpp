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

namespace COUNT {
inline constexpr auto LAYER = ARCHITECTURE::SIZE.size();

inline constexpr auto NODE =
    std::accumulate(ARCHITECTURE::SIZE.begin(), ARCHITECTURE::SIZE.end(), std::size_t{0});

inline constexpr auto WEIGHT = [] {
    std::size_t count = 0;

    for (std::size_t i = 1; i < ARCHITECTURE::SIZE.size(); ++i)
        count += ARCHITECTURE::SIZE[i - 1] * ARCHITECTURE::SIZE[i];

    return count;
}();

inline constexpr auto BIAS =
    std::accumulate(ARCHITECTURE::SIZE.begin() + 1, ARCHITECTURE::SIZE.end(), std::size_t{0});

}  // namespace COUNT

namespace neural_network {
inline std::array<std::int8_t, COUNT::WEIGHT> weights{};
inline std::array<std::int8_t, COUNT::BIAS> biases{};
}  // namespace neural_network
