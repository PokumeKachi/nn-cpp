#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "Activation.hpp"

using std::accumulate;
using std::array;
using std::to_array;

namespace ARCHITECTURE {
inline constexpr size_t INPUTS = 2;
inline constexpr auto HIDDEN = to_array<size_t>({3, 2});
inline constexpr auto ACTIVATION =
    to_array<Activation::Tag>({Activation::Identity{}, Activation::Identity{}});
}  // namespace ARCHITECTURE

namespace COUNT {
// inline constexpr auto HIDDEN_LAYERS = ARCHITECTURE::HIDDEN.size();
// inline constexpr auto LAYERS = 1 + HIDDEN_LAYERS;
inline constexpr auto HIDDEN_NODES =
    accumulate(ARCHITECTURE::HIDDEN.begin(), ARCHITECTURE::HIDDEN.end(), size_t{0});
inline constexpr auto NODES = ARCHITECTURE::INPUTS + HIDDEN_NODES;
inline constexpr auto CONNECTIONS = [] {
    if constexpr (ARCHITECTURE::HIDDEN.empty()) {
        return size_t{0};
    }
    size_t count = ARCHITECTURE::INPUTS * ARCHITECTURE::HIDDEN.front();
    for (size_t i = 1; i < ARCHITECTURE::HIDDEN.size(); ++i) {
        count += ARCHITECTURE::HIDDEN[i - 1] * ARCHITECTURE::HIDDEN[i];
    }
    return count;
}();
}  // namespace COUNT

namespace neural_network {
using PARAMETER_TYPE = Activation::PARAMETER_TYPE;
inline array<PARAMETER_TYPE, COUNT::CONNECTIONS> weights{};
inline array<PARAMETER_TYPE, COUNT::HIDDEN_NODES> biases{};
}  // namespace neural_network
