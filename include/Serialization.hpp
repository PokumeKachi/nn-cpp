#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "Model.hpp"

namespace neural_network_serializer {
constexpr std::array<char, 16> PATH_TO_SERIALIZED_BINARY = []() constexpr {
    std::uint64_t h = 1469598103934665603ull;

    auto mix = [&](std::uint64_t x) {
        h ^= x;
        h *= 1099511628211ull;
    };

    for (auto n : ARCHITECTURE::SIZE) mix(static_cast<std::uint64_t>(n));
    for (auto& a : ARCHITECTURE::ACTIVATION) mix(static_cast<std::uint64_t>(a.index()));

    constexpr char A[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    std::array<char, 16> out{};

    for (std::size_t i = 0; i < 11; ++i) {
        out[i] = A[h & 0x3F];
        h >>= 6;
    }

    std::copy_n(".bin", 4, out.begin() + 11);

    return out;
}();

bool save();
bool load();
}  // namespace neural_network_serializer
