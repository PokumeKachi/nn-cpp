#pragma once

#include <cstdint>
#include <variant>

namespace Activation {
struct Identity {
    static constexpr int8_t apply(int8_t x) noexcept { return x; }
    static constexpr int8_t derive(int8_t) noexcept { return 1; }
};

struct Relu {
    static constexpr int8_t apply(int8_t x) noexcept { return x < 0 ? 0 : x; }
    static constexpr int8_t derive(int8_t x) noexcept { return x > 0 ? 1 : 0; }
};

struct LeakyRelu {
    static constexpr int8_t apply(int8_t x) noexcept {
        return x >= 0 ? x : static_cast<int8_t>(x >> 3);
    }
    static constexpr int8_t derive(int8_t) noexcept { return 1; }
};

struct Relu6 {
    static constexpr int8_t apply(int8_t x) noexcept { return x < 0 ? 0 : (x > 6 ? 6 : x); }
    static constexpr int8_t derive(int8_t x) noexcept { return (x > 0 && x <= 6) ? 1 : 0; }
};

struct HardTanh {
    static constexpr int8_t apply(int8_t x) noexcept { return x < -1 ? -1 : (x > 1 ? 1 : x); }
    static constexpr int8_t derive(int8_t x) noexcept { return (x >= -1 && x <= 1) ? 1 : 0; }
};

struct BinaryStep {
    static constexpr int8_t apply(int8_t x) noexcept { return x >= 0 ? 1 : 0; }
    static constexpr int8_t derive(int8_t) noexcept { return 0; }
};

struct Signum {
    static constexpr int8_t apply(int8_t x) noexcept { return x > 0 ? 1 : (x < 0 ? -1 : 0); }
    static constexpr int8_t derive(int8_t) noexcept { return 0; }
};

struct Absolute {
    static constexpr int8_t apply(int8_t x) noexcept {
        return x == -128 ? 127 : (x < 0 ? static_cast<int8_t>(-x) : x);
    }
    static constexpr int8_t derive(int8_t x) noexcept { return x > 0 ? 1 : (x < 0 ? -1 : 0); }
};

struct Square {
    static constexpr int8_t apply(int8_t x) noexcept {
        const int32_t sq = static_cast<int32_t>(x) * static_cast<int32_t>(x);
        return sq > 127 ? 127 : static_cast<int8_t>(sq);
    }
    static constexpr int8_t derive(int8_t x) noexcept {
        return (x >= -11 && x <= 11) ? static_cast<int8_t>(2 * x) : 0;
    }
};

using Tag = std::variant<Activation::Identity, Activation::Relu, Activation::LeakyRelu,
                         Activation::Relu6, Activation::HardTanh, Activation::BinaryStep,
                         Activation::Signum, Activation::Absolute, Activation::Square>;

inline int8_t apply(const Tag& t, int8_t x) noexcept {
    return std::visit([x](auto tag) { return std::remove_cvref_t<decltype(tag)>::apply(x); }, t);
}

inline int8_t derive(const Tag& t, int8_t x) noexcept {
    return std::visit([x](auto tag) { return std::remove_cvref_t<decltype(tag)>::derive(x); }, t);
}
}  // namespace Activation
