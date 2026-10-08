#pragma once

#include <cmath>
#include <type_traits>
#include <variant>

namespace Activation {

using PARAMETER_TYPE = float;

struct Identity {
    static constexpr PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept { return x; }

    static constexpr PARAMETER_TYPE derive(PARAMETER_TYPE) noexcept { return 1.0f; }
};

struct Relu {
    static constexpr PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept { return x < 0.0f ? 0.0f : x; }

    static constexpr PARAMETER_TYPE derive(PARAMETER_TYPE x) noexcept {
        return x > 0.0f ? 1.0f : 0.0f;
    }
};

struct LeakyRelu {
    static constexpr PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept {
        return x >= 0.0f ? x : x * 0.125f;
    }

    static constexpr PARAMETER_TYPE derive(PARAMETER_TYPE x) noexcept {
        return x >= 0.0f ? 1.0f : 0.125f;
    }
};

struct Relu6 {
    static constexpr PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept {
        return x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x);
    }

    static constexpr PARAMETER_TYPE derive(PARAMETER_TYPE x) noexcept {
        return (x > 0.0f && x < 6.0f) ? 1.0f : 0.0f;
    }
};

struct HardTanh {
    static constexpr PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept {
        return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
    }

    static constexpr PARAMETER_TYPE derive(PARAMETER_TYPE x) noexcept {
        return (x > -1.0f && x < 1.0f) ? 1.0f : 0.0f;
    }
};

struct BinaryStep {
    static constexpr PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept {
        return x >= 0.0f ? 1.0f : 0.0f;
    }

    static constexpr PARAMETER_TYPE derive(PARAMETER_TYPE) noexcept { return 0.0f; }
};

struct Signum {
    static constexpr PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept {
        return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    }

    static constexpr PARAMETER_TYPE derive(PARAMETER_TYPE) noexcept { return 0.0f; }
};

struct Absolute {
    static constexpr PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept { return x < 0.0f ? -x : x; }

    static constexpr PARAMETER_TYPE derive(PARAMETER_TYPE x) noexcept {
        return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
    }
};

struct Square {
    static constexpr PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept {
        const PARAMETER_TYPE square = x * x;
        return square > 127.0f ? 127.0f : square;
    }

    static constexpr PARAMETER_TYPE derive(PARAMETER_TYPE x) noexcept {
        return x * x <= 127.0f ? 2.0f * x : 0.0f;
    }
};

struct Sigmoid {
    static PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept { return 1.0f / (1.0f + std::exp(-x)); }

    static PARAMETER_TYPE derive(PARAMETER_TYPE x) noexcept {
        const PARAMETER_TYPE y = apply(x);
        return y * (1.0f - y);
    }
};

struct Tanh {
    static PARAMETER_TYPE apply(PARAMETER_TYPE x) noexcept { return std::tanh(x); }

    static PARAMETER_TYPE derive(PARAMETER_TYPE x) noexcept {
        const PARAMETER_TYPE y = apply(x);
        return 1.0f - y * y;
    }
};

using Tag =
    std::variant<Identity, Relu, LeakyRelu, Relu6, HardTanh, BinaryStep, Signum, Absolute, Square>;

inline PARAMETER_TYPE apply(const Tag& tag, PARAMETER_TYPE x) noexcept {
    return std::visit(
        [x](const auto& activation) {
            using T = std::remove_cvref_t<decltype(activation)>;
            return T::apply(x);
        },
        tag);
}

inline PARAMETER_TYPE derive(const Tag& tag, PARAMETER_TYPE x) noexcept {
    return std::visit(
        [x](const auto& activation) {
            using T = std::remove_cvref_t<decltype(activation)>;
            return T::derive(x);
        },
        tag);
}

}  // namespace Activation
