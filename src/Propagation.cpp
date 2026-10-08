#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>

#include "Model.hpp"
#include "Propagation.hpp"

using std::array;
using std::copy_n;
using std::partial_sum;
using std::size;

using ARCHITECTURE::ACTIVATION;
using ARCHITECTURE::HIDDEN;
using ARCHITECTURE::INPUTS;
using neural_network::biases;
using neural_network::PARAMETER_TYPE;
using neural_network::weights;

namespace offset {
constexpr auto layer = [] {
    array<size_t, HIDDEN.size() + 1> result{};
    partial_sum(HIDDEN.begin(), HIDDEN.end(), result.begin() + 1);
    return result;
}();

constexpr auto weight = [] {
    array<size_t, HIDDEN.size() + 1> result{};

    result[1] = INPUTS * HIDDEN[0];

    for (size_t i = 2; i <= HIDDEN.size(); ++i) {
        result[i] = result[i - 1] + HIDDEN[i - 2] * HIDDEN[i - 1];
    }

    return result;
}();
}  // namespace offset

PARAMETER_TYPE& weightRef(size_t layer, size_t index, size_t prev_index) {
    const size_t previous_size = layer ? HIDDEN[layer - 1] : INPUTS;
    const size_t position = offset::weight[layer] + index * previous_size + prev_index;

    std::cout << "WEIGHT IS AT " << position << '\n';

    return weights[position];
}

PARAMETER_TYPE& biasRef(size_t layer, size_t index) {
    const size_t position = offset::layer[layer] + index;

    std::cout << "BIAS IS AT " << position << '\n';

    return biases[position];
}

namespace propagation {
array<PARAMETER_TYPE, HIDDEN.back()> forward(const array<PARAMETER_TYPE, INPUTS>& input) {
    constexpr size_t MAX_SIZE = std::ranges::max(HIDDEN);

    array<PARAMETER_TYPE, MAX_SIZE> buffer1;
    array<PARAMETER_TYPE, MAX_SIZE> buffer2;
    auto* previous = buffer1.data();
    auto* current = buffer2.data();
    copy_n(input.data(), input.size(), previous);

    const auto* weight = weights.data();
    const auto* bias = biases.data();

    for (size_t layer = 0; layer < size(HIDDEN); ++layer) {
        const size_t layer_size = HIDDEN[layer];
        const size_t previous_size = layer > 0 ? HIDDEN[layer - 1] : INPUTS;

        for (size_t index = 0; index < layer_size; ++index) {
            auto& activation = current[index];
            const auto* previous_activation = previous;

            activation = *bias++;
            for (size_t prev_index = 0; prev_index < previous_size; ++prev_index) {
                activation += *previous_activation++ * *weight++;
            }
            activation = Activation::apply(ACTIVATION[layer], activation);
        }

        std::swap(previous, current);
    }

    array<PARAMETER_TYPE, HIDDEN.back()> output{};
    copy_n(previous, output.size(), output.data());
    return output;
}

PARAMETER_TYPE backward(const array<PARAMETER_TYPE, INPUTS>& input,
                        const array<PARAMETER_TYPE, HIDDEN.back()>& target) {
    // std::array<PARAMETER_TYPE, offset::layer.back() - offset>
    // 819.6 1113.1

    return 0.0f;
}
}  // namespace propagation
