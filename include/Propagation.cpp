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
                        const array<PARAMETER_TYPE, HIDDEN.back()>& target,
                        PARAMETER_TYPE learning_rate) {
    // std::array<PARAMETER_TYPE, offset::layer.back() - offset>
    // 819.6 1113.1

    constexpr size_t LAYER_COUNT = HIDDEN.size();
    constexpr size_t NEURON_COUNT = offset::layer.back();
    constexpr size_t MAX_SIZE = std::max(INPUTS, std::ranges::max(HIDDEN));

    // Cache activations and pre-activations for the forward pass.
    array<PARAMETER_TYPE, NEURON_COUNT> activations{};
    array<PARAMETER_TYPE, NEURON_COUNT> pre_activations{};

    for (size_t layer = 0; layer < LAYER_COUNT; ++layer) {
        const size_t layer_offset = offset::layer[layer];
        const size_t previous_size = layer == 0 ? INPUTS : HIDDEN[layer - 1];

        const PARAMETER_TYPE* previous =
            layer == 0 ? input.data() : activations.data() + offset::layer[layer - 1];

        for (size_t index = 0; index < HIDDEN[layer]; ++index) {
            const size_t neuron = layer_offset + index;

            PARAMETER_TYPE z = biases[neuron];

            const size_t weight_offset = offset::weight[layer] + index * previous_size;

            for (size_t j = 0; j < previous_size; ++j) {
                z += weights[weight_offset + j] * previous[j];
            }

            pre_activations[neuron] = z;
            activations[neuron] = Activation::apply(ACTIVATION[layer], z);
        }
    }

    // Loss: 1/2 * sum((output - target)^2).
    array<PARAMETER_TYPE, NEURON_COUNT> grad_activations{};

    const size_t output_offset = offset::layer[LAYER_COUNT - 1];
    PARAMETER_TYPE loss = 0.0f;

    for (size_t i = 0; i < HIDDEN.back(); ++i) {
        const size_t neuron = output_offset + i;
        const PARAMETER_TYPE error = activations[neuron] - target[i];

        loss += 0.5f * error * error;
        grad_activations[neuron] = error;
    }

    // Accumulate gradients without modifying parameters.
    // offset::weight.back() is the total number of weights.
    array<PARAMETER_TYPE, offset::weight.back()> grad_weights{};
    array<PARAMETER_TYPE, NEURON_COUNT> grad_biases{};

    // Reverse layer traversal.
    for (size_t reverse = LAYER_COUNT; reverse-- > 0;) {
        const size_t layer = reverse;
        const size_t layer_offset = offset::layer[layer];

        const size_t previous_size = layer == 0 ? INPUTS : HIDDEN[layer - 1];

        const PARAMETER_TYPE* previous =
            layer == 0 ? input.data() : activations.data() + offset::layer[layer - 1];

        array<PARAMETER_TYPE, MAX_SIZE> grad_previous{};

        for (size_t index = 0; index < HIDDEN[layer]; ++index) {
            const size_t neuron = layer_offset + index;

            // dL/dz = dL/da * da/dz.
            const PARAMETER_TYPE delta =
                grad_activations[neuron] *
                Activation::derive(ACTIVATION[layer], pre_activations[neuron]);

            grad_biases[neuron] = delta;

            const size_t weight_offset = offset::weight[layer] + index * previous_size;

            for (size_t j = 0; j < previous_size; ++j) {
                const size_t position = weight_offset + j;

                // dL/dw = delta * previous activation.
                grad_weights[position] = delta * previous[j];

                // dL/d(previous activation) += delta * weight.
                grad_previous[j] += weights[position] * delta;
            }
        }

        // Supply gradients to the preceding layer.
        if (layer > 0) {
            copy_n(grad_previous.data(), previous_size,
                   grad_activations.data() + offset::layer[layer - 1]);
        }
    }

    // Update only after the entire backward pass is complete.
    for (size_t i = 0; i < grad_weights.size(); ++i) {
        weights[i] -= learning_rate * grad_weights[i];
    }

    for (size_t i = 0; i < grad_biases.size(); ++i) {
        biases[i] -= learning_rate * grad_biases[i];
    }

    return loss;
}
}  // namespace propagation
