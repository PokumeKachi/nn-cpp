#include "Node.hpp"

Node::Node(std::span<int8_t> p, std::span<int8_t> w, int8_t* b)
    : prev_activations(p), weights(w), bias(b) {}

int8_t Node::calculate_activation(int8_t (*fn)(int8_t) noexcept) const noexcept {
    int32_t sum = *bias;
    for (size_t i = 0; i < weights.size(); ++i)
        sum += static_cast<int32_t>(weights[i]) * static_cast<int32_t>(prev_activations[i]);
    if (sum < 0) sum = 0;
    if (sum > 127) sum = 127;
    if (sum < -128) sum = -128;
    return static_cast<int8_t>(sum);
}

size_t Node::get_incoming_connections() const { return prev_activations.size(); }
