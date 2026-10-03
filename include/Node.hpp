#pragma once

#include <cstdint>
#include <span>

class Node {
public:
    explicit Node(std::span<int8_t> p = {}, std::span<int8_t> w = {}, int8_t* b = &Node::sentinel);

    int8_t calculate_activation(int8_t (*fn)(int8_t) noexcept) const noexcept;
    size_t get_incoming_connections() const;

private:
    static inline int8_t sentinel{};
    std::span<const int8_t> prev_activations;
    std::span<const int8_t> weights;

    int8_t* bias{};
    size_t index{};
    int8_t (*activation_function)(int8_t) noexcept;
    int8_t (*activation_function_derivative)(int8_t) noexcept;
};
