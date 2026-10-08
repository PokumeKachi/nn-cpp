#pragma once

#include <array>

#include "Model.hpp"

using std::array;

using ARCHITECTURE::HIDDEN;
using ARCHITECTURE::INPUTS;
using neural_network::PARAMETER_TYPE;

namespace propagation {
array<PARAMETER_TYPE, HIDDEN.back()> forward(const array<PARAMETER_TYPE, INPUTS>& input);
PARAMETER_TYPE backward(const array<PARAMETER_TYPE, INPUTS>& input,
                        const array<PARAMETER_TYPE, HIDDEN.back()>& target);
}  // namespace propagation
