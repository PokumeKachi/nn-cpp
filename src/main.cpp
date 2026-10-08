#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "Model.hpp"
#include "Propagation.hpp"
#include "Serialization.hpp"

using std::array;
using std::cerr;
using std::cout;
using std::endl;
using std::fill;
using std::size_t;

#include <random>

std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

int main() {
    // if (!neural_network_serializer::load()) {
    // cerr << "SOMETHING WENT WRONG WHILE LOADING PARAMS\n";
    // }

    std::iota(neural_network::weights.begin(), neural_network::weights.end(), 1);
    std::iota(neural_network::biases.begin(), neural_network::biases.end(), 1);
    // std::generate(neural_network::weights.begin(), neural_network::weights.end(),
    //               [&] { return dist(rng); });
    // std::generate(neural_network::biases.begin(), neural_network::biases.end(),
    //               [&] { return dist(rng); });

    cout << "FORWARDING\n";

    const auto& outputs = propagation::forward({3.1, 5.3});

    for (const neural_network::PARAMETER_TYPE& output : outputs) {
        std::cout << output << ' ';
    }
    std::cout << "\nABOVE ARE THE OUTPUTS\n";

    if (!neural_network_serializer::save()) {
        cerr << "SOMETHING WENT WRONG WHILE SAVING PARAMS\n";
    }

    return 0;
}
