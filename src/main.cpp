#define DEBUG(x) std::cout << (x) << std::endl;

#include <algorithm>
#include <array>
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
std::uniform_real_distribution<float> dist(0.2f, 1.0f);

constexpr auto TARGET_FUNCTION = [](float a, float b) { return float(a * a * 2.0 + b * 3.0); };

float getLoss() {
    float totalLoss = 0;
    int count = 0;
    for (int ii = -10; ii <= 10; ++ii) {
        for (int jj = -10; jj <= 10; ++jj) {
            float i = ii / 10.0f;
            float j = jj / 10.0f;
            auto outputs = propagation::forward({i, i * i, j});
            float t = TARGET_FUNCTION(i, j);
            float denom = (t != 0.0f) ? std::abs(t) : 1.0f;
            totalLoss += std::abs(outputs[0] - t) / denom;
            ++count;
        }
    }
    return totalLoss / float(count);
}

void train() {
    auto shi = dist(rng);
    for (float learning_rate = 0.1; learning_rate >= 0.000000001;
         learning_rate /= 6.0 * shi) {
        // for (int _ = 0; _ < 10; ++_) {
        for (int wuhbluh = 0; wuhbluh < 1000; ++wuhbluh) {
            for (float i = -1.0; i < 1.0; i += 0.1 * shi) {
                for (float j = -1.0; j < 1.0; j += 0.1 * shi) {
                    propagation::backward({i, i * i, j}, {TARGET_FUNCTION(i, j)}, learning_rate);
                }
            }
        }
        // }
        DEBUG(getLoss());

        DEBUG("ABOVE IS THE FINAL LOSS");
    }
    DEBUG("TRAINING FINISHED");
}

int main() {
    if (!neural_network_serializer::load()) {
        cerr << "SOMETHING WENT WRONG WHILE LOADING PARAMS\n";
        std::fill(neural_network::weights.begin(), neural_network::weights.end(), 0.1);
        std::fill(neural_network::biases.begin(), neural_network::biases.end(), 0.1);
    }

    // std::iota(neural_network::weights.begin(), neural_network::weights.end(), 1);
    // std::iota(neural_network::biases.begin(), neural_network::biases.end(), 1);
    // std::generate(neural_network::weights.begin(), neural_network::weights.end(),
    //               [&] { return dist(rng); });
    // std::generate(neural_network::biases.begin(), neural_network::biases.end(),
    //               [&] { return dist(rng); });

    train();
    float x = 0.12124398734;
    float y = -0.1;

    DEBUG(propagation::forward({x, x * x, y})[0]);
    DEBUG(TARGET_FUNCTION(x, y));

    if (!neural_network_serializer::save()) {
        cerr << "SOMETHING WENT WRONG WHILE SAVING PARAMS\n";
    }

    return 0;
}
