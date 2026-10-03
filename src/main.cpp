#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

#include "Model.hpp"
#include "Serialization.hpp"

using std::array;
using std::cerr;
using std::cout;
using std::endl;
using std::fill;
using std::size_t;

// void make_nodes() {
//     fill(NODES.begin(), NODES.begin() + LAYER_SIZE[0], Node{});
//     // std::iota(WEIGHTS.begin(), WEIGHTS.end(), 5);
//
//     for (size_t i = 0; i < LAYER_SIZE[0]; ++i) {
//         NODES[i] = Node({}, {}, BIASES.begin() + i);
//     }
//
//     auto current_layer_index = LAYER_SIZE[0];
//     auto current_connection_index = 0;
//
//     for (size_t i = 1; i < LAYER_COUNT; ++i) {
//         for (size_t j = 0; j < LAYER_SIZE[i]; ++j) {
//             const auto& CONNECTIONS = LAYER_SIZE[i - 1];
//             NODES[current_layer_index + j] =
//                 Node{{ACTIVATIONS.data() + current_layer_index - CONNECTIONS, CONNECTIONS},
//                      {WEIGHTS.data() + current_connection_index, CONNECTIONS},
//                      BIASES.data() + current_layer_index + j};
//
//             current_connection_index += CONNECTIONS;
//         }
//
//         current_layer_index += LAYER_SIZE[i];
//     }
// }

int main() {
    // make_nodes();

    if (!neural_network_serializer::load()) {
        cerr << "SOMETHING WENT WRONG WHILE LOADING PARAMS\n";
    }

    auto current_index = LAYER_SIZE[0];

    // for (size_t i = 1; i < LAYER_COUNT; ++i) {
    //     for (size_t j = 0; j < NODE_LAYERS[i]; ++j, ++current_index) {
    //         cout << "NODE " << current_index + 1 << endl;
    //
    //         for (auto it = NODES[current_index].prev_activations.begin();
    //              it != NODES[current_index].prev_activations.end(); ++it) {
    //             cout << int(*it) << ' ';
    //         }
    //
    //         cout << "\nABOVE WAS THE PREVIOUS LAYER\n";
    //
    //         for (auto it = NODES[current_index].weights.begin();
    //              it != NODES[current_index].weights.end(); ++it) {
    //             cout << int(*it) << ' ';
    //         }
    //
    //         cout << "\nABOVE WERE THE CONNECTIONS\n";
    //
    //         cout << int(*NODES[current_index].bias);
    //
    //         cout << "\nABOVE WAS THE BIAS\n";
    //     }
    //
    //     cout << endl;
    // }

    if (!neural_network_serializer::save()) {
        cerr << "SOMETHING WENT WRONG WHILE SAVING PARAMS\n";
    }

    return 0;
}
