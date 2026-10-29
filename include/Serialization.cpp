
#include <cstdio>
#include <fstream>
#include <string>

#include "Serialization.hpp"

namespace neural_network_serializer {

bool save() {
    const std::string path = PATH_TO_SERIALIZED_BINARY.data();
    const std::string tmp = path + ".tmp";

    bool ok = false;

    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;

        f.write(reinterpret_cast<const char*>(neural_network::weights.data()),
                static_cast<std::streamsize>(neural_network::weights.size() *
                                             sizeof(neural_network::weights[0])));

        if (f) {
            f.write(reinterpret_cast<const char*>(neural_network::biases.data()),
                    static_cast<std::streamsize>(neural_network::biases.size() *
                                                 sizeof(neural_network::biases[0])));
        }

        if (f) f.flush();
        ok = static_cast<bool>(f);

        f.close();
        ok = ok && static_cast<bool>(f);
    }

    if (!ok) {
        std::remove(tmp.c_str());
        return false;
    }

    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }

    return true;
}

bool load() {
    std::ifstream f(PATH_TO_SERIALIZED_BINARY.data(), std::ios::binary);

    if (!f) return false;

    decltype(neural_network::weights) loaded_weights{};
    decltype(neural_network::biases) loaded_biases{};

    constexpr auto weight_bytes = static_cast<std::streamsize>(sizeof(neural_network::weights));

    constexpr auto bias_bytes = static_cast<std::streamsize>(sizeof(neural_network::biases));

    f.read(reinterpret_cast<char*>(loaded_weights.data()), weight_bytes);

    if (f.gcount() != weight_bytes) return false;

    f.read(reinterpret_cast<char*>(loaded_biases.data()), bias_bytes);

    if (f.gcount() != bias_bytes) return false;

    char extra;
    if (f.get(extra)) return false;
    if (!f.eof()) return false;

    neural_network::weights = loaded_weights;
    neural_network::biases = loaded_biases;

    return true;
}

}  // namespace neural_network_serializer
