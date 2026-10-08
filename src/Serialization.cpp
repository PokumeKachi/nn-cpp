#include "Serialization.hpp"

#include <cstdio>
#include <fstream>

namespace neural_network_serializer {
bool save() {
    const std::string tmp = std::string(PATH_TO_SERIALIZED_BINARY.data()) + ".tmp";

    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;

        f.write(reinterpret_cast<const char*>(neural_network::weights.data()),
                static_cast<std::streamsize>(neural_network::weights.size()));
        if (!f) return false;

        f.write(reinterpret_cast<const char*>(neural_network::biases.data()),
                static_cast<std::streamsize>(neural_network::biases.size()));
        if (!f) return false;

        f.flush();
        if (!f) return false;
    }

    if (std::rename(tmp.c_str(), PATH_TO_SERIALIZED_BINARY.data()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }

    return true;
}

bool load() {
    std::ifstream f(PATH_TO_SERIALIZED_BINARY.data(), std::ios::binary);
    if (!f) return false;

    f.read(reinterpret_cast<char*>(neural_network::weights.data()),
           static_cast<std::streamsize>(neural_network::weights.size()));
    if (f.gcount() != static_cast<std::streamsize>(neural_network::weights.size())) return false;

    f.read(reinterpret_cast<char*>(neural_network::biases.data()),
           static_cast<std::streamsize>(neural_network::biases.size()));
    if (f.gcount() != static_cast<std::streamsize>(neural_network::biases.size())) return false;

    return true;
}

}  // namespace neural_network_serializer
