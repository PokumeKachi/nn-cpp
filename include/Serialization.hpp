#include <cstdint>
#include <fstream>

#include "Model.hpp"

constexpr std::array<char, 12> MODEL_FILE_PATH = []() constexpr {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t x) {
        h ^= x;
        h *= 1099511628211ull;
    };
    for (auto n : LAYER_SIZE) mix(static_cast<uint64_t>(n));
    for (auto& a : LAYER_ACTIVATION_TYPE) mix(static_cast<uint64_t>(a.index()));

    constexpr char A[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    std::array<char, 12> out{};
    for (int i = 0; i < 11; ++i) {
        out[i] = A[h & 0x3F];
        h >>= 6;
    }
    return out;
}();

bool save_model(const std::string& path) {
    const std::string tmp = path + ".tmp";

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

    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool load_model(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    f.read(reinterpret_cast<char*>(neural_network::weights.data()), static_cast<std::streamsize>(WEIGHTS.size()));
    if (f.gcount() != static_cast<std::streamsize>(neural_network::weights.size())) return false;

    f.read(reinterpret_cast<char*>(neural_network::biases.data()), static_cast<std::streamsize>(BIASES.size()));
    if (f.gcount() != static_cast<std::streamsize>(neural_network::biases.size())) return false;

    return true;
}
