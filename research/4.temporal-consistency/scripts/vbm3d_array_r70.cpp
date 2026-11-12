// SPDX-License-Identifier: GPL-3.0-or-later
// Array I/O only. Official IPOL main/core sources and their notices remain intact.
#define main ipol_original_cli_main
#include "main.cpp"
#undef main

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>

int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("input output width height frames sigma Nf required");
        const unsigned width = std::stoul(argv[3]), height = std::stoul(argv[4]), frames = std::stoul(argv[5]);
        const float sigma = std::stof(argv[6]);
        const int nf = std::stoi(argv[7]);
        const uint64_t count = uint64_t(width)*height*frames*3;
        if (width < 16 || height < 16 || !frames || count > std::numeric_limits<unsigned>::max())
            throw std::runtime_error("unsupported dimensions");
        if (!(sigma > 0) || !std::isfinite(sigma) || (nf != 0 && nf != 4))
            throw std::runtime_error("positive finite sigma and registered Nf0/4 required");
        Video<float> noisy(width, height, frames, 3), basic, output;
        std::ifstream input(argv[1], std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(noisy.data.data()), count*sizeof(float)) || input.peek() != EOF)
            throw std::runtime_error("input size/read mismatch");
        if (!std::all_of(noisy.data.begin(), noisy.data.end(), [](float x){return std::isfinite(x);}))
            throw std::runtime_error("nonfinite input");
        Video<float> forward(width, height, frames, 2, 0.f), backward(width, height, frames, 2, 0.f);
        Parameters first, second;
        const unsigned unspecified = static_cast<unsigned>(-1);
        initializeParameters_1(first, -1, -1, nf, -1, -1, -1, -1, -1, -1, -1.f, -1.f,
                               unspecified, unspecified, sigma);
        initializeParameters_2(second, -1, -1, nf, -1, -1, -1, -1, -1, -1, -1.f,
                               unspecified, unspecified, sigma);
        first.mc = second.mc = false;
        printParameters(first, "hard thresholding"); printParameters(second, "Wiener filtering");
        const int status = run_vbm3d(sigma, noisy, forward, backward, basic, output, first, second, true);
        if (status != EXIT_SUCCESS || output.sz != noisy.sz || output.data.size() != count)
            throw std::runtime_error("official filter status/output shape failure");
        if (!std::all_of(output.data.begin(), output.data.end(), [](float x){return std::isfinite(x);}))
            throw std::runtime_error("nonfinite official output");
        std::ofstream target(argv[2], std::ios::binary | std::ios::out | std::ios::trunc);
        if (!target.write(reinterpret_cast<const char*>(output.data.data()), count*sizeof(float)))
            throw std::runtime_error("output write failed");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
