#include "DenseGEMM.h"
#include <stdexcept>

namespace inferon {

void dense_gemm_naive(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N
) {
    if (A.size() != M * K || B.size() != K * N || C.size() != M * N) {
        throw std::invalid_argument("Matrix dimensions do not match input sizes.");
    }

    // Initialize C with zeros
    std::fill(C.begin(), C.end(), 0.0f);

    const int TILE_SIZE = 16; // A common tile size

    for (int m = 0; m < M; m += TILE_SIZE) {
        for (int n = 0; n < N; n += TILE_SIZE) {
            for (int k = 0; k < K; k += TILE_SIZE) {
                // Mini-GEMM for the tiles
                for (int m_tile = m; m_tile < std::min(m + TILE_SIZE, M); ++m_tile) {
                    for (int n_tile = n; n_tile < std::min(n + TILE_SIZE, N); ++n_tile) {
                        for (int k_tile = k; k_tile < std::min(k + TILE_SIZE, K); ++k_tile) {
                            C[m_tile * N + n_tile] += A[m_tile * K + k_tile] * B[k_tile * N + n_tile];
                        }
                    }
                }
            }
        }
    }
}

void relu(const std::vector<float>& input, std::vector<float>& output) {
    if (input.size() != output.size()) {
        throw std::invalid_argument("Input and Output vectors should have the same size, for ReLu");
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

} // namespace inferon
