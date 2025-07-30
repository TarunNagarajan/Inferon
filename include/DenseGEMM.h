#ifndef INFERON_DENSE_GEMM_H
#define INFERON_DENSE_GEMM_H

#include <vector>

namespace inferon {

/**
 * @brief Performs a naive dense General Matrix Multiplication (GEMM): C = A * B.
 * @param A The first matrix (M x K).
 * @param B The second matrix (K x N).
 * @param C The output matrix (M x N).
 * @param M Number of rows in A and C.
 * @param K Number of columns in A and rows in B.
 * @param N Number of columns in B and C.
 */
void dense_gemm_naive(
    const std::vector<float>& A,
    const std::vector<float>& B,
    std::vector<float>& C,
    int M, int K, int N
);

/**
* @brief applies the ReLu activation function element-wise to a vector
* @param input the input vector
* @param output the output vector
**/

void relu(const std::vector<float>& input, std::vector<float>& output);

} // namespace inferon

#endif // INFERON_DENSE_GEMM_H
