#include "CSRMatrix.h"
#include <iostream>
#include <vector>

int main() {
    using namespace inferon;
    // Example dense matrix
    std::vector<std::vector<float>> dense = {
        {1.0f, 0.0f, 0.0f, 2.0f},
        {0.0f, 0.0f, 3.0f, 0.0f},
        {4.0f, 0.0f, 5.0f, 6.0f}
    };

    // Convert to CSR
    CSRMatrix csr = CSRMatrix::from_dense(dense);

    // Convert back to dense
    std::vector<std::vector<float>> dense2 = csr.to_dense();

    // Print the result
    std::cout << "Dense matrix after CSR round-trip:" << std::endl;
    for (const auto& row : dense2) {
        for (float val : row) {
            std::cout << val << ' ';
        }
        std::cout << std::endl;
    }
    return 0;
}
