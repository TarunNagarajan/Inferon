#include <pybind11/pybind11.h>
#include "DenseGEMM.h"

namespace py = pybind11;

// A simple C++ function to expose to Python
int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(inferon_core, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");

    m.def("dense_gemm_naive",
          [](const std::vector<float>& A, const std::vector<float>& B, int M, int K, int N) {
              std::vector<float> C(M * N);
              inferon::dense_gemm_naive(A, B, C, M, K, N);
              return C;
          },
          py::arg("A"), py::arg("B"), py::arg("M"), py::arg("K"), py::arg("N"),
          "Performs a naive dense General Matrix Multiplication (GEMM): C = A * B."
    );

    m.def("relu",
        [](const std::vector<float>& input) {
            std::vector<float> output(input.size()); 
            inferon::relu(input, output);
            return output;
        }, 
        py::arg("input"),
        "Performs the ReLu, activation function element-wise to a vector. Make sure that the input and output vector are of the same size. ");
}
