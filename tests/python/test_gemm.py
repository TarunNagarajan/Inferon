import numpy as np
import inferon_core
import pytest

def test_dense_gemm_naive():
    # Define matrix dimensions
    M, K, N = 10, 20, 15

    # Generate random NumPy arrays
    A_np = np.random.rand(M, K).astype(np.float32)
    B_np = np.random.rand(K, N).astype(np.float32)

    # Perform matrix multiplication using NumPy
    C_np = np.dot(A_np, B_np)

    # Flatten arrays for C++ function
    A_flat = A_np.flatten().tolist()
    B_flat = B_np.flatten().tolist()

    # Call the C++ GEMM function
    C_cpp_flat = inferon_core.dense_gemm_naive(A_flat, B_flat, M, K, N)

    # Reshape C++ result back to matrix form
    C_cpp = np.array(C_cpp_flat).reshape(M, N)

    # Assert that the results are numerically close
    np.testing.assert_allclose(C_np, C_cpp, rtol=1e-5, atol=1e-5)

    print(f"\nNumPy result shape: {C_np.shape}")
    print(f"C++ result shape: {C_cpp.shape}")
    print("Dense GEMM Naive test passed!")

# To run this test:
# 1. Navigate to the project root (C:/Users/ultim/Downloads/Inferon)
# 2. Build and install the Python package: pip install .
# 3. Run pytest from the project root: pytest tests/python/test_gemm.py

