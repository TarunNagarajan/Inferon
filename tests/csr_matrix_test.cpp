#include <gtest/gtest.h>
#include "CSRMatrix.h"

TEST(CSRMatrixTest, FromDenseEmptyMatrix) {
    std::vector<std::vector<float>> dense_matrix = {};
    inferon::CSRMatrix csr_matrix = inferon::CSRMatrix::from_dense(dense_matrix);

    ASSERT_TRUE(csr_matrix.values.empty());
    ASSERT_TRUE(csr_matrix.col_indices.empty());
    ASSERT_EQ(csr_matrix.row_ptr.size(), 1);
    ASSERT_EQ(csr_matrix.row_ptr[0], 0);
    ASSERT_EQ(csr_matrix.num_rows, 0);
    ASSERT_EQ(csr_matrix.num_cols, 0);
}

TEST(CSRMatrixTest, FromDenseSimpleMatrix) {
    std::vector<std::vector<float>> dense_matrix = {
        {1.0f, 0.0f, 2.0f},
        {0.0f, 3.0f, 0.0f},
        {4.0f, 0.0f, 5.0f}
    };
    inferon::CSRMatrix csr_matrix = inferon::CSRMatrix::from_dense(dense_matrix);

    // Expected values
    std::vector<float> expected_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<int32_t> expected_col_indices = {0, 2, 1, 0, 2};
    std::vector<int32_t> expected_row_ptr = {0, 2, 3, 5};

    ASSERT_EQ(csr_matrix.values, expected_values);
    ASSERT_EQ(csr_matrix.col_indices, expected_col_indices);
    ASSERT_EQ(csr_matrix.row_ptr, expected_row_ptr);
    ASSERT_EQ(csr_matrix.num_rows, 3);
    ASSERT_EQ(csr_matrix.num_cols, 3);
}

TEST(CSRMatrixTest, ToDenseSimpleMatrix) {
    inferon::CSRMatrix csr_matrix;
    csr_matrix.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    csr_matrix.col_indices = {0, 2, 1, 0, 2};
    csr_matrix.row_ptr = {0, 2, 3, 5};
    csr_matrix.num_rows = 3;
    csr_matrix.num_cols = 3;

    std::vector<std::vector<float>> expected_dense_matrix = {
        {1.0f, 0.0f, 2.0f},
        {0.0f, 3.0f, 0.0f},
        {4.0f, 0.0f, 5.0f}
    };

    std::vector<std::vector<float>> actual_dense_matrix = csr_matrix.to_dense();

    ASSERT_EQ(actual_dense_matrix.size(), expected_dense_matrix.size());
    for (size_t i = 0; i < actual_dense_matrix.size(); ++i) {
        ASSERT_EQ(actual_dense_matrix[i].size(), expected_dense_matrix[i].size());
        for (size_t j = 0; j < actual_dense_matrix[i].size(); ++j) {
            ASSERT_FLOAT_EQ(actual_dense_matrix[i][j], expected_dense_matrix[i][j]);
        }
    }
}

TEST(CSRMatrixTest, FromDenseToDenseRoundtrip) {
    std::vector<std::vector<float>> original_dense_matrix = {
        {1.0f, 0.0f, 2.0f, 0.0f},
        {0.0f, 3.0f, 0.0f, 6.0f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {7.0f, 0.0f, 8.0f, 0.0f}
    };

    inferon::CSRMatrix csr_matrix = inferon::CSRMatrix::from_dense(original_dense_matrix);
    std::vector<std::vector<float>> roundtrip_dense_matrix = csr_matrix.to_dense();

    ASSERT_EQ(roundtrip_dense_matrix.size(), original_dense_matrix.size());
    for (size_t i = 0; i < roundtrip_dense_matrix.size(); ++i) {
        ASSERT_EQ(roundtrip_dense_matrix[i].size(), original_dense_matrix[i].size());
        for (size_t j = 0; j < roundtrip_dense_matrix[i].size(); ++j) {
            ASSERT_FLOAT_EQ(roundtrip_dense_matrix[i][j], original_dense_matrix[i][j]);
        }
    }
}

TEST(CSRMatrixTest, FromDenseAllZeros) {
    std::vector<std::vector<float>> dense_matrix = {
        {0.0f, 0.0f},
        {0.0f, 0.0f}
    };
    inferon::CSRMatrix csr_matrix = inferon::CSRMatrix::from_dense(dense_matrix);

    ASSERT_TRUE(csr_matrix.values.empty());
    ASSERT_TRUE(csr_matrix.col_indices.empty());
    ASSERT_EQ(csr_matrix.row_ptr.size(), 3);
    ASSERT_EQ(csr_matrix.row_ptr[0], 0);
    ASSERT_EQ(csr_matrix.row_ptr[1], 0);
    ASSERT_EQ(csr_matrix.row_ptr[2], 0);
    ASSERT_EQ(csr_matrix.num_rows, 2);
    ASSERT_EQ(csr_matrix.num_cols, 2);

    std::vector<std::vector<float>> expected_dense_matrix = {
        {0.0f, 0.0f},
        {0.0f, 0.0f}
    };
    std::vector<std::vector<float>> actual_dense_matrix = csr_matrix.to_dense();
    ASSERT_EQ(actual_dense_matrix, expected_dense_matrix);
}

TEST(CSRMatrixTest, FromDenseFullMatrix) {
    std::vector<std::vector<float>> dense_matrix = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };
    inferon::CSRMatrix csr_matrix = inferon::CSRMatrix::from_dense(dense_matrix);

    std::vector<float> expected_values = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<int32_t> expected_col_indices = {0, 1, 0, 1};
    std::vector<int32_t> expected_row_ptr = {0, 2, 4};

    ASSERT_EQ(csr_matrix.values, expected_values);
    ASSERT_EQ(csr_matrix.col_indices, expected_col_indices);
    ASSERT_EQ(csr_matrix.row_ptr, expected_row_ptr);
    ASSERT_EQ(csr_matrix.num_rows, 2);
    ASSERT_EQ(csr_matrix.num_cols, 2);
}
