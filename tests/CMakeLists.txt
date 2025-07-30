
# Create an executable for our tests
add_executable(run_tests
    hello_test.cpp
    csr_matrix_test.cpp
)

# Link the test executable against Google Test
# gtest_main automatically creates a main function for the tests
target_link_libraries(run_tests
    GTest::gtest_main
)

add_test(NAME run_tests COMMAND run_tests)
