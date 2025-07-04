# CMake script for the 'tests' directory

# Create an executable for our tests
add_executable(run_tests
    hello_test.cpp
)

# Link the test executable against Google Test
# gtest_main automatically creates a main function for the tests
target_link_libraries(run_tests
    GTest::gtest_main
)

# Discover and add tests to the CTest runner
include(GoogleTest)
gtest_discover_tests(run_tests)
