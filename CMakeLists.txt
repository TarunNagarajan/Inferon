# CMake build system for Inferon
cmake_minimum_required(VERSION 3.15)
project(Inferon CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
# Add Google Test for unit testing
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Project Structure ---
# Enable testing
enable_testing()

# Add the 'tests' subdirectory to the build
add_subdirectory(tests)

# --- Welcome Message ---
message(STATUS "Inferon project configured. C++ standard: ${CMAKE_CXX_STANDARD}")
