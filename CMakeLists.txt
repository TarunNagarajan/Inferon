# CMake build system for Inferon
cmake_minimum_required(VERSION 3.15)
project(Inferon CXX)

# Set a default build type if none is specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

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

# Add pybind11 for Python bindings
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1 # Use a specific tag for stability
)
FetchContent_MakeAvailable(pybind11)

# Add the standalone ONNX parser
add_subdirectory(../onnx_parser ${CMAKE_BINARY_DIR}/onnx_parser-build)

# --- Project Structure ---
# Enable testing
enable_testing()

# Add the 'tests' subdirectory to the build
# Add the 'tests' subdirectory to the build
add_subdirectory(tests)

# Add the Python bindings module
pybind11_add_module(inferon_core src/bindings.cpp src/DenseGEMM.cpp)
target_include_directories(inferon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ../onnx_parser/include)
target_link_libraries(inferon_core PRIVATE onnx_parser)

# --- Welcome Message ---
message(STATUS "Inferon project configured. C++ standard: ${CMAKE_CXX_STANDARD}")
