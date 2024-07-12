cmake_minimum_required(VERSION 3.20)
project(CudaChallenge)
enable_language(CXX CUDA)

# C++ project default - feel free to use C++20 here if you prefer
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set up the external libraries using conan - make sure you run
# `conan install ..` in the build directory before you configure cmake
# or it will fail here!
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_library(CudaFilterLib
    include/Filter.hpp
    include/FilterPipeline.hpp
    src/FilterPipeline.cpp
    src/BilateralFilter.cu
    include/FilterFactory.hpp
    src/FilterFactory.cpp
    include/BilateralFilter.hpp
    include/cuda_utils.hpp

)

add_executable(CudaChallenge
        src/main.cpp
        include/NdVector.hpp
        src/Png.cpp
        include/Png.hpp
        )

#Test executable
add_executable(test_bilateral_filter
    tests/test_bilateral_filter.cpp
)
       

# Link Conan dependencies
find_package(Catch2 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(gsl-lite CONFIG REQUIRED)
find_package(PNG CONFIG REQUIRED)

# link all external libraries
target_link_libraries(CudaChallenge PRIVATE CudaFilterLib Catch2::Catch2 fmt::fmt gsl::gsl-lite PNG::PNG)

# link all external libraries for test executable
target_link_libraries(test_bilateral_filter PRIVATE
    CudaFilterLib
    Catch2::Catch2WithMain
    fmt::fmt
    gsl::gsl-lite
)

target_compile_definitions(CudaChallenge PRIVATE RESOURCE_DIR=\"${CMAKE_CURRENT_LIST_DIR}/data\" _USE_MATH_DEFINES)
target_compile_definitions(test_bilateral_filter PRIVATE RESOURCE_DIR=\"${CMAKE_CURRENT_LIST_DIR}/data\")

# Include directories
target_include_directories(CudaChallenge PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(test_bilateral_filter PRIVATE ${CMAKE_CURRENT_LIST_DIR})

#Enable testing
enable_testing()

# Add Catch2 integration for CTest
include(CTest)
include(Catch)
catch_discover_tests(test_bilateral_filter)