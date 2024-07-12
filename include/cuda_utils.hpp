#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call)                                                                           \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));       \
        }                                                                                          \
    }

#define CUDA_CHECK_NO_THROW(call)                                                                  \
    {                                                                                              \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;                   \
        }                                                                                          \
    }