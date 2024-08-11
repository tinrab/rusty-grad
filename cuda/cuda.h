#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(call)                                     \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << std::endl;                                \
            exit(1);                                               \
        }                                                          \
    }