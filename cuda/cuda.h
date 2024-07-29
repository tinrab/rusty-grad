#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void
checkCudaError(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess) {
        printf(
          "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error)
        );
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
