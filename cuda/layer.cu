#include <cassert>

#include "cuda.h"
#include "layer.h"

__global__ void dense_forward_kernel(
    const float* weights,
    const float* biases,
    const float* input,
    float* output,
    size_t input_size,
    size_t output_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (size_t j = 0; j < input_size; j++) {
            sum += weights[idx * input_size + j] * input[j];
        }
        output[idx] = sum + biases[idx];
    }
}

__global__ void relu_kernel(float* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

__global__ void sigmoid_kernel(float* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

__global__ void tanh_kernel(float* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = tanhf(data[idx]);
    }
}

float* DenseLayer::forward_device(
    float* device_input,
    size_t in_size,
    size_t& out_size
) {
    assert(in_size == this->_input_size);
    out_size = this->_output_size;
    float* device_output;
    CHECK_CUDA_ERROR(
        cudaMalloc(&device_output, this->_output_size * sizeof(float))
    );

    float* device_weights;
    CHECK_CUDA_ERROR(
        cudaMalloc(&device_weights, this->_weights.size() * sizeof(float))
    );
    CHECK_CUDA_ERROR(cudaMemcpy(
        device_weights,
        this->_weights.data(),
        this->_weights.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    float* device_biases;
    CHECK_CUDA_ERROR(
        cudaMalloc(&device_biases, this->_biases.size() * sizeof(float))
    );
    CHECK_CUDA_ERROR(cudaMemcpy(
        device_biases,
        this->_biases.data(),
        this->_biases.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    dim3 threads(256);
    dim3 blocks((this->_output_size + threads.x - 1) / threads.x);

    dense_forward_kernel<<<blocks, threads>>>(
        device_weights,
        device_biases,
        device_input,
        device_output,
        this->_input_size,
        this->_output_size
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cudaFree(device_weights);
    cudaFree(device_biases);

    return device_output;
}

float* ActivationLayer::forward_device(
    float* d_input,
    size_t in_size,
    size_t& out_size
) {
    out_size = in_size;
    dim3 threads(256);
    dim3 blocks((in_size + threads.x - 1) / threads.x);

    switch (this->_type) {
        case ActivationType::ReLU:
            relu_kernel<<<blocks, threads>>>(d_input, in_size);
            break;
        case ActivationType::Sigmoid:
            sigmoid_kernel<<<blocks, threads>>>(d_input, in_size);
            break;
        case ActivationType::Tanh:
            tanh_kernel<<<blocks, threads>>>(d_input, in_size);
            break;
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return d_input;
}
