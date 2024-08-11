#include "cuda.h"
#include "network.h"

Tensor Network::forward(const Tensor& input) {
    size_t current_size = input.size();
    float* device_current;
    CHECK_CUDA_ERROR(cudaMalloc(&device_current, current_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(
        device_current,
        input.data(),
        current_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    for (auto layer : this->_layers) {
        size_t next_size;
        float* device_next =
            layer->forward_device(device_current, current_size, next_size);
        if (device_current != device_next) {
            cudaFree(device_current);
        }
        device_current = device_next;
        current_size = next_size;
    }

    Tensor output(current_size, 1);
    CHECK_CUDA_ERROR(cudaMemcpy(
        output.data(),
        device_current,
        current_size * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    cudaFree(device_current);

    return output;
}