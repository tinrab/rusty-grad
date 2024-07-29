#include <iostream>

#include "cuda.h"
#include "math.h"
#include "network.h"

__global__ void
cuda_layer_dense_forward(
  float* weights,
  float* biases,
  size_t input_size,
  size_t output_size,
  float* input,
  float* output
)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= output_size) {
        return;
    }

    float sum = 0.0f;

    for (size_t i = 0; i < input_size; i++) {
        sum += input[i] * weights[i * output_size + idx];
    }

    output[idx] = sum + biases[idx];
}

Matrix
layer_dense_run_forward(DenseLayer* dense, Matrix input)
{
    size_t input_size = dense->input_size;
    size_t output_size = dense->output_size;

    float* dev_input;
    float* dev_output;
    float* dev_weights;
    float* dev_biases;

    size_t input_bytes = input_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);
    size_t weights_bytes = input_size * output_size * sizeof(float);
    size_t biases_bytes = output_size * sizeof(float);

    CHECK_CUDA_ERROR(cudaMalloc(&dev_input, input_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_output, output_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_weights, weights_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&dev_biases, biases_bytes));

    CHECK_CUDA_ERROR(
      cudaMemcpy(dev_input, input.data(), input_bytes, cudaMemcpyHostToDevice)
    );
    CHECK_CUDA_ERROR(cudaMemcpy(
      dev_weights, dense->weights.data(), weights_bytes, cudaMemcpyHostToDevice
    ));
    CHECK_CUDA_ERROR(cudaMemcpy(
      dev_biases, dense->biases.data(), biases_bytes, cudaMemcpyHostToDevice
    ));

    printf("start cuda_layer_dense_forward\n");
    cuda_layer_dense_forward<<<output_size / 256 + 1, 256>>>(
      dev_weights, dev_biases, input_size, output_size, dev_input, dev_output
    );
    printf("end cuda_layer_dense_forward\n");

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaGetLastError());

    Matrix output(output_size, 1);

    CHECK_CUDA_ERROR(cudaMemcpy(
      output.data(), dev_output, output_bytes, cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA_ERROR(cudaFree(dev_input));
    CHECK_CUDA_ERROR(cudaFree(dev_output));
    CHECK_CUDA_ERROR(cudaFree(dev_weights));
    CHECK_CUDA_ERROR(cudaFree(dev_biases));
    // CHECK_CUDA_ERROR(cudaDeviceReset());

    return output;
}

#include <string>

extern "C"
{
    Matrix network_forward(Network* network, Matrix input)
    {
        // Matrix output = input;

        // for (size_t i = 0; i < network->layers.size(); i++) {
        //     Layer& layer = network->layers[i];

        //     if (layer.dense != nullptr) {
        //         printf("%s\n", layer.dense->to_string().c_str());
        //         // output = layer_dense_run_forward(
        //         //   //   layer.dense, input.data(), output->data()
        //         //   layer.dense,
        //         //   input
        //         // );
        //     }
        // }

        // printf("output: (%d)\n", std::to_string(input._rows).c_str());
        // printf("output: (%d)\n", input._rows);

        printf("output: (%d, %d)\n", input.rows(), input.columns());

        return input;
    }
}
