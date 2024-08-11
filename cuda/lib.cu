#include <iostream>
#include <string>

#include "cuda.h"
#include "network.h"

extern "C" {
    typedef struct {
        float* data;
        size_t rows;
        size_t cols;
    } FfiTensor;

    typedef struct FfiNetwork FfiNetwork;
    typedef struct FfiLayer FfiLayer;

    struct FfiNetwork {
        Network* net;
    };

    struct FfiLayer {
        Layer* layer;
    };

    Tensor tensor_from_ffi(const FfiTensor* ffi) {
        Tensor t(ffi->rows, ffi->cols);
        std::memcpy(t.data(), ffi->data, ffi->rows * ffi->cols * sizeof(float));
        return t;
    }

    FfiTensor tensor_to_ffi(const Tensor& tensor) {
        FfiTensor ffi;
        ffi.rows = tensor.rows();
        ffi.cols = tensor.cols();
        size_t size = tensor.size();
        ffi.data = (float*)malloc(size * sizeof(float));
        std::memcpy(ffi.data, tensor.data(), size * sizeof(float));
        return ffi;
    }

    FfiLayer* create_dense_layer(
        size_t input_size,
        size_t output_size,
        const FfiTensor* weights,
        const FfiTensor* biases
    ) {
        Tensor w = tensor_from_ffi(weights);
        Tensor b = tensor_from_ffi(biases);
        FfiLayer* ffi_layer = new FfiLayer;
        ffi_layer->layer = new DenseLayer(input_size, output_size, w, b);
        return ffi_layer;
    }

    FfiLayer* create_activation_layer(int activation_type) {
        ActivationType type;
        switch (activation_type) {
            case 0:
                type = ActivationType::ReLU;
                break;
            case 1:
                type = ActivationType::Sigmoid;
                break;
            case 2:
                type = ActivationType::Tanh;
                break;
            default:
                type = ActivationType::ReLU;
                break;
        }
        FfiLayer* ffi_layer = new FfiLayer;
        ffi_layer->layer = new ActivationLayer(type);
        return ffi_layer;
    }

    FfiNetwork* create_network(FfiLayer** layers, size_t num_layers) {
        FfiNetwork* ffi_net = new FfiNetwork;
        ffi_net->net = new Network();
        for (size_t i = 0; i < num_layers; i++) {
            ffi_net->net->add_layer(layers[i]->layer);
        }
        return ffi_net;
    }

    FfiTensor network_forward(FfiNetwork* ffi_net, const FfiTensor* input) {
        Tensor in_tensor = tensor_from_ffi(input);
        Tensor out_tensor = ffi_net->net->forward(in_tensor);
        return tensor_to_ffi(out_tensor);
    }

    // void free_ffi_tensor(FfiTensor* tensor) {
    //     if (tensor && tensor->data) {
    //         free(tensor->data);
    //         tensor->data = nullptr;
    //     }
    // }

    // void free_network(FfiNetwork* ffi_net) {
    //     if (ffi_net) {
    //         delete ffi_net->net;
    //         delete ffi_net;
    //     }
    // }

    // void free_layer(FfiLayer* ffi_layer) {
    //     if (ffi_layer) {
    //         delete ffi_layer->layer;
    //         delete ffi_layer;
    //     }
    // }

    // Matrix network_forward(Network* network, Matrix input) {
    //     // Matrix output = input;

    //     // for (size_t i = 0; i < network->layers.size(); i++) {
    //     //     Layer& layer = network->layers[i];

    //     //     if (layer.dense != nullptr) {
    //     //         printf("%s\n", layer.dense->to_string().c_str());
    //     //         // output = layer_dense_run_forward(
    //     //         //   //   layer.dense, input.data(), output->data()
    //     //         //   layer.dense,
    //     //         //   input
    //     //         // );
    //     //     }
    //     // }

    //     // printf("output: (%d)\n", std::to_string(input._rows).c_str());
    //     // printf("output: (%d)\n", input._rows);

    //     // printf("output: (%d, %d)\n", input.rows(), input.columns());

    //     // Build a simple network:
    //     // Dense layer: 3 inputs -> 4 outputs, then ReLU,
    //     // Dense layer: 4 inputs -> 2 outputs, then Sigmoid.
    //     // Dense1: weights dimensions (4x3), biases (4x1)

    //     return input;
    // }
}
