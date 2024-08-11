#pragma once

#include <vector>

#include "layer.h"
#include "math.h"
#include "vec.h"

class Network {
  private:
    std::vector<Layer*> _layers;

  public:
    ~Network() {
        for (auto layer : _layers) {
            delete layer;
        }
    }

    void add_layer(Layer* layer) {
        _layers.push_back(layer);
    }

    Tensor forward(const Tensor& input);
};