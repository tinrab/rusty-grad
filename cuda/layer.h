#pragma once

#include <cassert>

#include "activation.h"
#include "tensor.h"

class Layer
{
  public:
    virtual ~Layer() {}

    virtual float*
    forward_device(float* device_input, size_t in_size, size_t& out_size) = 0;
};

class DenseLayer : public Layer
{
  private:
    size_t _input_size, _output_size;
    Tensor _weights;
    Tensor _biases;

  public:
    DenseLayer(
      size_t input_size,
      size_t output_size,
      const Tensor& weights,
      const Tensor& biases
    )
      : _input_size(input_size)
      , _output_size(output_size)
      , _weights(weights)
      , _biases(biases)
    {
        assert(_weights.size() == _input_size * _output_size);
        assert(_biases.size() == _output_size);
    }

    virtual float* forward_device(
      float* device_input,
      size_t in_size,
      size_t& out_size
    ) override;
};

class ActivationLayer : public Layer
{
  private:
    ActivationType _type;

  public:
    ActivationLayer(ActivationType type)
      : _type(type)
    {
    }

    virtual float* forward_device(
      float* device_input,
      size_t in_size,
      size_t& out_size
    ) override;
};
