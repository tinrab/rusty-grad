#pragma once

#include <string>

#include "../math.h"

class DenseLayer
{
  public:
    size_t input_size;
    size_t output_size;
    Matrix weights;
    Matrix biases;

    std::string to_string();
};
