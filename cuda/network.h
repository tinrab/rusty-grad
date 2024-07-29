#pragma once

#include <memory>

#include "layer/layer.h"
#include "math.h"
#include "vec.h"

class Network
{
  public:
    Vec<Layer> layers;
};
