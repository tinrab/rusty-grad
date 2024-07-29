#include <sstream>

#include "dense.h"

std::string
DenseLayer::to_string()
{
    std::stringstream ss;
    ss << "DenseLayer {\n";
    ss << "  input_size: " << std::to_string(this->input_size) << ",\n";
    ss << "  output_size: " << std::to_string(this->output_size) << ",\n";
    ss << "}\n";
    return ss.str();
}
