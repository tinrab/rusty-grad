#pragma once

#include <cstring>
#include <iostream>

class Tensor {
  private:
    float* _data;
    size_t _rows, _cols, _size;

  public:
    Tensor(size_t rows, size_t cols) :
        _rows(rows),
        _cols(cols),
        _size(rows * cols) {
        _data = new float[_size];
    }

    ~Tensor() {
        delete[] _data;
    }

    Tensor(const Tensor& other) :
        _rows(other._rows),
        _cols(other._cols),
        _size(other._size) {
        _data = new float[_size];
        std::memcpy(_data, other._data, _size * sizeof(float));
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (_size != other._size) {
                delete[] _data;
                _rows = other._rows;
                _cols = other._cols;
                _size = other._size;
                _data = new float[_size];
            }
            std::memcpy(_data, other._data, _size * sizeof(float));
        }
        return *this;
    }

    float* data() {
        return _data;
    }

    const float* data() const {
        return _data;
    }

    size_t rows() const {
        return _rows;
    }

    size_t cols() const {
        return _cols;
    }

    size_t size() const {
        return _size;
    }
};