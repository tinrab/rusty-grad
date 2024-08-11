#pragma once

#include <stdio.h>

class Matrix {
  private:
    size_t _rows;
    size_t _columns;
    float* _data;

  public:
    Matrix(size_t rows, size_t columns) : _rows(rows), _columns(columns) {
        this->_data = new float[rows * columns];
    }

    // ~Matrix()
    // {
    // printf("dealloc\n");
    // printf("dealloc: (%d, %d)\n", this->_rows, this->_columns);
    // delete[] this->_data;
    // }

    // Matrix(const Matrix& other)
    //   : _rows(other._rows)
    //   , _columns(other._columns)
    // {
    //     printf("copy ctor: (%d, %d)\n", this->_rows, this->_columns);
    //     this->_data = new float[other._rows * other._columns];
    //     // for (size_t i = 0; i < other._rows * other._columns; i++) {
    //     //     this->_data[i] = other._data[i];
    //     // }
    // }

    Matrix& operator=(const Matrix& other) {
        printf("COPY ASSIGN\n");
        // printf("copy assign: (%d, %d)\n", this->_rows, this->_columns);
        if (this == &other) {
            return *this;
        }

        return *this;
    }

    float& operator()(size_t i, size_t j) {
        return this->_data[i * this->_columns + j];
    }

    float operator[](size_t i) const {
        return this->_data[i];
    }

    size_t rows() const {
        return this->_rows;
    }

    size_t columns() const {
        return this->_columns;
    }

    size_t size() const {
        return this->_rows * this->_columns;
    }

    float* data() {
        return this->_data;
    }
};
