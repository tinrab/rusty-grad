#pragma once

#include <memory>

template<typename T>
class Vec
{
  private:
    size_t _len;
    T* _data;

  public:
    T& operator[](size_t i) { return this->_data[i]; }

    size_t size() const { return this->_len; }
};
