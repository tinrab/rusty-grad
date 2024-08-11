use std::ops::{Add, Mul, Sub};

/// A simple multi-dimensional tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

impl<T: Copy + std::fmt::Debug> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(expected, data.len(), "data length does not match shape");
        Tensor { shape, data }
    }

    pub fn from_elem(shape: Vec<usize>, value: T) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![value; size],
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.shape.len() {
            return None;
        }

        let mut flat_index = 0;
        let mut stride = 1;

        for (i, &idx) in indices.iter().rev().enumerate() {
            let dim = self.shape[self.shape.len() - 1 - i];
            if idx >= dim {
                return None;
            }
            flat_index += idx * stride;
            stride *= dim;
        }

        self.data.get(flat_index)
    }

    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape.iter().cloned().filter(|&d| d != 1).collect();
        if new_shape.is_empty() {
            // A scalar tensor must have exactly one element.
            assert_eq!(
                self.data.len(),
                1,
                "cannot squeeze tensor with more than one element into a scalar shape"
            );
        }
        Tensor {
            shape: new_shape,
            data: self.data.clone(),
        }
    }

    pub fn unsqueeze(&self, axis: usize) -> Self {
        assert!(axis <= self.shape.len(), "axis out of bounds");
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        Tensor {
            shape: new_shape,
            data: self.data.clone(),
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            new_size,
            self.data.len(),
            "new shape product must match number of elements"
        );
        Tensor {
            shape: new_shape,
            data: self.data.clone(),
        }
    }

    pub fn transpose(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "Transpose only supports 2D tensors");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut new_data = Vec::with_capacity(self.data.len());
        // Initialize with dummy values; T: Copy ensures this is valid.
        new_data.resize(self.data.len(), self.data[0]);
        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }
        Tensor {
            shape: vec![cols, rows],
            data: new_data,
        }
    }
}

// Element-wise addition
impl<T> Add for Tensor<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape, rhs.shape, "Shapes must be equal for addition");
        let data = self
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            shape: self.shape,
            data,
        }
    }
}

// Element-wise subtraction
impl<T> Sub for Tensor<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must be equal for subtraction"
        );
        let data = self
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a - b)
            .collect();
        Tensor {
            shape: self.shape,
            data,
        }
    }
}

// Element-wise multiplication
impl<T> Mul for Tensor<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape, rhs.shape,
            "Shapes must be equal for multiplication"
        );
        let data = self
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            shape: self.shape,
            data,
        }
    }
}

// Scalar multiplication: Tensor * scalar
impl<T> Mul<T> for Tensor<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        let data = self.data.into_iter().map(|a| a * rhs).collect();
        Tensor {
            shape: self.shape,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_get() {
        let tensor = Tensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.get(&[0, 0]), Some(&1));
        assert_eq!(tensor.get(&[0, 1]), Some(&2));
        assert_eq!(tensor.get(&[1, 2]), Some(&6));
        // Out of bounds access should return None.
        assert_eq!(tensor.get(&[2, 0]), None);
    }

    #[test]
    fn from_elem() {
        let tensor = Tensor::from_elem(vec![3, 2], 5);
        assert_eq!(tensor.shape(), &[3, 2]);
        for &elem in &tensor.data {
            assert_eq!(elem, 5);
        }
    }

    #[test]
    fn squeeze() {
        // Tensor with shape [1, 2, 3, 1] should become [2, 3]
        let tensor = Tensor::new(vec![1, 2, 3, 1], vec![1, 2, 3, 4, 5, 6]);
        let squeezed = tensor.squeeze();
        assert_eq!(squeezed.shape(), &[2, 3]);

        // Test squeezing to a scalar: tensor with shape [1, 1, 1] containing a single element.
        let tensor_scalar = Tensor::new(vec![1, 1, 1], vec![42]);
        let squeezed_scalar = tensor_scalar.squeeze();
        assert_eq!(squeezed_scalar.shape(), &[]);
        assert_eq!(squeezed_scalar.data, vec![42]);
    }

    #[test]
    fn unsqueeze() {
        let tensor = Tensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let unsqueezed = tensor.unsqueeze(0);
        assert_eq!(unsqueezed.shape(), &[1, 2, 3]);

        let unsqueezed1 = tensor.unsqueeze(1);
        assert_eq!(unsqueezed1.shape(), &[2, 1, 3]);

        let unsqueezed_last = tensor.unsqueeze(2);
        assert_eq!(unsqueezed_last.shape(), &[2, 3, 1]);
    }

    #[test]
    fn reshape() {
        let tensor = Tensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let reshaped = tensor.reshape(vec![3, 2]);
        assert_eq!(reshaped.shape(), &[3, 2]);
        // Data ordering remains the same.
        assert_eq!(reshaped.data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn transpose() {
        let tensor = Tensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let transposed = tensor.transpose();
        assert_eq!(transposed.shape(), &[3, 2]);
        // Transpose should swap rows and columns:
        // Original: [ [1, 2, 3], [4, 5, 6] ]
        // Transposed: [ [1, 4], [2, 5], [3, 6] ]
        assert_eq!(transposed.data, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn add() {
        let tensor1 = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]);
        let tensor2 = Tensor::new(vec![2, 2], vec![5, 6, 7, 8]);
        let result = tensor1 + tensor2;
        assert_eq!(result.data, vec![6, 8, 10, 12]);
    }

    #[test]
    fn sub() {
        let tensor1 = Tensor::new(vec![2, 2], vec![5, 7, 9, 11]);
        let tensor2 = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]);
        let result = tensor1 - tensor2;
        assert_eq!(result.data, vec![4, 5, 6, 7]);
    }

    #[test]
    fn mul_elementwise() {
        let tensor1 = Tensor::new(vec![2, 2], vec![1, 2, 3, 4]);
        let tensor2 = Tensor::new(vec![2, 2], vec![2, 3, 4, 5]);
        let result = tensor1 * tensor2;
        assert_eq!(result.data, vec![2, 6, 12, 20]);
    }

    #[test]
    fn scalar_mul() {
        let tensor = Tensor::new(vec![2, 3], vec![1, 2, 3, 4, 5, 6]);
        let result = tensor * 3;
        assert_eq!(result.data, vec![3, 6, 9, 12, 15, 18]);
    }
}
