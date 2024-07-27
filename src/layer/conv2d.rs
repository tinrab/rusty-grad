use crate::{
    initializer::{Initializer, InitializerLike},
    layer::LayerLike,
    math::{matrix_broadcast_dot, Matrix},
    optimizer::{Optimizer, OptimizerLike},
};

pub struct Convolution2dLayer {
    weights: Matrix,
    biases: Matrix,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    input: Option<Matrix>,
}

impl Convolution2dLayer {
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        initializer: &Initializer,
    ) -> Self {
        let weights = initializer.initialize_matrix((
            output_channels,
            input_channels * kernel_size.0 * kernel_size.1,
        ));
        let biases = initializer.initialize_matrix((output_channels, 1));

        Self {
            weights,
            biases,
            kernel_size,
            stride,
            padding,
            input: None,
        }
    }
}

impl LayerLike for Convolution2dLayer {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        todo!()
    }

    fn backward(
        &mut self,
        epoch: usize,
        output_gradient: &Matrix,
        optimizer: &mut Optimizer,
    ) -> Matrix {
        todo!()
    }
}
