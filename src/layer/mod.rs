use crate::{math::Matrix, optimizer::Optimizer};
use activation::ActivationLayer;
use conv2d::Convolution2dLayer;
use dense::DenseLayer;
use enum_dispatch::enum_dispatch;
use reshape::ReshapeLayer;

pub mod activation;
pub mod conv2d;
pub mod dense;
pub mod reshape;

#[enum_dispatch]
pub trait LayerLike {
    fn forward(&mut self, input: &Matrix) -> Matrix;

    fn backward(
        &mut self,
        epoch: usize,
        output_gradient: &Matrix,
        optimizer: &mut Optimizer,
    ) -> Matrix;
}

#[enum_dispatch(LayerLike)]
pub enum Layer {
    Dense(DenseLayer),
    Activation(ActivationLayer),
    Reshape(ReshapeLayer),
    Convolution2d(Convolution2dLayer),
}
