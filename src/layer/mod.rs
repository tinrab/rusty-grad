use crate::{math::Matrix, optimizer::Optimizer};
use activation::ActivationLayer;
use dense::DenseLayer;
use enum_dispatch::enum_dispatch;

pub mod activation;
pub mod dense;

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
}
