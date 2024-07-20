use crate::{activation::PureActivationFunctionLike, math::Matrix};

pub struct IdentityActivationFunction;

impl IdentityActivationFunction {
    pub fn new() -> Self {
        Self {}
    }
}

impl PureActivationFunctionLike for IdentityActivationFunction {
    fn forward(&self, input: &Matrix) -> Matrix {
        input.clone()
    }

    fn backward(&self, input: &Matrix) -> Matrix {
        let shape = input.shape();
        Matrix::identity(shape.0, shape.1)
    }
}
