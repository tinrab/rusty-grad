use crate::{activation::PureActivationFunctionLike, math::Matrix};

pub struct SigmoidActivationFunction;

impl SigmoidActivationFunction {
    pub fn new() -> Self {
        Self {}
    }
}

impl PureActivationFunctionLike for SigmoidActivationFunction {
    fn forward(&self, input: &Matrix) -> Matrix {
        input.map(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn backward(&self, input: &Matrix) -> Matrix {
        let sig = self.forward(input);
        sig.component_mul(&sig.map(|x| 1.0 - x))
    }
}
