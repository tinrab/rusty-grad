use crate::{activation::PureActivationFunctionLike, math::Matrix};

pub struct TanhActivationFunction;

impl TanhActivationFunction {
    pub fn new() -> Self {
        Self {}
    }
}

impl PureActivationFunctionLike for TanhActivationFunction {
    fn forward(&self, input: &Matrix) -> Matrix {
        input.map(|x| x.tanh())
    }

    fn backward(&self, input: &Matrix) -> Matrix {
        let tanh = self.forward(input);
        tanh.map(|x| 1.0 - x.powi(2))
    }
}
