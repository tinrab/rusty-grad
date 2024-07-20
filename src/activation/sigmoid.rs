use crate::{activation::PureActivationFunctionLike, math::Matrix};

pub struct SigmoidActivationFunction;

impl SigmoidActivationFunction {
    pub fn new() -> Self {
        Self {}
    }
}

impl PureActivationFunctionLike for SigmoidActivationFunction {
    fn forward(&self, input: &Matrix) -> Matrix {
        let exp = (input * -1.0f32).exp();
        exp.map(|x| 1.0 / (1.0 + x))
    }

    fn backward(&self, input: &Matrix) -> Matrix {
        let sig = self.forward(input);
        sig.component_mul(&sig.map(|x| 1.0 - x))
    }
}
