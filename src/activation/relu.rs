use crate::{activation::PureActivationFunctionLike, math::Matrix};

pub struct ReLuActivationFunction;

impl ReLuActivationFunction {
    pub fn new() -> Self {
        Self {}
    }
}

impl PureActivationFunctionLike for ReLuActivationFunction {
    fn forward(&self, input: &Matrix) -> Matrix {
        input.map(|x| x.max(0.0))
    }

    fn backward(&self, input: &Matrix) -> Matrix {
        input.map(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}
