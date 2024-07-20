use enum_dispatch::enum_dispatch;

use crate::math::Matrix;

#[enum_dispatch]
pub trait LossFunctionLike {
    fn forward(&self, input: &Matrix, target: &Matrix) -> f32;
    fn backward(&self, input: &Matrix, target: &Matrix) -> Matrix;
}

#[enum_dispatch(LossFunctionLike)]
pub enum LossFunction {
    Mse(MseLossFunction),
}

pub struct MseLossFunction;

impl MseLossFunction {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFunctionLike for MseLossFunction {
    fn forward(&self, input: &Matrix, target: &Matrix) -> f32 {
        let error = target - input;
        error.map(|x| x.powi(2)).mean()
    }

    fn backward(&self, input: &Matrix, target: &Matrix) -> Matrix {
        ((input - target) * 2.0) / input.len() as f32
    }
}
