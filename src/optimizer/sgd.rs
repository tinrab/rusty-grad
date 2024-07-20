use crate::math::Matrix;

use super::OptimizerLike;

pub struct SgdOptimizer {
    learning_rate: f32,
}

impl SgdOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl OptimizerLike for SgdOptimizer {
    fn update(
        &mut self,
        _epoch: usize,
        parameters: &Matrix,
        parameters_gradient: &Matrix,
    ) -> Matrix {
        parameters - parameters_gradient.scale(self.learning_rate)
    }
}
