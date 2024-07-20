use crate::math::Matrix;
use enum_dispatch::enum_dispatch;
use sgd::SgdOptimizer;

pub mod sgd;

#[enum_dispatch]
pub trait OptimizerLike {
    fn update(&mut self, epoch: usize, parameters: &Matrix, parameters_gradient: &Matrix)
        -> Matrix;
}

#[enum_dispatch(OptimizerLike)]
pub enum Optimizer {
    Sgd(SgdOptimizer),
}
