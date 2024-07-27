use crate::math::{Matrix, Shape, Vector};
use enum_dispatch::enum_dispatch;

#[enum_dispatch]
pub trait InitializerLike {
    fn initialize_matrix(&self, shape: Shape) -> Matrix;
}

#[enum_dispatch(InitializerLike)]
pub enum Initializer {
    Zero(ZeroInitializer),
    Uniform(UniformInitializer),
}

pub struct ZeroInitializer;

impl InitializerLike for ZeroInitializer {
    fn initialize_matrix(&self, shape: Shape) -> Matrix {
        Matrix::zeros(shape.0, shape.1)
    }
}

pub struct UniformInitializer {
    lower: f32,
    upper: f32,
}

impl UniformInitializer {
    pub fn new(lower: f32, upper: f32) -> Self {
        assert!(lower < upper, "lower must be less than upper");
        Self { lower, upper }
    }

    pub fn new_zero_one() -> Self {
        Self::new(0.0f32, 1.0f32)
    }

    pub fn new_half_centered() -> Self {
        Self::new(-0.5f32, 0.5f32)
    }
}

impl InitializerLike for UniformInitializer {
    fn initialize_matrix(&self, shape: Shape) -> Matrix {
        Matrix::new_random(shape.0, shape.1)
            .scale(self.upper - self.lower)
            .add_scalar(self.lower)
    }
}
