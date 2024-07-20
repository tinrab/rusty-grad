use enum_dispatch::enum_dispatch;
use identity::IdentityActivationFunction;
use relu::ReLuActivationFunction;
use sigmoid::SigmoidActivationFunction;
use softmax::SoftMaxActivationFunction;
use tanh::TanhActivationFunction;

use crate::math::Matrix;

pub mod identity;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;

#[enum_dispatch]
pub trait PureActivationFunctionLike {
    fn forward(&self, input: &Matrix) -> Matrix;
    fn backward(&self, input: &Matrix) -> Matrix;
}

#[enum_dispatch]
pub trait GradientActivationFunctionLike {
    fn forward(&self, input: &Matrix) -> Matrix;
    fn backward(&self, input: &Matrix, output_gradient: &Matrix) -> Matrix;
}

#[enum_dispatch(PureActivationFunctionLike)]
pub enum PureActivationFunction {
    Identity(IdentityActivationFunction),
    Tanh(TanhActivationFunction),
    ReLu(ReLuActivationFunction),
    Sigmoid(SigmoidActivationFunction),
}

#[enum_dispatch(GradientActivationFunctionLike)]
pub enum GradientActivationFunction {
    SoftMax(SoftMaxActivationFunction),
}

pub enum ActivationFunction {
    Pure(PureActivationFunction),
    Gradient(GradientActivationFunction),
}

macro_rules! impl_from {
    ($from:ty, $to:ident, $variant:ident) => {
        impl From<$from> for $to {
            fn from(from: $from) -> Self {
                Self::$variant(from.into())
            }
        }
    };
}

impl_from!(IdentityActivationFunction, ActivationFunction, Pure);
impl_from!(TanhActivationFunction, ActivationFunction, Pure);
impl_from!(ReLuActivationFunction, ActivationFunction, Pure);
impl_from!(SigmoidActivationFunction, ActivationFunction, Pure);
impl_from!(SoftMaxActivationFunction, ActivationFunction, Gradient);
