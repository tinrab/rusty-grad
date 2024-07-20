use crate::{
    activation::{ActivationFunction, GradientActivationFunctionLike, PureActivationFunctionLike},
    layer::LayerLike,
    math::Matrix,
    optimizer::Optimizer,
};

pub struct ActivationLayer {
    activation_function: ActivationFunction,
    input: Option<Matrix>,
    // output: Option<Matrix>,
}

impl ActivationLayer {
    pub fn new<T: Into<ActivationFunction>>(activation_function: T) -> Self {
        Self {
            activation_function: activation_function.into(),
            input: None,
            // output: None,
        }
    }
}

impl LayerLike for ActivationLayer {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.input = Some(input.clone());
        let output = match &self.activation_function {
            ActivationFunction::Pure(activation_function) => activation_function.forward(input),
            ActivationFunction::Gradient(activation_function) => activation_function.forward(input),
        };
        // self.output = Some(output.clone());
        output
    }

    fn backward(
        &mut self,
        _epoch: usize,
        output_gradient: &Matrix,
        _optimizer: &mut Optimizer,
    ) -> Matrix {
        let input = self
            .input
            .as_ref()
            .expect("forward must be called before backward");
        match &self.activation_function {
            ActivationFunction::Pure(activation_function) => {
                // println!("output_gradient: {:?}", output_gradient.shape());
                // println!("input: {:?}", input.shape());
                output_gradient.component_mul(&activation_function.backward(input))
            }
            ActivationFunction::Gradient(activation_function) => {
                activation_function.backward(input, output_gradient)
            }
        }
    }
}
