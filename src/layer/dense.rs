use crate::{
    initializer::{Initializer, InitializerLike},
    layer::LayerLike,
    math::{matrix_broadcast_dot, Matrix},
    optimizer::{Optimizer, OptimizerLike},
};

pub struct DenseLayer {
    weights: Matrix,
    biases: Matrix,
    input: Option<Matrix>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, initializer: &Initializer) -> Self {
        let weights = initializer.initialize_matrix((output_size, input_size));
        let biases = initializer.initialize_matrix((output_size, 1));

        Self {
            weights,
            biases,
            input: None,
        }
    }
}

impl LayerLike for DenseLayer {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.input = Some(input.clone());

        let mut output = matrix_broadcast_dot(&self.weights, input);
        output += &self.biases;

        output
    }

    fn backward(
        &mut self,
        epoch: usize,
        output_gradient: &Matrix,
        optimizer: &mut Optimizer,
    ) -> Matrix {
        let input_gradient = matrix_broadcast_dot(&self.weights.transpose(), output_gradient);
        let weights_gradient =
            matrix_broadcast_dot(output_gradient, &self.input.as_ref().unwrap().transpose());

        self.weights = optimizer.update(epoch, &self.weights, &weights_gradient);
        self.biases = optimizer.update(epoch, &self.biases, output_gradient);

        input_gradient
    }
}

#[cfg(test)]
mod tests {
    use crate::{initializer::ZeroInitializer, optimizer::sgd::SgdOptimizer};

    use super::*;

    #[test]
    fn it_works() {
        let mut d = DenseLayer::new(2, 3, &ZeroInitializer.into());
        let f1 = d.forward(&Matrix::from_row_slice(2, 1, &[2.0, 3.0]));
        println!("{}", f1);
        let mut opt: Optimizer = SgdOptimizer::new(0.1f32).into();
        let f2 = d.backward(0, &Matrix::from_row_slice(3, 1, &[2.0, 3.0, 4.0]), &mut opt);
        println!("{}", f2);
    }
}
