use crate::{
    layer::LayerLike,
    math::{matrix_broadcast_dot, Matrix},
    optimizer::{Optimizer, OptimizerLike},
};

pub struct DenseLayer {
    weights: Matrix,
    biases: Matrix,
    input: Option<Matrix>,
    // output: Option<Matrix>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Matrix::new_random(output_size, input_size).add_scalar(-0.5f32);
        let biases = Matrix::new_random(output_size, 1).add_scalar(-0.5f32);
        // let weights = Matrix::new_random(output_size, input_size);
        // let biases = Matrix::new_random(output_size, 1);

        // let weights = Matrix::repeat(output_size, input_size, 1.0f32);
        // let biases = Matrix::repeat(output_size, 1, 1.0f32);

        Self {
            weights,
            biases,
            input: None,
            // output: None,
        }
    }
}

impl LayerLike for DenseLayer {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        self.input = Some(input.clone());
        // self.output = Some(output.clone());

        // println!("input: {:?}", input.shape());
        // println!("weights: {:?}", self.weights.shape());
        let mut output = matrix_broadcast_dot(&self.weights, input);

        // let biases = self.biases.transpose();
        // for mut row in output.row_iter_mut() {
        //     row += &biases;
        // }
        output += &self.biases;

        output
    }

    fn backward(
        &mut self,
        epoch: usize,
        output_gradient: &Matrix,
        optimizer: &mut Optimizer,
    ) -> Matrix {
        // let input_gradient = output_gradient * self.weights.transpose();
        // let weights_gradient = self.input.clone().unwrap().transpose() * output_gradient;
        // println!("weights: {:?}", self.weights.transpose().shape());
        // println!("{}", &self.weights);
        // println!("output_gradient: {:?}", output_gradient.shape());

        // let input_gradient = matrix_broadcast_dot(&self.weights.transpose(), output_gradient);
        let input_gradient = matrix_broadcast_dot(&self.weights.transpose(), output_gradient);
        let weights_gradient =
            matrix_broadcast_dot(output_gradient, &self.input.as_ref().unwrap().transpose());

        // println!("output_gradient: {:?}", output_gradient.shape());
        // println!("input: {:?}", input.shape());

        // self.weights = optimizer.update(epoch, &self.weights, &weights_gradient);
        // self.biases = optimizer.update(epoch, &self.biases, output_gradient);

        let lr = 0.1f32;
        self.weights -= lr * weights_gradient;
        self.biases -= lr * output_gradient;

        input_gradient
    }
}

#[cfg(test)]
mod tests {
    use crate::optimizer::sgd::SgdOptimizer;

    use super::*;

    #[test]
    fn it_works() {
        let mut d = DenseLayer::new(2, 3);
        let f1 = d.forward(&Matrix::from_row_slice(2, 1, &[2.0, 3.0]));
        println!("{}", f1);
        let mut opt: Optimizer = SgdOptimizer::new(0.1f32).into();
        let f2 = d.backward(0, &Matrix::from_row_slice(3, 1, &[2.0, 3.0, 4.0]), &mut opt);
        println!("{}", f2);
    }
}
