use crate::{activation::GradientActivationFunctionLike, math::Matrix};
use std::cell::Cell;

pub struct SoftMaxActivationFunction {
    output: Cell<Matrix>,
}

impl SoftMaxActivationFunction {
    pub fn new() -> Self {
        Self {
            output: Cell::default(),
        }
    }
}

fn softmax(mut input: Matrix) -> Matrix {
    for mut row in input.row_iter_mut() {
        // avoid imprecision by subtracting the maximum
        let max = row.max();
        row.iter_mut().for_each(|x| *x = (*x - max).exp());
        let sum = row.sum();
        row.iter_mut().for_each(|x| *x /= sum);
    }
    input.clone()
}

fn softmax_derivative(input: &Matrix) -> Matrix {
    let s = softmax(input.clone());
    let (row_count, column_count) = input.shape();

    let mut output = Matrix::zeros(row_count, column_count);

    for i in 0..row_count {
        for j in 0..column_count {
            let s_ij = s[(i, j)];
            output[(i, j)] = s_ij * (1.0 - s_ij);
        }
    }

    output
}

impl GradientActivationFunctionLike for SoftMaxActivationFunction {
    fn forward(&self, input: &Matrix) -> Matrix {
        let output = softmax(input.clone());
        self.output.set(output.clone());
        output
    }

    fn backward(&self, input: &Matrix, output_gradient: &Matrix) -> Matrix {
        let output = self.output.take();
        self.output.set(output.clone());

        softmax_derivative(input)
    }
}
