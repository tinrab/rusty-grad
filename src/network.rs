use crate::{
    layer::{Layer, LayerLike},
    loss::{LossFunction, LossFunctionLike},
    math::{Matrix, Vector},
    optimizer::Optimizer,
};

pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, input: &Matrix) -> Matrix {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn predict(&mut self, input: Vector) -> Vector {
        let output = self.forward(&Matrix::from_columns(&[input]));
        output.column(0).clone_owned()
    }

    pub fn epoch(
        &mut self,
        epoch: usize,
        input: &[Matrix],
        target: &[Matrix],
        loss_function: &LossFunction,
        optimizer: &mut Optimizer,
    ) -> f32 {
        let mut loss = 0.0;
        for sample in 0..input.len() {
            let output = self.forward(&input[sample]);

            loss += loss_function.forward(&output, &target[sample]);

            let mut error_gradient = loss_function.backward(&output, &target[sample]);
            for layer in self.layers.iter_mut().rev() {
                error_gradient = layer.backward(epoch, &error_gradient, optimizer);
            }
        }

        loss / input.len() as f32
    }

    pub fn train(
        &mut self,
        input: Vec<Matrix>,
        target: Vec<Matrix>,
        epochs: usize,
        loss_function: &LossFunction,
        mut optimizer: Optimizer,
    ) -> f32 {
        let mut loss = f32::INFINITY;
        for epoch in 0..epochs {
            loss = self.epoch(epoch, &input, &target, loss_function, &mut optimizer);
        }
        loss
    }
}
