use crate::{
    layer::{Layer, LayerLike},
    loss::{LossFunction, LossFunctionLike},
    math::{Matrix, Vector},
    optimizer::Optimizer,
};

pub struct Network {
    layers: Vec<Layer>,
    loss_function: LossFunction,
}

impl Network {
    pub fn new(layers: Vec<Layer>, loss_function: LossFunction) -> Self {
        Self {
            layers,
            loss_function,
        }
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

    // pub fn predict(&mut self, input: Vec<Matrix>) -> Vec<Matrix> {
    //     let mut result = Vec::with_capacity(input.len());

    //     for sample in input {
    //         let mut output = sample;
    //         for layer in &mut self.layers {
    //             output = layer.forward(output);
    //         }
    //         result.push(output);
    //     }

    //     result
    // }

    pub fn train(
        &mut self,
        input: Vec<Matrix>,
        target: Vec<Matrix>,
        epochs: usize,
        mut optimizer: Optimizer,
    ) {
        for epoch in 0..epochs {
            let mut loss = 0.0f32;
            for sample in 0..input.len() {
                let output = self.forward(&input[sample]);

                loss += self.loss_function.forward(&output, &target[sample]);

                let mut error_gradient = self.loss_function.backward(&output, &target[sample]);
                for layer in self.layers.iter_mut().rev() {
                    error_gradient = layer.backward(epoch, &error_gradient, &mut optimizer);
                }
            }

            loss /= input.len() as f32;
            if epoch % 100 == 0 {
                println!("Epoch: {}, Loss: {}", epoch, loss);
            }
        }
    }
}
