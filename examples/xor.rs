use rusty_neuron::{
    activation::tanh::TanhActivationFunction,
    initializer::{Initializer, UniformInitializer},
    layer::{activation::ActivationLayer, dense::DenseLayer},
    loss::{LossFunction, MseLossFunction},
    math::Matrix,
    network::Network,
    optimizer::{sgd::SgdOptimizer, Optimizer},
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // xor dataset
    let input = vec![
        Matrix::from_row_slice(2, 1, &[0.0, 0.0]),
        Matrix::from_row_slice(2, 1, &[0.0, 1.0]),
        Matrix::from_row_slice(2, 1, &[1.0, 0.0]),
        Matrix::from_row_slice(2, 1, &[1.0, 1.0]),
    ];
    let target = vec![
        Matrix::from_row_slice(1, 1, &[0.0]),
        Matrix::from_row_slice(1, 1, &[1.0]),
        Matrix::from_row_slice(1, 1, &[1.0]),
        Matrix::from_row_slice(1, 1, &[0.0]),
    ];

    let initializer: Initializer = UniformInitializer::new_half_centered().into();
    let mut optimizer: Optimizer = SgdOptimizer::new(0.1f32).into();
    let loss_function: LossFunction = MseLossFunction::new().into();

    let mut nn = Network::new(vec![
        DenseLayer::new(2, 3, &initializer).into(),
        ActivationLayer::new(TanhActivationFunction::new()).into(),
        DenseLayer::new(3, 1, &initializer).into(),
        ActivationLayer::new(TanhActivationFunction::new()).into(),
    ]);

    const EPOCHS: usize = 1000;
    for epoch in 0..EPOCHS {
        let mut loss = 0.0;
        for sample in 0..input.len() {
            loss += nn.epoch(
                epoch,
                &loss_function,
                &input[sample],
                &target[sample],
                &mut optimizer,
            );
        }

        loss /= input.len() as f32;
        if epoch % 100 == 0 || epoch == EPOCHS - 1 {
            println!("Epoch: {}, Loss: {}", epoch, loss);
        }
    }

    Ok(())
}
