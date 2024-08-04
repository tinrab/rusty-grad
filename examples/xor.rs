use rusty_grad::{
    activation::sigmoid::SigmoidActivationFunction,
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

    let initializer: Initializer = UniformInitializer::new_zero_one().into();
    let mut optimizer: Optimizer = SgdOptimizer::new(0.01f32).into();
    let loss_function: LossFunction = MseLossFunction::new().into();

    let mut nn = Network::new(vec![
        DenseLayer::new(2, 3, &initializer).into(),
        ActivationLayer::new(SigmoidActivationFunction::new()).into(),
        DenseLayer::new(3, 1, &initializer).into(),
        ActivationLayer::new(SigmoidActivationFunction::new()).into(),
    ]);

    const EPOCHS: usize = 1000;
    for epoch in 0..EPOCHS {
        let loss = nn.epoch(epoch, &input, &target, &loss_function, &mut optimizer);
        if epoch % 200 == 0 || epoch == EPOCHS - 1 {
            println!("Epoch: {}, Loss: {}", epoch, loss);
        }
    }

    Ok(())
}
