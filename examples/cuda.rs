use image::{GenericImageView, ImageReader};
use nalgebra::{Dyn, RowVector3};
use rusty_grad::{
    activation::{
        sigmoid::SigmoidActivationFunction, softmax::SoftMaxActivationFunction,
        tanh::TanhActivationFunction, PureActivationFunction, PureActivationFunctionLike,
    },
    cuda::cuda_network_forward,
    initializer::{Initializer, UniformInitializer},
    layer::{
        activation::ActivationLayer, dense::DenseLayer, reshape::ReshapeLayer, Layer, LayerLike,
    },
    loss::{LossFunction, LossFunctionLike, MseLossFunction},
    math::Matrix,
    network::Network,
    optimizer::{self, sgd::SgdOptimizer, Optimizer},
};
use std::{error::Error, fs::File, io::Read, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let initializer: Initializer = UniformInitializer::new_half_centered().into();
    let mut network = Network::new(vec![
        DenseLayer::new(8, 1, &initializer).into(),
        // DenseLayer::new(64, 2048, &initializer).into(),
        // ActivationLayer::new(TanhActivationFunction::new()).into(),
        // DenseLayer::new(2048, 2048, &initializer).into(),
        // DenseLayer::new(2048, 2048, &initializer).into(),
        // DenseLayer::new(2048, 2048, &initializer).into(),
        // DenseLayer::new(2048, 2048, &initializer).into(),
        // ActivationLayer::new(SigmoidActivationFunction::new()).into(),
        // DenseLayer::new(2048, 1, &initializer).into(),
        // ActivationLayer::new(TanhActivationFunction::new()).into(),
    ]);

    // let input = Matrix::new_random(64, 1);
    let input = Matrix::from_row_slice(
        8,
        1,
        &[
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32,
        ],
    );

    // let now = Instant::now();
    // for _ in 0..5 {
    //     network.forward(&input);
    // }
    // println!("CPU: {:?}", now.elapsed());

    let output = cuda_network_forward(network, input);
    println!("{}", output);

    Ok(())
}
