use std::error::Error;

use nalgebra::{Dyn, RowVector3};
use rusty_neuron::{
    activation::{
        softmax::SoftMaxActivationFunction, tanh::TanhActivationFunction, PureActivationFunction,
        PureActivationFunctionLike,
    },
    layer::{activation::ActivationLayer, dense::DenseLayer, Layer, LayerLike},
    loss::{LossFunctionLike, MseLossFunction},
    math::Matrix,
    network::Network,
    optimizer::{self, sgd::SgdOptimizer, Optimizer},
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Hello, Rusty Neuron!");

    {
        // let input = Matrix::from_row_slice(1, 3, &[1.0, 0.5, -1.0]);
        // let target = Matrix::from_row_slice(1, 3, &[0.0, 0.5, 1.0]);
        // #[rustfmt::skip]
        // let input = Matrix::from_row_slice(2, 3, &[
        //     1.0, 2.0, 3.0,
        //     4.0, 5.0, 6.0,
        // ]);
        // #[rustfmt::skip]
        // let target = Matrix::from_row_slice(2, 3, &[
        //     0.0, 0.5, 1.0,
        //     1.0, 0.5, 0.0,
        // ]);

        // println!("mse loss");
        // let mse_loss = MseLossFunction::new();
        // println!("{}", MseLossFunction::new().forward(&input, &target));
        // println!("{}", MseLossFunction::new().backward(&input, &target));

        // println!("tanh activation");
        // let mut tanh_layer = ActivationLayer::new(TanhActivationFunction::new());
        // println!("{}", tanh_layer.forward(&input));
        // println!("{}", tanh_layer.backward(&input, 0.1));

        // println!("sigmoid activation");
        // let mut sigmoid_layer = ActivationLayer::new(SigmoidActivationFunction::new());
        // println!("{}", sigmoid_layer.forward(&input));
        // println!("{}", sigmoid_layer.backward(&input, 0.1));

        // println!("softmax activation");
        // let mut sgd_optimizer: Optimizer = SgdOptimizer::new(0.1f32).into();
        // let mut sm: Layer = ActivationLayer::new(SoftMaxActivationFunction::new()).into();
        // println!("{}", sm.forward(&input));
        // println!("{}", sm.backward(0, &input, &mut sgd_optimizer));

        // println!("dense");
        // let mut dense_layer = DenseLayer::new(3, 3);
        // let mut sgd_optimizer: Optimizer = SgdOptimizer::new(0.1f32).into();
        // println!("{}", dense_layer.forward(&input));
        // println!(
        //     "{}",
        //     dense_layer.backward(0, &mse_loss.backward(&input, &target), &mut sgd_optimizer)
        // );
        // println!("{}", dense_layer.forward(&input));
    }

    let mut nn = Network::new(
        vec![
            DenseLayer::new(2, 3).into(),
            ActivationLayer::new(TanhActivationFunction::new()).into(),
            DenseLayer::new(3, 1).into(),
            ActivationLayer::new(TanhActivationFunction::new()).into(),
        ],
        MseLossFunction::new().into(),
    );

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

    let mut optimizer: Optimizer = SgdOptimizer::new(0.1f32).into();
    nn.train(input.clone(), target, 1000, optimizer);

    // println!("prediction: {}", nn.predict(input.clone())[0]);

    Ok(())
}
