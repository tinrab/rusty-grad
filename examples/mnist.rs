use rusty_neuron::{
    activation::tanh::TanhActivationFunction,
    initializer::{Initializer, UniformInitializer},
    layer::{activation::ActivationLayer, dense::DenseLayer},
    loss::{LossFunction, MseLossFunction},
    math::Matrix,
    network::Network,
    optimizer::{sgd::SgdOptimizer, Optimizer},
};
use std::{error::Error, fs::File, io::Read};

const TRAIN_IMAGES: &str = "./examples/data/mnist/train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "./examples/data/mnist/train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "./examples/data/mnist/t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "./examples/data/mnist/t10k-labels-idx1-ubyte";

fn main() -> Result<(), Box<dyn Error>> {
    // let labels_filename = "./examples/data/mnist/train-labels-idx1-ubyte";
    // let images_filename = "./examples/data/mnist/train-images-idx3-ubyte";

    // // print labels
    // let mut file = File::open(labels_filename)?;
    // let mut buffer = Vec::new();
    // file.read_to_end(&mut buffer)?;
    // let labels = buffer.chunks(1).map(|chunk| chunk[0]).collect::<Vec<u8>>();

    // // image data to png
    // let mut file = File::open(images_filename)?;
    // let mut buffer = Vec::new();
    // file.read_to_end(&mut buffer)?;

    // let size = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as usize;
    // let images = buffer[16..].chunks(28 * 28).collect::<Vec<&[u8]>>();

    // for (idx, image) in images.iter().enumerate() {
    //     let mut img = image::ImageBuffer::new(28, 28);
    //     for (i, pixel) in image.iter().enumerate() {
    //         let x = i % 28;
    //         let y = i / 28;
    //         img.put_pixel(x as u32, y as u32, image::Rgb([*pixel, *pixel, *pixel]));
    //     }
    //     img.save(format!("./examples/data/mnist/{}.png", idx))?;
    // }

    const TRAIN_SIZE: usize = 1000;
    let mut input: Vec<Matrix> = Vec::new();
    let mut file = File::open(TRAIN_IMAGES)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    for image_buffer in buffer[16..].chunks(28 * 28).take(TRAIN_SIZE) {
        let mut data = Vec::with_capacity(28 * 28);
        for pixel in image_buffer {
            data.push(*pixel as f32 / 255.0);
        }
        input.push(Matrix::from_row_slice(28 * 28, 1, &data));
    }

    let mut target: Vec<Matrix> = Vec::new();
    let mut file = File::open(TRAIN_LABELS)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    for label in buffer[8..].chunks(1).take(TRAIN_SIZE) {
        let mut data = vec![0.0; 10];
        data[label[0] as usize] = 1.0;
        target.push(Matrix::from_row_slice(10, 1, &data));
    }

    // for i in 0..30 {
    //     let img = matrix_to_image(&input[30 + i]);
    //     img.save(format!("./examples/data/mnist/images/{}.png", i))?;
    // }

    let initializer: Initializer = UniformInitializer::new_half_centered().into();
    let mut optimizer: Optimizer = SgdOptimizer::new(0.01f32).into();
    let loss_function: LossFunction = MseLossFunction::new().into();

    let mut nn = Network::new(vec![
        // ReshapeLayer::new((28, 28), (28 * 28, 1)).into(),
        DenseLayer::new(28 * 28, 40, &initializer).into(),
        ActivationLayer::new(TanhActivationFunction::new()).into(),
        DenseLayer::new(40, 10, &initializer).into(),
        ActivationLayer::new(TanhActivationFunction::new()).into(),
    ]);

    const EPOCHS: usize = 10;
    for epoch in 0..EPOCHS {
        let loss = nn.epoch(epoch, &input, &target, &loss_function, &mut optimizer);
        if epoch % 3 == 0 || epoch == EPOCHS - 1 {
            println!("Epoch: {}, Loss: {}", epoch, loss);
        }
    }

    Ok(())
}

fn matrix_to_image(matrix: &Matrix) -> image::RgbImage {
    let mut img = image::ImageBuffer::new(28, 28);
    for (i, pixel) in matrix.iter().enumerate() {
        let x = i / 28;
        let y = i % 28;
        let pixel = (pixel * 255.0) as u8;
        img.put_pixel(x as u32, y as u32, image::Rgb([pixel, pixel, pixel]));
    }
    img
}
