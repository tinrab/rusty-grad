use super::matrix::CudaMatrix;
use crate::{
    layer::{dense::DenseLayer, Layer},
    math::Matrix,
};

#[repr(C)]
pub struct CudaLayer {
    dense: *mut CudaDenseLayer,
}

#[repr(C)]
pub struct CudaDenseLayer {
    input_size: usize,
    output_size: usize,
    weights: CudaMatrix,
    biases: CudaMatrix,
}

impl From<Layer> for CudaLayer {
    fn from(layer: Layer) -> Self {
        match layer {
            Layer::Dense(layer) => layer.into(),
            _ => unimplemented!(),
        }
    }
}

impl From<DenseLayer> for CudaLayer {
    fn from(layer: DenseLayer) -> Self {
        Self {
            dense: Box::leak(Box::new(layer.into())),
        }
    }
}

impl From<DenseLayer> for CudaDenseLayer {
    fn from(layer: DenseLayer) -> Self {
        Self {
            input_size: layer.input_size,
            output_size: layer.output_size,
            weights: layer.weights.into(),
            biases: layer.biases.into(),
        }
    }
}
