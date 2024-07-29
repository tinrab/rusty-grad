use crate::{layer::Layer, math::Matrix, network::Network};
use layer::CudaLayer;
use matrix::CudaMatrix;
use vec::CudaVec;

mod layer;
mod matrix;
mod vec;

#[repr(C)]
struct CudaNetwork {
    layers: CudaVec<CudaLayer>,
}

#[link(name = "ai", kind = "static")]
extern "C" {
    #[allow(improper_ctypes)]
    fn network_forward(network: *mut CudaNetwork, input: CudaMatrix) -> CudaMatrix;
}

pub fn cuda_network_forward(network: Network, input: Matrix) -> Matrix {
    let cuda_layers: Vec<CudaLayer> = network.layers.into_iter().map(Into::into).collect();

    let mut cuda_network = CudaNetwork {
        layers: cuda_layers.into(),
    };

    // let mut output: CudaMatrix = Matrix::repeat(4, 1, 0.0f32).into();

    unsafe { network_forward(&mut cuda_network, input.into()).into() }
}
