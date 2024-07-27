use crate::{
    layer::LayerLike,
    math::{Matrix, Shape},
    optimizer::Optimizer,
};
use nalgebra::Dyn;

pub struct ReshapeLayer {
    input_shape: (Dyn, Dyn),
    output_shape: (Dyn, Dyn),
}

impl ReshapeLayer {
    pub fn new(input_shape: Shape, output_shape: Shape) -> Self {
        Self {
            input_shape: (Dyn(input_shape.0), Dyn(input_shape.1)),
            output_shape: (Dyn(output_shape.0), Dyn(output_shape.1)),
        }
    }
}

impl LayerLike for ReshapeLayer {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        input
            .clone()
            .reshape_generic(self.output_shape.0, self.output_shape.1)
    }

    fn backward(
        &mut self,
        _epoch: usize,
        output_gradient: &Matrix,
        _optimizer: &mut Optimizer,
    ) -> Matrix {
        output_gradient
            .clone()
            .reshape_generic(self.input_shape.0, self.input_shape.1)
    }
}

#[cfg(test)]
mod tests {
    use crate::optimizer::sgd::SgdOptimizer;

    use super::*;

    #[test]
    fn it_works() {
        let mut layer = ReshapeLayer::new((2, 3), (3, 2));

        let input = Matrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = layer.forward(&input);
        assert_eq!(output.shape(), (3, 2));

        let output_gradient = Matrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let input_gradient =
            layer.backward(0, &output_gradient, &mut SgdOptimizer::new(0.1f32).into());
        assert_eq!(input_gradient.shape(), (2, 3));
    }
}
