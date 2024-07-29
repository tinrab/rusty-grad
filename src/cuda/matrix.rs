use nalgebra::{Dyn, VecStorage};

use crate::math::Matrix;

#[repr(C)]
pub struct CudaMatrix {
    rows: usize,
    columns: usize,
    data: *mut f32,
}

impl CudaMatrix {
    pub fn len(&self) -> usize {
        self.rows * self.columns
    }
}

impl From<Matrix> for CudaMatrix {
    fn from(matrix: Matrix) -> Self {
        let (rows, columns) = matrix.shape();
        // TODO: Can clone be avoided?
        let data = matrix.data.as_vec().clone();
        Self {
            rows,
            columns,
            data: data.leak().as_mut_ptr(),
        }
    }
}

impl From<CudaMatrix> for Matrix {
    fn from(matrix: CudaMatrix) -> Self {
        let len = matrix.len();
        let data = unsafe { Vec::from_raw_parts(matrix.data, len, len) };
        Matrix::from_data(VecStorage::new(Dyn(matrix.rows), Dyn(matrix.columns), data))
    }
}

impl Drop for CudaMatrix {
    fn drop(&mut self) {
        let len = self.len();
        unsafe {
            Vec::from_raw_parts(self.data, len, len);
        }
    }
}
