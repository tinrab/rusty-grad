#[repr(C)]
pub struct CudaTensor {
    pub data: *mut f32,
    pub shape: Vec<usize>,
}
