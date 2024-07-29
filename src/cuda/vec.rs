#[repr(C)]
pub struct CudaVec<T> {
    len: usize,
    data: *mut T,
}

impl<T> Drop for CudaVec<T> {
    fn drop(&mut self) {
        unsafe {
            Vec::from_raw_parts(self.data, self.len, self.len);
        }
    }
}

impl<T> From<Vec<T>> for CudaVec<T> {
    fn from(vec: Vec<T>) -> Self {
        let vec = vec.leak();
        Self {
            len: vec.len(),
            data: vec.as_mut_ptr(),
        }
    }
}
