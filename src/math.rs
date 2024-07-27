pub type Vector = nalgebra::OVector<f32, nalgebra::Dyn>;

pub type Matrix = nalgebra::OMatrix<f32, nalgebra::Dyn, nalgebra::Dyn>;

pub type Shape = (usize, usize);

pub fn matrix_broadcast_dot(a: &Matrix, b: &Matrix) -> Matrix {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if (a_shape.0 == 1 || a_shape.1 == 1) && a_shape == b_shape {
        return a.component_mul(b);
    }

    a * b
}

// pub fn matrix_broadcast_component_mul(a: &Matrix, b: &Matrix) -> Matrix {
//     let a_shape = a.shape();
//     let b_shape = b.shape();

//     if a_shape != b_shape && a_shape.0 == b_shape.1 {}

//     a.component_mul(b)
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_broadcast_dot() {
        #[rustfmt::skip]
        assert_eq!(
            matrix_broadcast_dot(
                &Matrix::from_row_slice(1, 3, &[
                    2.0, 2.0, 2.0,
                ]),
                &Matrix::from_row_slice(1, 3, &[
                    2.0, 2.0, 2.0,
                ]),
            ),
            Matrix::from_row_slice(1, 3, &[
                4.0, 4.0, 4.0,
            ]),
        );

        #[rustfmt::skip]
        assert_eq!(
            matrix_broadcast_dot(
                &Matrix::from_row_slice(2, 2, &[
                    1.0, 2.0,
                    3.0, 4.0,
                ]),
                &Matrix::from_row_slice(2, 2, &[
                    1.0, 2.0,
                    3.0, 4.0,
                ]),
            ),
            Matrix::from_row_slice(2, 2, &[
                7.0, 10.0,
                15.0, 22.0,
            ]),
        );

        #[rustfmt::skip]
        assert_eq!(
            matrix_broadcast_dot(
                &Matrix::from_row_slice(2, 3, &[
                    1.0, 2.0, 3.0,
                    4.0, 5.0, 6.0,
                ]),
                &Matrix::from_row_slice(3, 2, &[
                    1.0, 2.0,
                    3.0, 4.0,
                    5.0, 6.0,
                ]),
            ),
            Matrix::from_row_slice(2, 2, &[
                22.0, 28.0,
                49.0, 64.0,
            ]),
        );

        #[rustfmt::skip]
        assert_eq!(
            matrix_broadcast_dot(
                &Matrix::from_row_slice(3, 1, &[
                    2.0,
                    2.0,
                    2.0,
                ]),
                &Matrix::from_row_slice(1, 1, &[
                    3.0,
                ]),
            ),
            Matrix::from_row_slice(3, 1, &[
                6.0,
                6.0,
                6.0,
            ]),
        );

        #[rustfmt::skip]
        assert_eq!(
            matrix_broadcast_dot(
                &Matrix::from_row_slice(3, 2, &[
                    2.0, 2.0,
                    2.0, 2.0,
                    2.0, 2.0,
                ]),
                &Matrix::from_row_slice(2, 1, &[
                    3.0,
                    3.0,
                ]),
            ),
            Matrix::from_row_slice(3, 1, &[
                12.0,
                12.0,
                12.0,
            ]),
        );
    }

    // #[test]
    // fn m_broadcast_mul() {
    //     #[rustfmt::skip]
    //     assert_eq!(
    //         matrix_broadcast_component_mul(
    //             &Matrix::from_row_slice(3, 1, &[
    //                 2.0,
    //                 2.0,
    //                 2.0,
    //             ]),
    //             &Matrix::from_row_slice(3, 1, &[
    //                 3.0,
    //                 3.0,
    //                 3.0,
    //             ]),
    //         ),
    //         Matrix::from_row_slice(3, 1, &[
    //             6.0,
    //             6.0,
    //             6.0,
    //         ]),
    //     );
    // }
}
