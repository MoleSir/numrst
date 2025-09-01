use crate::{Error, FloatDType, Matrix, Result, ToMatrixView};

pub fn cholesky<T: FloatDType, M: ToMatrixView<T>>(mat: M) -> Result<Matrix<T>> {
    let mat = mat.to_matrix_view()?;
    let (n, _) = mat.shape();
    let l = Matrix::<T>::zeros(n, n)?;

    for i in 0..n {
        // diag
        let mut sum = T::zero();
        for k in 0..i {
            sum = sum + l.g(i, k) * l.g(i, k);
        }
        let diag = mat.g(i, i) - sum;
        if diag <= T::zero() {
            return Err(Error::Msg("Matrix is not positive definite".into()));
        }
        l.s(i, i, diag.sqrt());

        // below diag
        for j in i+1..n {
            let mut sum = T::zero();
            for k in 0..i {
                sum = sum + l.g(j, k) * l.g(i, k);
            }
            l.s(j, i, (mat.g(j, i) - sum) / l.g(i, i));
        }
    }

    Ok(l)
}

#[cfg(test)]
mod test {
    use crate::NdArray;
    use crate::linalg;

    #[test]
    fn test_cholesky_simple() {
        let a = NdArray::new(&[
            [4., 12., -16.],
            [12., 37., -43.],
            [-16., -43., 98.],
        ]).unwrap();

        let l = linalg::cholesky(&a).unwrap();
        let l = l.to_ndarray();
        // println!("{}", l);

        let a_rec = l.matmul(&l.transpose_last().unwrap()).unwrap();
        assert!(a_rec.allclose(&a, 1e-6, 1e-6));
    }

    #[test]
    fn test_cholesky_identity() {
        let a = NdArray::<f64>::eye(4).unwrap();
        let l = linalg::cholesky(&a).unwrap().to_ndarray();
        // println!("{}", l);
        let rec = l.matmul(&l.transpose_last().unwrap()).unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));
    }

    #[test]
    fn test_cholesky_random_pos_def() {
        let b = NdArray::new(&[
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 10.],
        ]).unwrap();
        let a = b.matmul(&b.transpose_last().unwrap()).unwrap();

        let l = linalg::cholesky(&a).unwrap().to_ndarray();
        // println!("{}", l);
        let a_rec = l.matmul(&l.transpose_last().unwrap()).unwrap();
        assert!(a_rec.allclose(&a, 1e-6, 1e-6));
    }

    #[test]
    #[should_panic]
    fn test_cholesky_non_pos_def() {
        let a = NdArray::new(&[
            [1., 2.],
            [2., 1.],
        ]).unwrap();

        linalg::cholesky(&a).unwrap();
    }

    #[test]
    fn test_cholesky_high_dim() {
        let a = NdArray::<f64>::randn(0., 1., (10, 10)).unwrap();
        // h = a @ a.T
        let h = a.matmul(&a.transpose_last().unwrap()).unwrap();
        
        let l = linalg::cholesky(&h).unwrap();
        let l = l.to_ndarray();
        // println!("{}", l);
        let h_rec = l.matmul(&l.transpose_last().unwrap()).unwrap();
        assert!(h_rec.allclose(&h, 1e-6, 1e-6));
    }
}
