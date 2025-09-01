use crate::{Error, FloatDType, Matrix, Result, ToMatrixView, ToVectorView, Vector};
use crate::linalg;

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

/// Solve A x = y using Cholesky (A must be SPD)
pub fn cholesky_solve<T, M, V>(a: M, y: V) -> Result<Vector<T>>
where
    T: FloatDType,
    M: ToMatrixView<T>,
    V: ToVectorView<T>,
{
    // A = L @ L.T
    // L @ L.T @ x = y
    // 1: L @ z = y
    // 2: L.T @ x = z
    let l = linalg::cholesky(a)?;
    let l = l.to_matrix_view()?;
    let y = y.to_vector_view()?;
    let n = y.len();

    // Step 1: forward solve L z = y
    let z = Vector::<T>::zeros(n)?;
    for i in 0..n {
        let mut sum = T::zero();
        for k in 0..i {
            sum = sum + l.g(i, k) * z.g(k);
        }
        z.s(i, (y.g(i) - sum) / l.g(i, i));
    }

    // Step 2: backward solve L^T x = z
    let x = Vector::<T>::zeros(n)?;
    for i in (0..n).rev() {
        let mut sum = T::zero();
        for k in i+1..n {
            sum = sum + l.g(k, i) * x.g(k);
        }
        x.s(i, (z.g(i) - sum) / l.g(i, i));
    }

    Ok(x)
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

    
    #[test]
    fn test_cholesky_solve_simple() {
        let a = NdArray::new(&[
            [4., 1., 2.],
            [1., 2., 0.],
            [2., 0., 3.],
        ]).unwrap();
        let y = NdArray::from_vec([9., 3., 8.].to_vec(), 3).unwrap();

        let x = linalg::cholesky_solve(&a, &y).unwrap();
        let expected = NdArray::from_vec([1., 1., 2.].to_vec(), 3).unwrap();

        assert!(x.to_ndarray().allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_cholesky_solve_identity() {
        let a = NdArray::<f64>::eye(4).unwrap();
        let y = NdArray::from_vec([1., 2., 3., 4.].to_vec(), 4).unwrap();

        let x = linalg::cholesky_solve(&a, &y).unwrap();
        assert!(x.to_ndarray().allclose(&y, 1e-6, 1e-6));
    }
}
