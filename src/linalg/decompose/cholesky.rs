use crate::{linalg::LinalgError, FloatDType, NdArray, Result};

/// Computes the Cholesky decomposition of a matrix `A`.
///
/// # Description
/// Cholesky decomposition factorizes a symmetric positive-definite matrix `A`
/// into a lower triangular matrix `L` such that `A = L * L^T`.
///
/// This implementation only computes the lower triangular factor `L`.
/// The upper triangular factor can be obtained as `L^T`.
///
/// # Type Parameters
/// - `T`: The floating-point data type. Must implement `FloatDType`.
///
/// # Parameters
/// - `arr`: The input matrix `A` to decompose. Must be symmetric and positive-definite.
///
/// # Returns
/// - `L`: where `L` is a lower triangular matrix such that `A = L * L^T`.
///
/// # Notes
/// - Input matrix `A` must be square (`n x n`). Non-square matrices are invalid.
/// - Symmetry of `A` is assumed; this implementation does not explicitly check symmetry.
/// - The decomposition fails immediately if a non-positive pivot is encountered.
/// - This function is useful for solving linear systems `A x = y` when `A` is symmetric
///   and positive-definite, and for generating random samples with a multivariate
///   normal distribution.
///
/// # Example
/// ```rust
/// # use numrst::{linalg, NdArray};
/// let a = NdArray::new(&[
///     [4.0, 12.0, -16.0],
///     [12.0, 37.0, -43.0],
///     [-16.0, -43.0, 98.0],
/// ]).unwrap();
/// let l = linalg::cholesky(&a).unwrap();
/// // Now a ≈ L * L^T
/// ```
pub fn cholesky<T: FloatDType>(arr: &NdArray<T>) -> Result<NdArray<T>> {
    let mat = arr.matrix_view()?;
    let (n, _) = mat.shape();
    let l_arr = NdArray::<T>::zeros(mat.shape())?;

    {
        let mut l = l_arr.matrix_view_mut().unwrap();
        for i in 0..n {
            // diag
            let mut sum = T::zero();
            for k in 0..i {
                sum = sum + l.g(i, k) * l.g(i, k);
            }
            let diag = mat.g(i, i) - sum;
            if diag <= T::zero() {
                return Err(LinalgError::ExpectPositiveDefiniteMatrix { op: "cholesky" })?;
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
    
    }

    Ok(l_arr)
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
        // println!("{}", l);

        let a_rec = l.matmul(&l.transpose_last().unwrap()).unwrap();
        assert!(a_rec.allclose(&a, 1e-6, 1e-6));
    }

    #[test]
    fn test_cholesky_identity() {
        let a = NdArray::<f64>::eye(4).unwrap();
        let l = linalg::cholesky(&a).unwrap();
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

        let l = linalg::cholesky(&a).unwrap();
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
        // println!("{}", l);
        let h_rec = l.matmul(&l.transpose_last().unwrap()).unwrap();
        assert!(h_rec.allclose(&h, 1e-6, 1e-6));
    }
}
