use crate::{FloatDType, Result, ToMatrixView, ToVectorView, Vector};
use crate::linalg;
use super::utils;

/// Solve A x = y using Cholesky (A must be SPD)
pub fn cholesky_solve<T, M, V>(a: M, y: V) -> Result<Vector<T>>
where
    T: FloatDType,
    M: ToMatrixView<T>,
    V: ToVectorView<T>,
{
    let (a, y) = utils::check_solve_arg(&a, &y)?;

    // A = L @ L.T
    // L @ L.T @ x = y
    // 1: L @ z = y
    // 2: L.T @ x = z
    let l = linalg::cholesky(a)?;
    let l = l.to_matrix_view()?;
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
