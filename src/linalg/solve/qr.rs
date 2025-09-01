use crate::{linalg, FloatDType, Result, ToMatrixView, ToVectorView, Vector};

use super::utils;

/// Solve linear equation A x = y using QR decomposition
pub fn qr_solve<T, M, V>(a: M, y: V) -> Result<Vector<T>> 
where
    T: FloatDType,
    M: ToMatrixView<T>,
    V: ToVectorView<T>,
{
    let (a, y) = utils::check_solve_arg(&a, &y)?;

    let (q, r) = linalg::qr(a)?;             // Step 1: QR decomposition
    let q = q.to_matrix_view()?;                // Q: (m, m)
    let r = r.to_matrix_view()?;                // R: (m, n)
    
    let n = r.shape().1;                        // number of unknowns
    let rhs = Vector::<T>::zeros(n)?;       // rhs = Q^T * y

    // Step 2: compute Q^T * y
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..y.len() {
            sum += q.g(j, i) * y.g(j);
        }
        rhs.s(i, sum);
    }

    // Step 3: backward substitution R x = rhs
    let x = Vector::<T>::zeros(n)?;
    for i in (0..n).rev() {
        let mut sum = T::zero();
        for j in i+1..n {
            sum += r.g(i, j) * x.g(j);
        }
        x.s(i, (rhs.g(i) - sum) / r.g(i, i));
    }

    Ok(x)
}

#[cfg(test)]
mod test {
    use crate::{NdArray, linalg};

    #[test]
    fn test_qr_solve_simple() {
        // Ax = y, A = [[3, 1], [1, 2]], y = [9, 8], solution x = [2, 3]
        let a = NdArray::new(&[
            [3., 1.],
            [1., 2.],
        ]).unwrap();

        let y = NdArray::from_vec([9., 8.].to_vec(), 2).unwrap();

        let x = linalg::qr_solve(&a, &y).unwrap();
        let expected = NdArray::from_vec([2., 3.].to_vec(), 2).unwrap();

        assert!(x.to_ndarray().allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_qr_solve_identity() {
        // Identity matrix, solution should equal y
        let a = NdArray::<f64>::eye(4).unwrap();
        let y = NdArray::from_vec([1., 2., 3., 4.].to_vec(), 4).unwrap();

        let x = linalg::qr_solve(&a, &y).unwrap();
        assert!(x.to_ndarray().allclose(&y, 1e-12, 1e-12));
    }
}
