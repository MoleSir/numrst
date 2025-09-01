mod lu;
mod cholesky;
mod qr;
mod utils;

pub use lu::*;
pub use cholesky::*;
pub use qr::*;

use crate::{linalg, FloatDType, Result, ToMatrixView, ToVectorView, Vector};

/// Enumeration of available methods for solving a linear system `A x = y`.
///
/// # Variants
/// - `LU`: Solve using **LU decomposition**. Suitable for general square matrices.
/// - `Cholesky`: Solve using **Cholesky decomposition**. Requires the matrix to be
///   symmetric and positive-definite.
/// - `Qr`: Solve using **QR decomposition**. Can handle rectangular or rank-deficient matrices.
pub enum SolveMethod {
    LU,
    Cholesky,
    Qr
}

/// Solves the linear system `A x = y` using the specified decomposition method.
///
/// # Description
/// This function provides a unified interface to solve linear systems using different
/// decomposition techniques:
/// 
/// 1. **LU decomposition** (`SolveMethod::LU`) – suitable for general square matrices.  
/// 2. **Cholesky decomposition** (`SolveMethod::Cholesky`) – requires `A` to be symmetric
///    and positive-definite.  
/// 3. **QR decomposition** (`SolveMethod::Qr`) – can handle rectangular matrices and
///    least-squares solutions.
///
/// # Type Parameters
/// - `T`: Floating-point numeric type. Must implement `FloatDType`.
/// - `M`: Type that can be converted to a matrix view (`ToMatrixView<T>`).
/// - `V`: Type that can be converted to a vector view (`ToVectorView<T>`).
///
/// # Parameters
/// - `a`: The coefficient matrix `A` of the system.
/// - `y`: The right-hand side vector `y`.
/// - `method`: The decomposition method to use (`LU`, `Cholesky`, or `Qr`).
///
/// # Returns
/// - `Ok(x)` where `x` is the solution vector such that `A * x ≈ y`.
/// - Returns an error if:
///     - The matrix is singular (for LU or Cholesky).  
///     - The matrix is not square when required (LU/Cholesky).  
///     - The matrix is not positive-definite for Cholesky.  
///     - Other decomposition or numerical errors occur.
///
/// # Example
/// ```rust
/// # use numrst::{linalg, NdArray};
/// let a = NdArray::new(&[
///     [3.0, 1.0],
///     [1.0, 2.0]
/// ]).unwrap();
/// let y = NdArray::from_vec([9.0, 8.0].to_vec(), 2).unwrap();
///
/// let x = linalg::solve(&a, &y, linalg::SolveMethod::LU).unwrap();
/// // x ≈ [2.0, 3.0]
/// ```
pub fn solve<T, M, V>(a: M, y: V, method: SolveMethod) -> Result<Vector<T>>
where
    T: FloatDType,
    M: ToMatrixView<T>,
    V: ToVectorView<T>,
{
    match method {
        SolveMethod::LU => linalg::lu_solve(a, y),
        SolveMethod::Cholesky => linalg::cholesky_solve(a, y),
        SolveMethod::Qr => linalg::qr_solve(a, y),
    }
}

#[cfg(test)]
mod test {
    use crate::{linalg, NdArray};

    #[test]
    fn test_solve_compare_methods() {
        let a = NdArray::new(&[
            [4., 2., 0.],
            [2., 5., 1.],
            [0., 1., 3.],
        ]).unwrap();

        let y = NdArray::from_vec([2., 1., 3.].to_vec(), 3).unwrap();

        let x_lu = linalg::solve(&a, &y, linalg::SolveMethod::LU).unwrap();
        let x_chol = linalg::solve(&a, &y, linalg::SolveMethod::Cholesky).unwrap();
        let x_qr = linalg::solve(&a, &y, linalg::SolveMethod::Qr).unwrap();

        let x_lu_arr = x_lu.to_ndarray();
        let x_chol_arr = x_chol.to_ndarray();
        let x_qr_arr = x_qr.to_ndarray();

        assert!(x_lu_arr.allclose(&x_chol_arr, 1e-6, 1e-6));
        assert!(x_lu_arr.allclose(&x_qr_arr, 1e-6, 1e-6));
        assert!(x_chol_arr.allclose(&x_qr_arr, 1e-6, 1e-6));
    }

    #[test]
    fn test_big_solve() {
        let a = NdArray::<f64>::randn(0.0, 1.0, (7, 7)).unwrap();
        let y = NdArray::<f64>::randn(0.0, 1.0, (7,)).unwrap();
        let x = linalg::solve(&a, &y, linalg::SolveMethod::Qr).unwrap().to_ndarray();
        
        let execpt = a.matmul(&x.unsqueeze(1).unwrap()).unwrap().squeeze(1).unwrap();
        assert!(execpt.allclose(&y, 1e-5, 1e-5));
    }
}
