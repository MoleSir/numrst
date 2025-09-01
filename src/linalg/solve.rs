use crate::{linalg, FloatDType, Result, ToMatrixView, ToVectorView, Vector};

pub enum SolveMethod {
    LU,
    Cholesky,
    Qr
}

/// Solve system: `a x = y`
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
