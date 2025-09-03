use crate::{FloatDType, NdArray, Result};
use crate::linalg;
use super::utils;

pub fn cholesky_solve<T: FloatDType>(a: &NdArray<T>, y: &NdArray<T>) -> Result<NdArray<T>> {
    utils::check_solve_arg(&a, &y)?;

    // A = L @ L.T
    // L @ L.T @ x = y
    // 1: L @ z = y
    // 2: L.T @ x = z
    let l_arr = linalg::cholesky(a)?;
    let l = l_arr.matrix_view_unsafe()?;
    let y = y.vector_view_unsafe()?;
    let n = y.len();

    unsafe {
        // Step 1: forward solve L z = y
        let z_arr = NdArray::<T>::zeros(n)?;   
        let mut z = z_arr.vector_view_unsafe().unwrap();
        for i in 0..n {
            let mut sum = T::zero();
            for k in 0..i {
                sum = sum + l.g(i, k) * z.g(k);
            }
            z.s(i, (y.g(i) - sum) / l.g(i, i));
        }

        // Step 2: backward solve L^T x = z
        let x_arr = NdArray::<T>::zeros(n)?;
        {
            let mut x = x_arr.vector_view_unsafe().unwrap();
            for i in (0..n).rev() {
                let mut sum = T::zero();
                for k in i+1..n {
                    sum = sum + l.g(k, i) * x.g(k);
                }
                x.s(i, (z.g(i) - sum) / l.g(i, i));
            }
        }

        Ok(x_arr)
    }
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

        assert!(x.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_cholesky_solve_identity() {
        let a = NdArray::<f64>::eye(4).unwrap();
        let y = NdArray::from_vec([1., 2., 3., 4.].to_vec(), 4).unwrap();

        let x = linalg::cholesky_solve(&a, &y).unwrap();
        assert!(x.allclose(&y, 1e-6, 1e-6));
    }
}
