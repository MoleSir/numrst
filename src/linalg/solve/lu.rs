use crate::{linalg::{self, LinalgError}, FloatDType, NdArray, Result};
use super::utils;

pub fn lu_solve<T: FloatDType>(a: &NdArray<T>, y: &NdArray<T>) -> Result<NdArray<T>> {
    utils::check_solve_arg(&a, &y)?;
    let (l, u) = linalg::lu(a)?;
    let y = y.vector_view()?;

    let l = l.matrix_view()?;
    let u = u.matrix_view()?;

    unsafe {
        // Ax = y --> LUx = y --> Lz = y & Ux = z
        let n = y.len();

        // Forward substitution: L z = y
        // L's diag are all T::one()
        let z_arr = NdArray::<T>::zeros(n)?;
        let mut z = z_arr.vector_view()?;
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..i {
                sum = sum + l.g(i, j) * z.g(j);
            }
            z.s(i, (y.g(i) - sum) / l.g(i, i));
        }

        // Backward substitution: U x = z
        let x_arr = NdArray::<T>::zeros(n)?;
        {
            let mut x = x_arr.vector_view().unwrap();
            for i in (0..n).rev() {
                let mut sum = T::zero();
                for j in i+1..n {
                    sum = sum + u.g(i, j) * x.g(j);
                }
                x.s(i, (z.g(i) - sum) / u.g(i, i));
            }
        }

        Ok(x_arr)
    }
}

pub fn plu_solve<T: FloatDType>(a: &NdArray<T>, y: &NdArray<T>) -> Result<NdArray<T>> {
    utils::check_solve_arg(&a, &y)?;    
    let (p, l, u) = linalg::plu(a)?;
    let l = l.matrix_view()?;
    let u = u.matrix_view()?;

    let py: NdArray<T> = linalg::mat_mul_vec(&p, y)?;
    let py = py.vector_view().unwrap();
    
    let n = py.len();

    unsafe {
        // L·z = P·y
        let z_arr = NdArray::<T>::zeros(n)?;
        let mut z = z_arr.vector_view().unwrap();
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..i {
                sum = sum + l.g(i, j) * z.g(j);
            }
            z.s(i, py.g(i) - sum);
        }

        // U·x = z
        let x_arr = NdArray::<T>::zeros(n)?;
        {
            let mut x = x_arr.vector_view().unwrap();
            for i in (0..n).rev() {
                let mut sum = T::zero();
                for j in i+1..n {
                    sum = sum + u.g(i, j) * x.g(j);
                }
                let u_ii = u.g(i, i);
                if u_ii.abs() <= T::epsilon() {
                    return Err(LinalgError::SingularMatrix)?;
                }
                x.s(i, (z.g(i) - sum) / u_ii);
            }
        }
        Ok(x_arr)
    }
}

#[cfg(test)]
mod tests {
    use crate::{linalg::{self, LinalgError}, Error, NdArray};

    #[test]
    fn test_lu_solve_simple() {
        let a = NdArray::new(&[
            [3., 1.],
            [1., 2.],
        ]).unwrap();
        let y = NdArray::from_vec([9., 8.].to_vec(), 2).unwrap();

        let x = linalg::lu_solve(&a, &y).unwrap();
        let expected = NdArray::from_vec([2., 3.].to_vec(), 2).unwrap();
        assert!(x.allclose(&expected, 1e-6, 1e-6));

        let y_rec = a.matmul(&x.unsqueeze(1).unwrap()).unwrap()
                     .squeeze(1).unwrap();
        assert!(y.allclose(&y_rec, 1e-6, 1e-6));
    }

    #[test]
    fn test_lu_singular_vs_plu() {
        let a = NdArray::new(&[
            [1., 2., 3.],
            [1., 2., 3.],
            [4., 5., 6.],
        ]).unwrap();
        let y = NdArray::from_vec([6., 6., 15.].to_vec(), 3).unwrap();

        let lu_res = linalg::lu_solve(&a, &y);
        assert!(matches!(lu_res, Err(Error::Linalg(LinalgError::SingularMatrix))));

        let plu_res = linalg::plu_solve(&a, &y);
        assert!(plu_res.is_err());
    }

    #[test]
    fn test_non_square_matrix() {
        let a = NdArray::new(&[
            [1., 2., 3.],
            [4., 5., 6.],
        ]).unwrap();
        let y = NdArray::from_vec([7., 8.].to_vec(), 2).unwrap();

        let res = linalg::lu_solve(&a, &y);
        assert!(matches!(res, Err(Error::Linalg(LinalgError::NonSquareMatrix(_)))));

        let res = linalg::plu_solve(&a, &y);
        assert!(matches!(res, Err(Error::Linalg(LinalgError::NonSquareMatrix(_)))));
    }

    #[test]
    fn test_random_invertible() {
        let a = NdArray::<f64>::randn(0.0, 1.0, (4, 4)).unwrap();
        let y = NdArray::<f64>::randn(0.0, 1.0, (4,)).unwrap();

        let x = linalg::lu_solve(&a, &y).unwrap();
        let y_rec = a.matmul(&x.unsqueeze(1).unwrap()).unwrap()
                     .squeeze(1).unwrap();
        assert!(y.allclose(&y_rec, 1e-6, 1e-6));

        let x2 = linalg::plu_solve(&a, &y).unwrap();
        let y_rec2 = a.matmul(&x2.unsqueeze(1).unwrap()).unwrap()
                      .squeeze(1).unwrap();
        assert!(y.allclose(&y_rec2, 1e-6, 1e-6));
    }

    #[test]
    fn test_zero_matrix() {
        let a = NdArray::new(&[
            [0., 0.],
            [0., 0.],
        ]).unwrap();
        let y = NdArray::from_vec([0., 0.].to_vec(), 2).unwrap();

        assert!(matches!(linalg::lu_solve(&a, &y), Err(Error::Linalg(LinalgError::SingularMatrix))));
        assert!(matches!(linalg::plu_solve(&a, &y), Err(Error::Linalg(LinalgError::SingularMatrix))));
    }
}
