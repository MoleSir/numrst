use crate::{linalg::LinalgError, FloatDType, Matrix, Result, ToMatrixView};

pub fn lu<T: FloatDType, M: ToMatrixView<T>>(mat: M) -> Result<(Matrix<T>, Matrix<T>)> {
    let mat = mat.to_matrix_view()?;
    let (m, n) = mat.shape();
    let l = Matrix::<T>::eye(m)?;
    let u = Matrix::<T>::zeros(m, n)?;
    let k = m.min(n); 

    for i in 0..k {
        // U
        for j in i..n {
            let sum = l.row(i)?.dot(&u.col(j)?)?; // (m,) dot (m,)
            let arr_v = mat.g(i, j); // A[i, j]
            u.s(i, j, arr_v - sum);
        }

        // Check
        let u_ii = u.g(i, i);
        if u_ii.abs() <= T::epsilon() {
            return Err(LinalgError::SingularMatrix)?;
        }

        // L
        for j in i+1..m {
            let sum = l.row(j)?.dot(&u.col(i)?)?;
            let arr_v = mat.g(j, i);
            let u_v = u.g(i, i);
            l.s(j, i, (arr_v - sum) / u_v);
        }
    }

    Ok((l, u))
}

/// References: https://github.com/LucasRoig/DecompositionPLU/blob/master/decomposition.py
pub fn plu<T: FloatDType, M: ToMatrixView<T>>(mat: M) 
    -> Result<(Matrix<T>, Matrix<T>, Matrix<T>)> 
{
    let mat = mat.to_matrix_view()?;
    let (m, n) = mat.shape();

    let u = mat.copy();
    let l = Matrix::<T>::eye(m)?;
    let p = Matrix::<T>::eye(m)?;

    let k = m.min(n);

    for i in 0..k {
        let mut pivot_row = i;
        let mut max_val = u.g(i, i).abs();
        for j in (i+1)..m {
            let val = u.g(j, i).abs();
            if val > max_val {
                max_val = val;
                pivot_row = j;
            }
        }

        if pivot_row != i {
            u.swap_rows(i, pivot_row)?;
            p.swap_rows(i, pivot_row)?;
            if i > 0 {
                l.swap_rows_partial(i, pivot_row, i)?;
            }
        }

        let pivot = u.g(i, i);
        if pivot.abs() <= T::epsilon() {
            continue;
        }

        for j in (i+1)..m {
            let factor = u.g(j, i) / pivot;
            l.s(j, i, factor);

            for kcol in i..n {
                let new_val = u.g(j, kcol) - factor * u.g(i, kcol);
                u.s(j, kcol, new_val);
            }
        }
    }

    Ok((p, l, u))
}

#[cfg(test)]
mod test {
    use crate::{linalg::{self, LinalgError}, Error, NdArray};

    #[test]
    fn test_lu() {
        let arr = NdArray::new(&[
            [2., 3., 1.],
            [4., 7., 3.],
            [6., 18., 5.],
        ]).unwrap();
        let (l, u) = linalg::lu(&arr).unwrap();
        let l = l.to_ndarray();
        let u = u.to_ndarray();
        let arr_rec = l.matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }

    #[test]
    fn test_lu_decomp_reconstruct() {
        let arr = NdArray::new(&[
            [2., 3., 1.],
            [4., 7., 3.],
            [6., 18., 5.],
        ]).unwrap();
        let (l, u) = linalg::lu(&arr).unwrap();
        let l = l.to_ndarray();
        let u = u.to_ndarray();
        let arr_rec = l.matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }

    #[test]
    fn test_lu_identity() {
        let a = NdArray::<f64>::eye(4).unwrap();
        let (l, u) = linalg::lu(&a).unwrap();
        let l = l.to_ndarray();
        let u = u.to_ndarray();
        let rec = l.matmul(&u).unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));
    }

    #[test]
    fn test_plu_decomp_reconstruct() {
        let arr = NdArray::new(&[
            [0., 2., 3.],
            [4., 7., 3.],
            [6., 18., 5.],
        ]).unwrap();
    
        let (p, l, u) = linalg::plu(&arr).unwrap();
        let p = p.to_ndarray();
        let l = l.to_ndarray();
        let u = u.to_ndarray();

        let arr_rec = p.transpose_last().unwrap().matmul(&l).unwrap().matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }

    #[test]
    fn test_lu_rectangular() {
        let arr = NdArray::new(&[
            [2., 3., 1.],
            [4., 7., 3.],
        ]).unwrap(); // 2x3
    
        let (l, u) = linalg::lu(&arr).unwrap();
        let l = l.to_ndarray();
        let u = u.to_ndarray();
    
        let arr_rec = l.matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }

    #[test]
    fn test_lu_singular() {
        let a = NdArray::new(&[
            [1., 2., 3.],
            [1., 2., 3.],
            [4., 5., 6.],
        ]).unwrap();
        let res = linalg::lu(&a);
        matches!(res, Err(Error::Linalg(LinalgError::SingularMatrix)));
    }

    #[test]
    fn test_plu_singular1() {
        let a = NdArray::new(&[
            [1., 2., 3.],
            [1., 2., 3.],
            [4., 5., 6.],
        ]).unwrap();
        let result = linalg::plu(&a);
        assert!(result.is_ok());
        let (p, l, u) = result.unwrap();
        let p = p.to_ndarray();
        let l = l.to_ndarray();
        let u = u.to_ndarray();

        let arr_rec = p.transpose_last().unwrap().matmul(&l).unwrap().matmul(&u).unwrap();
        assert!(arr_rec.allclose(&a, 1e-4, 1e-4));
    }

    #[test]
    fn test_plu_singular() {
        let arr = NdArray::new(&[
            [2., 4., 1.],
            [4., 8., 2.],
            [1., 2., 0.5],
        ]).unwrap();
    
        let result = linalg::plu(&arr);
        assert!(result.is_ok());
        let (p, l, u) = result.unwrap();
        let p = p.to_ndarray();
        let l = l.to_ndarray();
        let u = u.to_ndarray();

        let arr_rec = p.transpose_last().unwrap().matmul(&l).unwrap().matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }    

    #[test]
    fn test_lu_rand() {
        let arr = NdArray::<f64>::randn(0.0, 1.0, (6, 6)).unwrap();
        let result = linalg::lu(&arr);
        assert!(result.is_ok());
        let (l, u) = result.unwrap();
        let l = l.to_ndarray();
        let u = u.to_ndarray();

        let arr_rec = l.matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));

        let arr = NdArray::<f64>::randn(0.0, 1.0, (8, 6)).unwrap();
        let result = linalg::lu(&arr);
        assert!(result.is_ok());
        let (l, u) = result.unwrap();
        let l = l.to_ndarray();
        let u = u.to_ndarray();

        let arr_rec = l.matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }   

    #[test]
    fn test_plu_rand() {
        let arr = NdArray::<f64>::randn(0.0, 1.0, (6, 6)).unwrap();
        let result = linalg::plu(&arr);
        assert!(result.is_ok());
        let (p, l, u) = result.unwrap();
        let p = p.to_ndarray();
        let l = l.to_ndarray();
        let u = u.to_ndarray();

        let arr_rec = p.transpose_last().unwrap().matmul(&l).unwrap().matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }   
}