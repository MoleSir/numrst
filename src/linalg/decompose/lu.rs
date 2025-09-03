use crate::{linalg::LinalgError, FloatDType, NdArray, Result};

/// Computes the LU decomposition of a matrix `A`.
///
/// # Description
/// LU decomposition factorizes a square matrix `A` into a lower triangular matrix `L`
/// and an upper triangular matrix `U` such that `A = L * U`.
/// 
/// This implementation does **not** perform row pivoting. As a result, it can fail
/// with `LinalgError::SingularMatrix` if the matrix is singular or nearly singular,
/// i.e., when a pivot element is too close to zero.
///
/// # Parameters
/// - `mat`: The input matrix `A` to decompose.
///
/// # Returns
/// - `(L, U)` where:
///   - `L` is a lower triangular matrix with ones on the diagonal.
///   - `U` is an upper triangular matrix.
///
/// # Notes
/// - For stability on nearly singular or ill-conditioned matrices, consider using
///   `plu` which includes partial pivoting.
/// - The input matrix can be rectangular, but the decomposition is only guaranteed
///   to be valid if it is square and non-singular.
///
/// # Example
/// ```rust
/// use numrst::{linalg, NdArray};
/// let a = NdArray::new(&[
///     [3.0, 1.0],
///     [1.0, 2.0]
/// ]).unwrap();
/// let (l, u) = linalg::lu(&a).unwrap();
/// // Now a ≈ L * U
/// ```
pub fn lu<T: FloatDType>(arr: &NdArray<T>) -> Result<(NdArray<T>, NdArray<T>)> {
    let mat = arr.matrix_view_unsafe()?;
    let (m, n) = mat.shape();

    let l_arr = NdArray::<T>::eye(m)?;
    let u_arr = NdArray::<T>::zeros((m, n))?;
    let k = m.min(n);

    unsafe  {
        let mut l = l_arr.matrix_view_unsafe()?;
        let mut u = u_arr.matrix_view_unsafe()?;
    
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
    }

    Ok((l_arr, u_arr))
}

/// Computes the PLU decomposition of a matrix `A`.
///
/// # Description
/// PLU decomposition factorizes a matrix `A` into a permutation matrix `P`,
/// a lower triangular matrix `L`, and an upper triangular matrix `U` such that
/// `P * A = L * U`.
///
/// This implementation includes **partial pivoting** for numerical stability.
/// It selects the largest pivot in each column and swaps rows accordingly.
/// 
/// Unlike the plain `lu` function, `plu` can handle matrices that are nearly singular
/// by reordering rows to avoid zero pivots. However, if an entire column is zero,
/// the decomposition will still produce zeros in `U` and may not yield a unique solution
/// when used to solve linear systems.
///
/// # Parameters
/// - `mat`: The input matrix `A` to decompose.
///
/// # Returns
/// - `(P, L, U)` where:
///   - `P` is a permutation matrix representing row swaps.
///   - `L` is a lower triangular matrix with ones on the diagonal.
///   - `U` is an upper triangular matrix.
///
/// # Notes
/// - `plu` is generally more robust than `lu` for solving linear systems,
///   especially when `A` is nearly singular.
/// - The decomposition can be used to solve `A x = y` by computing:
///   1. `P * y`
///   2. Solve `L * z = P * y` (forward substitution)
///   3. Solve `U * x = z` (backward substitution)
/// 
/// # References: 
/// https://github.com/LucasRoig/DecompositionPLU/blob/master/decomposition.py
///  
/// # Example
/// ```rust
/// # use numrst::{linalg, NdArray};
/// let a = NdArray::new(&[
///     [0.0, 2.0],
///     [1.0, 2.0]
/// ]).unwrap();
/// let (p, l, u) = linalg::plu(&a).unwrap();
/// // Now P * A ≈ L * U
/// ```
pub fn plu<T: FloatDType>(arr: &NdArray<T>) -> Result<(NdArray<T>, NdArray<T>, NdArray<T>)> {
    let mat = arr.matrix_view_unsafe()?;
    let (m, n) = mat.shape();

    unsafe {
        let u_arr = mat.copy(); // (m, n)
        let l_arr = NdArray::<T>::eye(m)?; // (m, m)
        let p_arr = NdArray::<T>::eye(m)?;
    
        {
    
            let mut u = u_arr.matrix_view_unsafe().unwrap();
            let mut l = l_arr.matrix_view_unsafe().unwrap();
            let mut p = p_arr.matrix_view_unsafe().unwrap();
        
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
        }
    
        Ok((p_arr, l_arr, u_arr))
    }
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
        let arr_rec = l.matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }

    #[test]
    fn test_lu_identity() {
        let a = NdArray::<f64>::eye(4).unwrap();
        let (l, u) = linalg::lu(&a).unwrap();
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

        let arr_rec = p.transpose_last().unwrap().matmul(&l).unwrap().matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }    

    #[test]
    fn test_lu_rand() {
        let arr = NdArray::<f64>::randn(0.0, 1.0, (6, 6)).unwrap();
        let result = linalg::lu(&arr);
        assert!(result.is_ok());
        let (l, u) = result.unwrap();

        let arr_rec = l.matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));

        let arr = NdArray::<f64>::randn(0.0, 1.0, (8, 6)).unwrap();
        let result = linalg::lu(&arr);
        assert!(result.is_ok());
        let (l, u) = result.unwrap();

        let arr_rec = l.matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }   

    #[test]
    fn test_plu_rand() {
        let arr = NdArray::<f64>::randn(0.0, 1.0, (6, 6)).unwrap();
        let result = linalg::plu(&arr);
        assert!(result.is_ok());
        let (p, l, u) = result.unwrap();

        let arr_rec = p.transpose_last().unwrap().matmul(&l).unwrap().matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }   
}