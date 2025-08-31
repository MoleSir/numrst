use crate::{FloatDType, Matrix, Result, ToMatrixView, ToVectorView, Vector};

/// Solve linear equaltion $A x = Y$
pub fn lu_solve<T, M, V>(a: M, y: V) -> Result<Vector<T>> 
where 
    T: FloatDType,
    M: ToMatrixView<T>,
    V: ToVectorView<T>,
{
    let (l, u) = lu(a)?;
    luy_solve(l, u, y)
}

pub fn lu<T: FloatDType, M: ToMatrixView<T>>(mat: M) -> Result<(Matrix<T>, Matrix<T>)> {
    let mat = mat.to_matrix_view()?;
    let (m, n) = mat.shape();
    let l = Matrix::<T>::eye(n)?;
    let u = Matrix::<T>::zeros(m, n)?;

    for i in 0..n {
        // U
        for j in i..n {
            let sum = l.row(i)?.dot(&u.col(j)?)?;
            let arr_v = mat.g(i, j); 
            u.s(i, j, arr_v - sum);
        }

        // L
        for j in i+1..n {
            let sum = l.row(j)?.dot(&u.col(i)?)?;
            let arr_v = mat.g(j, i);
            let u_v = u.g(i, i);
            l.s(j, i, (arr_v - sum) / u_v);
        }
    }

    Ok((l, u))
}

fn luy_solve<T, M1, M2, V>(l: M1, u: M2, y: V) -> Result<Vector<T>> 
where 
    T: FloatDType,
    M1: ToMatrixView<T>,
    M2: ToMatrixView<T>,
    V: ToVectorView<T>,
{
    let l = l.to_matrix_view()?;
    let u = u.to_matrix_view()?;
    let y = y.to_vector_view()?;

    // Ax = y --> LUx = y --> Lz = y & Ux = z
    let n = y.len();

    // Forward substitution: L z = y
    // L's diag are all T::one()
    let z = Vector::<T>::zeros(n)?;
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..i {
            sum = sum + l.g(i, j) * z.g(j);
        }
        z.s(i, (y.g(i) - sum) / l.g(i, i));
    }

    // Backward substitution: U x = z
    let x = Vector::<T>::zeros(n)?;
    for i in (0..n).rev() {
        let mut sum = T::zero();
        for j in i+1..n {
            sum = sum + u.g(i, j) * x.g(j);
        }
        x.s(i, (z.g(i) - sum) / u.g(i, i));
    }

    Ok(x)
}


#[cfg(test)]
mod test {
    use crate::{linalg::lu_solve, NdArray};

    #[test]
    fn test_lu() {
        let arr = NdArray::new(&[
            [2., 3., 1.],
            [4., 7., 3.],
            [6., 18., 5.],
        ]).unwrap();
        let (l, u) = crate::linalg::lu(&arr).unwrap();
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
        let (l, u) = crate::linalg::lu(&arr).unwrap();
        let l = l.to_ndarray();
        let u = u.to_ndarray();
        let arr_rec = l.matmul(&u).unwrap();
        assert!(arr_rec.allclose(&arr, 1e-4, 1e-4));
    }

    #[test]
    fn test_lu_solve_simple() {
        let a = NdArray::new(&[
            [3., 1.],
            [1., 2.],
        ]).unwrap();
        let y = NdArray::from_vec([9., 8.].to_vec(), 2).unwrap();

        let x = lu_solve(&a, &y).unwrap();
        let expected = NdArray::from_vec([2., 3.].to_vec(), 2).unwrap();
        assert!(x.to_ndarray().allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_lu_identity() {
        let a = NdArray::<f64>::eye(4).unwrap();
        let (l, u) = crate::linalg::lu(&a).unwrap();
        let l = l.to_ndarray();
        let u = u.to_ndarray();
        let rec = l.matmul(&u).unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));
    }

    #[test]
    fn test_lu_random_matrix() {
        let a = NdArray::new(&[
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 10.],
        ]).unwrap();
        let (l, u) = crate::linalg::lu(&a).unwrap();
        let rec = l.to_ndarray().matmul(&u.to_ndarray()).unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));
    }
}