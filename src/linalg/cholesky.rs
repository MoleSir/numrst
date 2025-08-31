use crate::{Error, FloatDType, Matrix, Result, ToMatrixView, ToVectorView, Vector};

pub fn cholesky<T: FloatDType, M: ToMatrixView<T>>(mat: M) -> Result<Matrix<T>> {
    let mat = mat.to_matrix_view()?;
    let (n, _) = mat.shape();
    let l = Matrix::<T>::zeros(n, n)?;

    for i in 0..n {
        // diag
        let mut sum = T::zero();
        for k in 0..i {
            sum = sum + l.g(i, k) * l.g(i, k);
        }
        let diag = mat.g(i, i) - sum;
        if diag <= T::zero() {
            return Err(Error::Msg("Matrix is not positive definite".into()));
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

    Ok(l)
}

/// Solve A x = y using Cholesky (A must be SPD)
pub fn chol_solve<T, M, V>(a: M, y: V) -> Result<Vector<T>>
where
    T: FloatDType,
    M: ToMatrixView<T>,
    V: ToVectorView<T>,
{
    let l = cholesky(a)?;
    let l = l.to_matrix_view()?;
    let y = y.to_vector_view()?;
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