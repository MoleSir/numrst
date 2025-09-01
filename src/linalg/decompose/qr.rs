use crate::{FloatDType, Matrix, Result, ToMatrixView, Vector};

/// Computes the QR decomposition of a matrix `A` using Householder reflections.
///
/// # Description
/// QR decomposition factorizes a matrix `A` into an orthogonal matrix `Q`
/// and an upper triangular matrix `R` such that `A = Q * R`.
///
/// This implementation uses **Householder reflections**, which are numerically
/// stable and suitable for both square and rectangular matrices.
///
/// # Type Parameters
/// - `T`: The floating-point data type. Must implement `FloatDType`.
/// - `M`: A type that can be converted to a matrix view (`ToMatrixView<T>`).
///
/// # Parameters
/// - `a`: The input matrix `A` to decompose. Can be rectangular (`m x n`).
///
/// # Returns
/// `(Q, R)` where:
///   - `Q` is an orthogonal matrix of size `m x m` (`Q^T * Q = I`).
///   - `R` is an upper triangular matrix of size `m x n`.
///
/// # Notes
/// - This implementation produces a full `Q` of size `m x m`. The upper-left
///   `m x n` block can be used for a reduced QR decomposition if desired.
/// - The algorithm iteratively constructs Householder vectors to zero out
///   sub-diagonal elements column by column.
/// - `Q` is built as the product of Householder transformations applied to the identity matrix.
///
/// # Example
/// ```rust
/// # use numrst::{linalg, NdArray};
/// let a = NdArray::new(&[
///     [12.0, -51.0, 4.0],
///     [6.0, 167.0, -68.0],
///     [-4.0, 24.0, -41.0],
/// ]).unwrap();
/// let (q, r) = linalg::qr(&a).unwrap();
/// // Now a ≈ Q * R
/// // Q is orthogonal, R is upper triangular
/// ```
pub fn qr<T, M>(a: M) -> Result<(Matrix<T>, Matrix<T>)> 
where 
    T: FloatDType,
    M: ToMatrixView<T>
{
    let a = a.to_matrix_view()?;
    let (m, n) = a.shape();
    let r = a.copy(); // (m, n)
    let q = Matrix::<T>::eye(m)?; // (m, m)

    for k in 0..n {
        let x = Vector::<T>::zeros(m - k)?;
        for i in 0..(m - k) {
            x.s(i, r.g(k + i, k));
        }

        // v
        let v = x.copy();
        let sign = if x.g(0) >= T::zero() { T::one() } else { -T::one() };
        let sum_ = x.iter().map(|v| v.powi(2)).sum::<T>();
        let norm_x = sum_.sqrt();
        v.s(0, v.g(0) + sign * norm_x);

        // H = I - 2 vv^T / (v^T v)
        let beta = (T::one() + T::one()) / crate::linalg::dot(&v, &v)?;
        for j in k..n {
            // r[k.., j] -= beta * v * (v^T r[k.., j])
            let mut proj = T::zero();
            for i in 0..v.len() {
                proj += v.g(i) * r.g(k + i, j);
            }
            proj *= beta;
            for i in 0..v.len() {
                r.s(k + i, j, r.g(k + i, j) - proj * v.g(i));
            }
        }

        // q[:, k..] -= q[:, k..] * beta * v * v^T
        for i in 0..m {
            let mut proj = T::zero();
            for j in 0..v.len() {
                proj += q.g(i, k + j) * v.g(j);
            }
            proj *= beta;
            for j in 0..v.len() {
                q.s(i, k + j, q.g(i, k + j) - proj * v.g(j));
            }
        }
    }

    Ok((q, r))
}

#[cfg(test)]
mod test {
    use crate::{NdArray, linalg};

    #[test]
    fn test_qr_simple() {
        let a = NdArray::new(&[
            [12., -51., 4.],
            [6., 167., -68.],
            [-4., 24., -41.],
        ]).unwrap();

        let (q, r) = linalg::qr(&a).unwrap();
        let q = q.to_ndarray();
        let r = r.to_ndarray();

        // 检查 A ≈ Q * R
        let a_rec = q.matmul(&r).unwrap();
        println!("{}", a_rec);
        assert!(a_rec.allclose(&a, 1e-6, 1e-6));

        // 检查 Q 是否正交：Q^T Q ≈ I
        let qtq = q.transpose_last().unwrap().matmul(&q).unwrap();
        let i = NdArray::<f64>::eye(q.dims()[1]).unwrap();
        assert!(qtq.allclose(&i, 1e-6, 1e-6));
    }

    #[test]
    fn test_qr_identity() {
        let a = NdArray::<f64>::eye(4).unwrap();
        let (q, r) = linalg::qr(&a).unwrap();
        let _ = q.to_ndarray();
        let _ = r.to_ndarray();
    }

    #[test]
    fn test_qr_rectangular() {
        // 4x3 矩阵
        let a = NdArray::new(&[
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 10.],
            [1., 0., 0.],
        ]).unwrap();

        let (q, r) = linalg::qr(&a).unwrap();
        let q = q.to_ndarray();
        let r = r.to_ndarray();

        // A ≈ Q * R
        let a_rec = q.matmul(&r).unwrap();
        assert!(a_rec.allclose(&a, 1e-6, 1e-6));

        // Q^T Q ≈ I
        let qtq = q.transpose_last().unwrap().matmul(&q).unwrap();
        let m = q.dims()[1];
        let i = NdArray::<f64>::eye(m).unwrap();
        assert!(qtq.allclose(&i, 1e-6, 1e-6));
    }
}
