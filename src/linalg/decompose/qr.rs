use crate::{FloatDType, NdArray, Result};

/// Computes the QR decomposition of a matrix `A` using Householder reflections.
///
/// # Description
/// QR decomposition factorizes a matrix `A` into an orthogonal matrix `Q`
/// and an upper triangular matrix `R` such that `A = Q * R`.
///
/// This implementation uses **Householder reflections**, which are numerically
/// stable and suitable for both square and rectangular matrices.
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
pub fn qr<T: FloatDType>(arr: &NdArray<T>) -> Result<(NdArray<T>, NdArray<T>)> {
    let a = arr.matrix_view()?;    
    let (m, n) = a.shape();
    
    unsafe {
        let r_arr = a.copy(); // (m, n)
        let q_arr = NdArray::<T>::eye(m)?; // (m, m)

        let mut r = r_arr.matrix_view().unwrap();
        let mut q = q_arr.matrix_view().unwrap();

        for k in 0..n {
            let x_arr = NdArray::<T>::zeros(m - k)?;
            let mut x = x_arr.vector_view().unwrap(); 
            for i in 0..(m - k) {
                x.s(i, r.g(k + i, k));
            }
    
            // v
            let v_arr = x.copy();
            let mut v = v_arr.vector_view().unwrap(); 

            let sign = if x.g(0) >= T::zero() { T::one() } else { -T::one() };
            let norm_x = x.norm();
            v.s(0, v.g(0) + sign * norm_x);
    
            // H = I - 2 vv^T / (v^T v)
            let beta = (T::one() + T::one()) / v.dot(&v)?;
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
        
        Ok((q_arr, r_arr))
    }
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
        let (_, _) = linalg::qr(&a).unwrap();
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
