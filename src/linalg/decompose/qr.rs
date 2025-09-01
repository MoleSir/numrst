use crate::{FloatDType, Matrix, Result, ToMatrixView, Vector};

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

        // 构造 v
        let v = x.copy();
        let sign = if x.g(0) >= T::zero() { T::one() } else { T::one() };
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

        // 更新 q
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
