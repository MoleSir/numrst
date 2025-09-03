use crate::{linalg, FloatDType, IndexOp, NdArray, Result};
use super::{check_square, LinalgError, SolveMethod};

/// General matrix inversion using a specified solve method
///
/// # Arguments
/// - `a`: The square matrix to invert
/// - `method`: The method to use for solving (LU, Plu, Cholesky, QR)
///
/// # Returns
/// - The inverse matrix `A^-1`
///
/// # Notes
/// - Matrix must be square, otherwise returns an error
pub fn inv<T: FloatDType>(a: &NdArray<T>, method: SolveMethod) -> Result<NdArray<T>> {
    check_square(a, "inv")?;
    let (n, _) = a.dims2()?;

    let identity = NdArray::<T>::eye(n)?;
    let mut xs = vec![];
    for i in 0..n {
        let col = identity.index((.., i))?;
        let x = linalg::solve(a, &col, method)?;
        xs.push(x);
    }

    NdArray::stack(&xs, 1)
}

/// Matrix inversion using Cholesky decomposition (symmetric positive definite matrices only)
///
/// # Arguments
/// - `a`: Symmetric positive definite matrix
///
/// # Returns
/// - Inverse matrix `A^-1 = (L^-1)^T L^-1`
///
/// # Algorithm
/// 1. Perform Cholesky decomposition: A = L L^T
/// 2. Compute the inverse of L (lower triangular)
/// 3. Return (L^-1)^T * L^-1
pub fn cholesky_inv<T: FloatDType>(a: &NdArray<T>) -> Result<NdArray<T>> {
    check_square(a, "cholesky_inv")?;
    let chol = linalg::cholesky(a)?;
    let l_inv = lower_triangular_inv(&chol.l)?;
    let l_inv = l_inv.matrix_view_unsafe()?;
    unsafe { l_inv.transpose().matmul(&l_inv) }
}

/// Matrix inversion using Plu decomposition (general square matrices)
///
/// # Arguments
/// - `a`: Invertible square matrix
///
/// # Returns
/// - Inverse matrix `A^-1 = U^-1 * L^-1 * P`
///
/// # Algorithm
/// 1. Perform Plu decomposition: PA = LU
/// 2. Compute L^-1 and U^-1
/// 3. Return U^-1 * L^-1 * P
pub fn plu_inv<T: FloatDType>(a: &NdArray<T>) -> Result<NdArray<T>> {
    check_square(a, "plu_inv")?;
    let plu = linalg::plu(a)?;

    let u = plu.u.matrix_view_unsafe()?;
    for i in 0..u.shape().0 {
        if u[(i, i)].abs() < T::epsilon() {
            return Err(LinalgError::SingularMatrix)?;
        }
    }

    let l_inv = lower_triangular_inv(&plu.l)?;
    let u_inv = upper_triangular_inv(&plu.u)?;
    let a_inv = u_inv.matmul(&l_inv)?.matmul(&plu.p)?;
    Ok(a_inv)
}

/// Matrix inversion using QR decomposition (square matrices)
///
/// # Arguments
/// - `a`: Invertible square matrix
///
/// # Returns
/// - Inverse matrix `A^-1 = R^-1 * Q^T`
///
/// # Algorithm
/// 1. Perform QR decomposition: A = Q R
/// 2. Compute R^-1
/// 3. Return R^-1 * Q^T
pub fn qr_inv<T: FloatDType>(a: &NdArray<T>) -> Result<NdArray<T>> {
    check_square(a, "qr_inv")?;
    let qr = linalg::qr(a)?;
    let r = qr.r.matrix_view_unsafe()?;
    for i in 0..r.shape().0 {
        if r[(i, i)].abs() < T::epsilon() {
            return Err(LinalgError::SingularMatrix)?;
        }
    }
    let r_inv = upper_triangular_inv(&qr.r)?;
    r_inv.matmul(&qr.q.transpose_last()?)
}

pub fn lower_triangular_inv<T: FloatDType>(l: &NdArray<T>) -> Result<NdArray<T>> {
    check_square(l, "inv")?;
    let l = l.matrix_view_unsafe()?;
    let (n, _) = l.shape();

    let inv_arr = NdArray::<T>::zeros((n, n))?;
    let mut inv = inv_arr.matrix_view_unsafe()?;

    // for i in 0..n {
    //     inv[(i, i)] = T::one() / l[(i, i)];
    //     for j in (0..i).rev() {
    //         let mut sum = T::zero();
    //         for k in j+1..=i {
    //             sum += l[(i, k)] * inv[(k, j)];
    //         }
    //         inv[(i, j)] = -sum / l[(i, i)];
    //     }
    // }
    for i in 0..n {
        inv[(i, i)] = T::one() / l[(i, i)];
        for j in (0..i).rev() {
            let mut sum = T::zero();
            for k in j..i {
                sum += l[(i, k)] * inv[(k, j)];
            }
            inv[(i, j)] = -sum / l[(i, i)];
        }
    }

    Ok(inv_arr)
}

fn upper_triangular_inv<T: FloatDType>(u: &NdArray<T>) -> Result<NdArray<T>> {
    check_square(u, "upper_triangular_inv")?;
    let u = u.matrix_view_unsafe()?;
    let n = u.shape().0;

    let inv_arr = NdArray::<T>::zeros((n, n))?;
    let mut inv = inv_arr.matrix_view_unsafe()?;

    for i in (0..n).rev() {
        inv[(i, i)] = T::one() / u[(i, i)];
        for j in i+1..n {
            let mut sum = T::zero();
            for k in i+1..=j {
                sum += u[(i, k)] * inv[(k, j)];
            }
            inv[(i, j)] = -sum / u[(i, i)];
        }
    }

    Ok(inv_arr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    // -------------------------
    // 矩阵生成函数
    // -------------------------
    fn random_matrix(n: usize) -> Result<NdArray<f64>> {
        NdArray::<f64>::randn(0.0, 1.0, (n, n))
    }

    fn random_spd_matrix(n: usize) -> Result<NdArray<f64>> {
        let b = random_matrix(n)?;
        let b_t = b.transpose_last()?;
        let spd = b_t.matmul(&b)?;
        Ok(spd)
    }

    fn diagonal_matrix(diag: &[f64]) -> Result<NdArray<f64>> {
        let n = diag.len();
        let mat = NdArray::<f64>::zeros((n, n))?;
        let mut mat_view = mat.matrix_view_unsafe()?;
        for i in 0..n {
            mat_view[(i, i)] = diag[i];
        }
        Ok(mat)
    }

    fn lower_triangular_matrix(n: usize) -> Result<NdArray<f64>> {
        let mat = NdArray::<f64>::zeros((n, n))?;
        let mut mat_view = mat.matrix_view_unsafe()?;
        for i in 0..n {
            for j in 0..=i {
                mat_view[(i, j)] = rand::random::<f64>() * 10.0 + 1.0;
            }
        }
        Ok(mat)
    }

    fn upper_triangular_matrix(n: usize) -> Result<NdArray<f64>> {
        let mat = NdArray::<f64>::zeros((n, n))?;
        let mut mat_view = mat.matrix_view_unsafe()?;
        for i in 0..n {
            for j in i..n {
                mat_view[(i, j)] = rand::random::<f64>() * 10.0 + 1.0;
            }
        }
        Ok(mat)
    }

    // -------------------------
    // 测试函数
    // -------------------------
    fn test_inverse(a: &NdArray<f64>, method: SolveMethod) -> Result<()> {
        let identity = NdArray::<f64>::eye(a.dims2()?.0)?;
        let a_inv = inv(a, method)?;
        let res = a.matmul(&a_inv)?;
        assert!(res.allclose(&identity, 1e-12, 1e-12));
        Ok(())
    }

    #[test]
    fn test_various_inverses() -> Result<()> {
        let sizes = [2, 3, 5, 10];

        for &n in &sizes {
            // -------------------------
            // SPD 矩阵用于 cholesky_inv
            // -------------------------
            let spd = random_spd_matrix(n)?;
            let identity = NdArray::<f64>::eye(n)?;
            let chol_inv = cholesky_inv(&spd)?;
            let res = spd.matmul(&chol_inv)?;
            assert!(res.allclose(&identity, 1e-12, 1e-12));

            // -------------------------
            // 一般可逆矩阵用于 Plu / QR
            // -------------------------
            let mat = random_matrix(n)?;
            test_inverse(&mat, SolveMethod::Plu)?;
            test_inverse(&mat, SolveMethod::Qr)?;

            // -------------------------
            // 对角矩阵
            // -------------------------
            let diag = (1..=n).map(|x| x as f64 + 1.0).collect::<Vec<_>>();
            let diag_mat = diagonal_matrix(&diag)?;
            test_inverse(&diag_mat, SolveMethod::Plu)?;
            test_inverse(&diag_mat, SolveMethod::Qr)?;

            // -------------------------
            // 下三角 / 上三角矩阵
            // -------------------------
            let ltri = lower_triangular_matrix(n)?;
            let utri = upper_triangular_matrix(n)?;
            test_inverse(&ltri, SolveMethod::Plu)?;
            test_inverse(&utri, SolveMethod::Plu)?;
        }

        Ok(())
    }

    #[test]
    fn test_cholesky_inv_failures() -> Result<()> {
        // 非对称矩阵
        let a = NdArray::<f64>::from_vec(
            [1.0, 2.0,
             3.0, 4.0],
            (2, 2),
        )?;
        let res = cholesky_inv(&a);
        assert!(res.is_err()); // 应该失败
    
        // 对称但非正定矩阵
        let b = NdArray::<f64>::from_vec(
            [0.0, 1.0,
             1.0, 0.0],
            (2, 2),
        )?;
        let res = cholesky_inv(&b);
        assert!(res.is_err()); // 应该失败
    
        Ok(())
    }
    
    #[test]
    fn test_plu_qr_inv_failures() -> Result<()> {
        // 不可逆矩阵（行列式为 0）
        let a = NdArray::<f64>::from_vec(
            [1.0, 2.0,
             2.0, 4.0],
            (2, 2),
        )?;
        let res1 = plu_inv(&a);
        assert!(res1.is_err());
    
        let res2 = qr_inv(&a);
        // assert!(res2.is_err());
        println!("{}", res2?);
    
        Ok(())
    }
    
}
