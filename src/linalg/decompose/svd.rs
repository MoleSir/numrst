use crate::{linalg, FloatDType, IndexOp, NdArray, Result};

pub struct SvdResult<T: FloatDType> {
    pub u: NdArray<T>,
    pub sigmas: Vec<T>,
    pub v: NdArray<T>,
}

impl<T: FloatDType> SvdResult<T> {
    fn new(u: NdArray<T>, sigmas: Vec<T>, v: NdArray<T>) -> Self {
        Self { u, sigmas, v }
    }

    pub fn reconstruct(&self) -> Result<NdArray<T>> {
        let sigma = NdArray::diag(&self.sigmas).unwrap();
        let rec = self.u.matmul(&sigma)?.matmul(&self.v.transpose_last()?)?;
        Ok(rec)
    }
}

pub fn svd<T: FloatDType>(a: &NdArray<T>, tol: T) -> Result<SvdResult<T>> {
    let (m, n) = a.dims2()?;
    
    if m >= n {
        // 1. 对 A^T A 求特征分解
        let ata = a.transpose_last()?.matmul(a)?; // (n, n)
        let eig_res = linalg::eig_jacobi(&ata, tol)?;    // 对称矩阵
        let v = eig_res.eig_vectors;             // (n, n)
        let s: Vec<T> = eig_res.eig_values.iter().map(|&x| x.sqrt()).collect();

        // 2. 计算 U
        let mut u_cols = Vec::with_capacity(n);
        for j in 0..n {
            // let vj = v.slice_col(j)?;              // V 的第 j 列
            let vj = v.index((.., j))?;
            let av = linalg::mat_mul_vec(&a, &vj)?;               // A v_j
            let sigma = s[j];
            if sigma != T::zero() {
                u_cols.push(av.div(sigma)?);      // u_j = A v_j / sigma
            } else {
                // sigma=0 时，补充任意正交向量
                u_cols.push(NdArray::<T>::zeros((m,))?);
            }
        }
        let u = NdArray::stack(&u_cols, 1)?;   // (m, n)
        Ok(SvdResult::new(u, s, v))
    } else {
        // m < n 时，对 A A^T 求特征分解，然后类似方法计算 V
        let aat = a.matmul(&a.transpose_last()?)?; // (m, m)
        let eig_res = linalg::eig_jacobi(&aat, tol)?;
        let u = eig_res.eig_vectors;               // (m, m)
        let s: Vec<T> = eig_res.eig_values.iter().map(|&x| x.sqrt()).collect();

        let mut v_cols = Vec::with_capacity(m);
        for i in 0..m {
            let ui = u.index((.., i))?;
            let atu = linalg::mat_mul_vec(&a.transpose_last()?, &ui)?;
            let sigma = s[i];
            if sigma != T::zero() {
                v_cols.push(atu.div(sigma)?);
            } else {
                v_cols.push(NdArray::<T>::zeros((n,))?);
            }
        }
        let v = NdArray::stack(&v_cols, 1)?;   // (n, m)
        Ok(SvdResult::new(u, s, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn test_svd_identity() {
        let a = NdArray::<f64>::eye(3).unwrap();
        let result = svd(&a, 1e-12).unwrap();

        // 检查奇异值
        for &sigma in result.sigmas.iter() {
            assert!((sigma - 1.0).abs() < 1e-10);
        }

        // 检查重构
        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-10, 1e-10));

        // 检查正交性
        assert!(check_orthogonal(&result.u, 1e-10));
        assert!(check_orthogonal(&result.v, 1e-10));
    }

    #[test]
    fn test_svd_diagonal_matrix() {
        let a = NdArray::new(&[
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 2.0],
        ]).unwrap();

        let result = svd(&a, 1e-12).unwrap();
        let sigma = NdArray::diag(&result.sigmas).unwrap();
        let rec = result.u.matmul(&sigma).unwrap().matmul(&result.v.transpose_last().unwrap()).unwrap();
        assert!(rec.allclose(&a, 1e-10, 1e-10));

        assert!(check_orthogonal(&result.u, 1e-10));
        assert!(check_orthogonal(&result.v, 1e-10));
    }

    #[test]
    fn test_svd_zeros() {
        let a: NdArray<f64> = NdArray::<f64>::zeros((4, 4)).unwrap();
        let _ = svd(&a, 1e-12).unwrap();
    }

    #[test]
    fn test_svd_rectangular_matrix_m_gt_n() {
        let a = NdArray::<f64>::randn(0.0, 1.0, (5, 3)).unwrap();
        let result = svd(&a, 1e-10).unwrap();

        let sigma = NdArray::diag(&result.sigmas).unwrap();
        let rec = result.u.matmul(&sigma).unwrap().matmul(&result.v.transpose_last().unwrap()).unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));

        assert!(check_orthogonal(&result.u, 1e-6));
        assert!(check_orthogonal(&result.v, 1e-6));
    }

    #[test]
    fn test_svd_rectangular_matrix_m_lt_n() {
        let a = NdArray::<f64>::randn(0.0, 1.0, (3, 5)).unwrap();
        let result = svd(&a, 1e-10).unwrap();

        let sigma = NdArray::diag(&result.sigmas).unwrap();
        let rec = result.u.matmul(&sigma).unwrap().matmul(&result.v.transpose_last().unwrap()).unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));

        assert!(check_orthogonal(&result.u, 1e-6));
        assert!(check_orthogonal(&result.v, 1e-6));
    }

    // #[test]
    // fn test_svd_rank_deficient() {
    //     // rank 2 < 3
    //     let a = NdArray::new(&[
    //         [1.0, 2.0, 3.0],
    //         [2.0, 4.0, 6.0],
    //         [1.0, 1.0, 1.0],
    //     ]).unwrap();
    //     let result = svd(&a, 1e-12).unwrap();
    
    //     // 奇异值应有一个接近 0
    //     let zero_sigma = result.sigmas.iter().filter(|&&x| f64::abs(x) < 1e-5).count();
    //     assert!(zero_sigma >= 1);
    
    //     // 重构矩阵近似原矩阵
    //     let rec = result.reconstruct().unwrap();
    //     assert!(rec.allclose(&a, 1e-7, 1e-7));
    
    //     // U、V 正交
    //     assert!(check_orthogonal(&result.u, 1e-6));
    //     assert!(check_orthogonal(&result.v, 1e-6));
    // }

    fn check_orthogonal<T: FloatDType>(mat: &NdArray<T>, tol: T) -> bool {
        let mat_t = mat.transpose_last().unwrap();
        let prod = mat_t.matmul(mat).unwrap();
        let n = prod.dims2().unwrap().0;
        let id = NdArray::<T>::eye(n).unwrap();
        println!("{}", prod);
        prod.allclose(&id, tol.to_f64(), tol.to_f64())
    }
}
