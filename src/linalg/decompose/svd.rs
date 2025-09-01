use rand_distr::{Distribution, StandardNormal};
use crate::{linalg, FloatDType, IndexOp, NdArray, Result};

pub fn svd<T: FloatDType>(a: &NdArray<T>, tol: T) -> Result<(NdArray<T>, Vec<T>, NdArray<T>)> {
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
        Ok((u, s, v))
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
        Ok((u, s, v))
    }
}

pub fn svd_lowrank<T: FloatDType>(a: &NdArray<T>, num_iters: usize) -> Result<(Vec<NdArray<T>>, Vec<T>, Vec<NdArray<T>>)> 
where 
    StandardNormal: Distribution<T>
{
    let (m, n) = a.dims2()?;
    let k = m.min(n);

    let mut vs = vec![];
    let mut ss = vec![];
    let mut us = vec![];

    let a_copy = a.copy();

    for _ in 0..k {
        // 1. right vector 
        // RS = A.T @ A === (n, m) @ (m, n) = (n, n)
        let rs = a_copy.transpose_last()?.matmul(&a_copy)?;
        
        // power_iteration get eig value and vector
        let (_, eig_v) = power_iteration(&rs, num_iters)?; // (n,)
        eig_v.div_assign(linalg::norm(&eig_v))?;
        vs.push(eig_v.clone());

        // 2. sigma and left vector
        let av = linalg::mat_mul_vec(a, &eig_v)?;
        let sigma = linalg::norm(&av);
        let eig_u = av.div(sigma)?; // (m,)

        us.push(eig_u.clone());  
        ss.push(sigma);

        // 3. update a copy
        // (m, 1) @ (1, n) => (m, n)
        let uv = linalg::outer(&eig_u, &eig_v)?;
        uv.mul_assign(sigma)?;
        a_copy.sub_assign(&uv)?;
    }

    return Ok((us, ss, vs))
}

fn power_iteration<T>(mat: &NdArray<T>, num_iters: usize) -> Result<(T, NdArray<T>)> 
where 
    T: FloatDType,
    StandardNormal: Distribution<T>
{
    let (m, n) = mat.dims2()?; // (n, n)
    assert_eq!(m, n);
    let mut b = NdArray::<T>::randn(T::from_f64(0.), T::from_f64(1.), (n, 1))?; // (n, 1)

    for _ in 0..num_iters {
        // (n, n) @ (n, 1) = (n, 1)
        b = mat.matmul(&b)?;
        let norm_b = linalg::norm(&b);
        b.div_assign(norm_b)?;
    }

    // (1, n) @ (n, n) @ (n, 1)
    let eig_value = b.transpose_last()?
        .matmul(&mat)?
        .matmul(&b)?
        .to_scalar()?;
    let eig_vector = b;

    Ok((eig_value, eig_vector.squeeze(1)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    fn check_orthogonal<T: FloatDType>(mat: &NdArray<T>, tol: T) -> bool {
        let mat_t = mat.transpose_last().unwrap();
        let prod = mat_t.matmul(mat).unwrap();
        let n = prod.dims2().unwrap().0;
        let id = NdArray::<T>::eye(n).unwrap();
        prod.allclose(&id, tol.to_f64(), tol.to_f64())
    }

    #[test]
    fn test_svd_identity() {
        let a = NdArray::<f64>::eye(3).unwrap();
        let (u, s, v) = svd(&a, 1e-12).unwrap();

        // 检查奇异值
        for &sigma in s.iter() {
            assert!((sigma - 1.0).abs() < 1e-10);
        }

        // 检查重构
        let sigma = NdArray::diag(&s).unwrap();
        let rec = u.matmul(&sigma).unwrap().matmul(&v.transpose_last().unwrap()).unwrap();
        assert!(rec.allclose(&a, 1e-10, 1e-10));

        // 检查正交性
        assert!(check_orthogonal(&u, 1e-10));
        assert!(check_orthogonal(&v, 1e-10));
    }

    #[test]
    fn test_svd_diagonal_matrix() {
        let a = NdArray::new(&[
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, 2.0],
        ]).unwrap();

        let (u, s, v) = svd(&a, 1e-12).unwrap();
        let sigma = NdArray::diag(&s).unwrap();
        let rec = u.matmul(&sigma).unwrap().matmul(&v.transpose_last().unwrap()).unwrap();
        assert!(rec.allclose(&a, 1e-10, 1e-10));

        assert!(check_orthogonal(&u, 1e-10));
        assert!(check_orthogonal(&v, 1e-10));
    }

    #[test]
    fn test_svd_rectangular_matrix_m_gt_n() {
        let a = NdArray::<f64>::randn(0.0, 1.0, (5, 3)).unwrap();
        let (u, s, v) = svd(&a, 1e-10).unwrap();

        let sigma = NdArray::diag(&s).unwrap();
        let rec = u.matmul(&sigma).unwrap().matmul(&v.transpose_last().unwrap()).unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));

        assert!(check_orthogonal(&u, 1e-6));
        assert!(check_orthogonal(&v, 1e-6));
    }

    #[test]
    fn test_svd_rectangular_matrix_m_lt_n() {
        let a = NdArray::<f64>::randn(0.0, 1.0, (3, 5)).unwrap();
        let (u, s, v) = svd(&a, 1e-10).unwrap();

        let sigma = NdArray::diag(&s).unwrap();
        let rec = u.matmul(&sigma).unwrap().matmul(&v.transpose_last().unwrap()).unwrap();
        assert!(rec.allclose(&a, 1e-6, 1e-6));

        assert!(check_orthogonal(&u, 1e-6));
        assert!(check_orthogonal(&v, 1e-6));
    }
}
