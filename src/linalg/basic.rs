use crate::{FloatDType, NdArray, NumDType, Result};
use super::LinalgError;

pub fn norm<T: FloatDType>(m: &NdArray<T>) -> T {
    let sum = m.iter()
        .map(|v| v.powi(2))
        .sum::<T>();
    sum.sqrt()
}

pub fn dot<T: NumDType>(a: &NdArray<T>, b: &NdArray<T>) -> Result<T> {
    let a = a.vector_view()?;
    let b = b.vector_view()?;

    if a.len() != b.len() {
        Err(LinalgError::VectorLenMismatch { len1: a.len(), len2: b.len(), op: "dot" })?;
    }

    let result = a.into_iter().zip(b.into_iter()).map(|(a, b)| a * b).sum::<T>();
    Ok(result)
}

pub fn outer<T: NumDType>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    let a = a.vector_view()?;
    let b = b.vector_view()?;

    let m = a.len();
    let n = b.len();

    let res_arr = NdArray::zeros((m, n))?;
    {
        let mut res = res_arr.matrix_view_mut().unwrap();
        for i in 0..m {
            for j in 0..n {
                let v = a.g(i) * b.g(j);
                res.s(i, j, v);
            }
        }
    }

    Ok(res_arr)
}


pub fn matmul<T: NumDType>(a: &NdArray<T>, b: &NdArray<T>) -> Result<NdArray<T>> {
    let a = a.matrix_view()?;
    let b = b.matrix_view()?; 

    let (m, k1) = a.shape();
    let (k2, n) = b.shape();

    if k1 != k2 {
        Err(LinalgError::MatmulShapeMismatch { lhs: a.shape(), rhs: b.shape() })?;
    }
    let z = k1;

    let res_arr = NdArray::<T>::zeros((m, n))?;
    {
        let mut res = res_arr.matrix_view_mut().unwrap();
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..z {
                    sum += a.g(i, k) * b.g(k, j);
                }
                res.s(i, j, sum);
            }
        }
    }

    Ok(res_arr)
}

pub fn mat_mul_vec<T: NumDType>(mat: &NdArray<T>, vec: &NdArray<T>) -> Result<NdArray<T>> {
    let mat = mat.matrix_view()?;
    let vec = vec.vector_view()?;

    let (m, k1) = mat.shape();
    let k2 = vec.len();

    if k1 != k2 {
        Err(LinalgError::MatMulVecShapeMismatch { shape: mat.shape(), len: vec.len() })?;
    }
    let k = k1; // (m, k) @ (k) = (m)

    let res_arr = NdArray::<T>::zeros(m)?;
    {
        let mut res = res_arr.vector_view_mut().unwrap();
        for i in 0..m {
            let mut sum = T::zero();
            for j in 0..k {
                sum += mat.g(i, j) * vec.g(j);
            }
            res.s(i, sum);
        }
    }

    Ok(res_arr)
}

pub fn vec_mul_mat<T: FloatDType>(vec: &NdArray<T>, mat: &NdArray<T>) -> Result<NdArray<T>> {
    let vec = vec.vector_view()?;
    let mat = mat.matrix_view()?;

    let k1 = vec.len();
    let (k2, n) = mat.shape();

    if k1 != k2 {
        Err(LinalgError::VecMulMatShapeMismatch { len: vec.len(), shape: mat.shape() })?;
    }
    let k = k1; // (1, k) @ (k, n) = (1, n)

    let res_arr = NdArray::<T>::zeros(n)?;
    {
        let mut res = res_arr.vector_view_mut().unwrap();
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..k {
                sum += vec.g(j) * mat.g(j, i);
            }
            res.s(i, sum);
        }
    }

    Ok(res_arr)
}

pub fn trace<T: NumDType>(mat: &NdArray<T>) -> Result<T> {
    let mat = mat.matrix_view()?;
    let (m, n) = mat.shape();
    if m != n {
        Err(LinalgError::ExpectMatrixSquare { shape: mat.shape(), op: "trace" })?;
    }
    
    let t = (0..m).into_iter()
        .map(|i| mat.g(i, i))
        .product::<T>();

    Ok(t)
}

pub fn is_square<T: NumDType>(mat: &NdArray<T>) -> Result<bool> {
    let mat = mat.matrix_view()?;
    let (m, n) = mat.shape();
    Ok(m == n)
}

#[cfg(test)]
mod test {
    use crate::{NdArray, linalg};

    #[test]
    fn test_dot_basic() {
        let a = NdArray::new(&[1., 2., 3.]).unwrap();
        let b = NdArray::new(&[4., 5., 6.]).unwrap();
        let res = linalg::dot(&a, &b).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(res, 32.);
    }

    #[test]
    fn test_dot_zero_vector() {
        let a = NdArray::new(&[0., 0., 0.]).unwrap();
        let b = NdArray::new(&[1., 2., 3.]).unwrap();
        let res = linalg::dot(&a, &b).unwrap();
        assert_eq!(res, 0.);
    }

    #[test]
    fn test_dot_negative() {
        let a = NdArray::new(&[1., -2., 3.]).unwrap();
        let b = NdArray::new(&[-1., 4., -3.]).unwrap();
        let res = linalg::dot(&a, &b).unwrap();
        // 1*-1 + -2*4 + 3*-3 = -1 -8 -9 = -18
        assert_eq!(res, -18.);
    }

    #[test]
    fn test_dot_incompatible_size() {
        let a = NdArray::new(&[1., 2., 3.]).unwrap();
        let b = NdArray::new(&[4., 5.]).unwrap();
        let res = linalg::dot(&a, &b);
        assert!(res.is_err());
    }


    #[test]
    fn test_matmul_basic() {
        let a = NdArray::new(&[
            [1., 2.],
            [3., 4.],
        ]).unwrap();
        let b = NdArray::new(&[
            [5., 6.],
            [7., 8.],
        ]).unwrap();

        let c = linalg::matmul(&a, &b).unwrap();
        let expected = NdArray::new(&[
            [19., 22.],
            [43., 50.],
        ]).unwrap();

        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_matmul_incompatible() {
        let a = NdArray::new(&[[1., 2.]]).unwrap(); // 1x2
        let b = NdArray::new(&[[3., 4.], [5., 6.], [7., 8.]]).unwrap(); // 3x2
        let res = linalg::matmul(&a, &b);
        assert!(res.is_err());
    }

    #[test]
    fn test_trace_square() {
        let a = NdArray::new(&[
            [1., 2.],
            [3., 4.],
        ]).unwrap();
        let t = linalg::trace(&a).unwrap();
        assert_eq!(t, 1. * 4.); // 注意 trace 是 sum 对角线，如果你之前实现是 sum，这里要改
    }

    #[test]
    fn test_trace_non_square() {
        let a = NdArray::new(&[
            [1., 2., 3.],
            [4., 5., 6.],
        ]).unwrap();
        let t = linalg::trace(&a);
        assert!(t.is_err());
    }

    #[test]
    fn test_is_square_true() {
        let a = NdArray::new(&[
            [1., 2.],
            [3., 4.],
        ]).unwrap();
        let res = linalg::is_square(&a).unwrap();
        assert!(res);
    }

    #[test]
    fn test_is_square_false() {
        let a = NdArray::new(&[
            [1., 2., 3.],
            [4., 5., 6.],
        ]).unwrap();
        let res = linalg::is_square(&a).unwrap();
        assert!(!res);
    }
}
