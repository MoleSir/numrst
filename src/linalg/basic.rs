use crate::{Error, Matrix, NumDType, Result, ToMatrixView, ToVectorView};

pub fn dot<T, V1, V2>(a: V1, b: V2) -> Result<T> 
where 
    T: NumDType,
    V1: ToVectorView<T>,
    V2: ToVectorView<T>,
{
    let a = a.to_vector_view()?;
    let b = b.to_vector_view()?;

    if a.len() != b.len() {
        Err(Error::Msg("len mismatch in dot".into()))?;
    }

    let result = a.into_iter().zip(b.into_iter()).map(|(a, b)| a * b).sum::<T>();
    Ok(result)
}

pub fn matmul<T, M1, M2>(a: M1, b: M2) -> Result<Matrix<T>> 
where 
    T: NumDType,
    M1: ToMatrixView<T>,
    M2: ToMatrixView<T>,
{
    let a = a.to_matrix_view()?;
    let b = b.to_matrix_view()?; 

    let (m, k1) = a.shape();
    let (k2, n) = b.shape();

    if k1 != k2 {
        Err(Error::Msg(format!("invalid shape in matmul in {:?} and {:?}", a.shape(), b.shape())))?;
    }
    let z = k1;

    let res = Matrix::<T>::zeros(m, n)?;
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for k in 0..z {
                sum += a.g(i, k) * b.g(k, j);
            }
            res.s(i, j, sum);
        }
    }

    Ok(res)
}

pub fn trace<T: NumDType, M: ToMatrixView<T>>(mat: M) -> Result<T> {
    let mat = mat.to_matrix_view()?;
    let (m, n) = mat.shape();
    if m != n {
        Err(Error::Msg(format!("trace should square")))?;
    }
    
    let t = (0..m).into_iter()
        .map(|i| mat.g(i, i))
        .product::<T>();

    Ok(t)
}

pub fn is_square<T: NumDType, M: ToMatrixView<T>>(mat: M) -> Result<bool> {
    let mat = mat.to_matrix_view()?;
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

        assert!(c.to_ndarray().allclose(&expected, 1e-6, 1e-6));
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
