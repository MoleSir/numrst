use crate::{Error, Matrix, NdArray, NumDType, Result, ToMatrixView, ToVectorView};

pub enum Norm {
    Fro,         // Frobenius 
    Inf,         // ∞ 
    NegInf,      // -∞ 
    P(f64),      // p
}

pub fn norm<T: NumDType>(array: &NdArray<T>, ord: Norm) -> f64 {
    match ord {
        Norm::Fro | Norm::P(2.0) => {
            array.iter()
                .map(|x| x.to_f64().powi(2))
                .sum::<f64>()
                .sqrt()
        }
        Norm::P(1.0) => array.iter().map(|x| x.to_f64().abs()).sum(),
        Norm::P(p) => array.iter().map(|x| x.to_f64().abs().powf(p)).sum::<f64>().powf(1.0 / p),
        Norm::Inf => array.iter().map(|x| x.to_f64().abs()).fold(0.0, f64::max),
        Norm::NegInf => array.iter().map(|x| x.to_f64().abs()).fold(f64::INFINITY, f64::min),
    }
}

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
}
