use crate::{Error, IndexOp, NdArray, NumDType, Result};

pub fn eye<T: NumDType>(size: usize) -> Result<NdArray<T>> {
    let arr = NdArray::<T>::zeros((size, size))?;
    for n in 0..size {
        arr.index((n, n))?.affine_assign(T::one(), T::one())?;
    }
    Ok(arr)
}

pub enum Norm {
    Fro,         // Frobenius 范数
    Inf,         // ∞ 范数
    NegInf,      // -∞ 范数
    P(f64),      // p 范数
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

pub fn dot<T: NumDType>(a: &NdArray<T>, b: &NdArray<T>) -> Result<T> {
    let n = a.dims1()?;
    let m = b.dims1()?;
    if m != n {
        return Err(Error::Msg("dot only apply two size len vector".into()));
    }

    let result = a.iter().zip(b.iter()).map(|(a, b)| a * b).sum::<T>();

    Ok(result)
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
