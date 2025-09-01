use rand_distr::{Distribution, StandardNormal};
use crate::{FloatDType, NdArray, Result};

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
        eig_v.div_assign(norm(&eig_v))?;
        vs.push(eig_v.clone());

        // 2. sigma and left vector
        let av = a.matmul(&eig_v.unsqueeze(1)?)?.squeeze(1)?; // (m, n) @ (n, 1) = (m, 1) => (m,)
        let sigma = norm(&av);
        let eig_u = av.div(sigma)?; // (m,)

        us.push(eig_u.clone());  
        ss.push(sigma);

        // 3. update a copy
        // (m, 1) @ (1, n) => (m, n)
        let uv = eig_u.squeeze(1)?.matmul(&eig_v.squeeze(0)?)?;
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
        let norm_b = norm(&b);
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

fn norm<T: FloatDType>(m: &NdArray<T>) -> T {
    let sum = m.iter()
        .map(|v| v.powi(2))
        .sum::<T>();
    sum.sqrt()
}