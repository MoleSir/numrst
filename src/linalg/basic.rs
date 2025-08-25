use crate::{NdArray, NumDType};

pub fn norm<T: NumDType>(array: &NdArray<T>) -> f64 {
    let sum = array.iter()
        .map(|scale| scale.to_f64())
        .map(|v| v.powf(2.0))
        .sum::<f64>();
    sum.sqrt()
}

