use crate::NdArray;

pub fn norm(array: &NdArray) -> f64 {
    let sum = array.iter()
        .map(|scale| scale.to_f64())
        .map(|v| v.powf(2.0))
        .sum::<f64>();
    sum.sqrt()
}

