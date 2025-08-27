use approx::relative_eq;

use crate::{Result, Storage};

use super::{DType, FloatCategory, FloatDType, NumDType, WithDType};

impl WithDType for f64 {
    const DTYPE: DType = DType::F64;
}

impl NumDType for f64 {
    type Category = FloatCategory;

    fn from_f64(v: f64) -> Self {
        v as f64
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as f64
    }

    fn to_usize(self) -> usize {
        self as usize
    }

    fn minimum(lhs: Self, rhs: Self) -> Self {
        if lhs > rhs { rhs } else { lhs }
    }

    fn maximum(lhs: Self, rhs: Self) -> Self {
        if lhs < rhs { rhs } else { lhs }
    }

    fn close(self, other: Self, rtol: f64, atol: f64) -> bool {
        relative_eq!(self, other, epsilon = atol, max_relative = rtol)
    }

    fn to_range_storage(start: Self, end: Self) -> Result<Storage<Self>> {
        let mut vec = vec![];
        let mut v = start;
        while v < end {
            vec.push(v);
            v += 1.0;
        }
        Ok(Storage::new(vec))
    }
}

impl FloatDType for f64 {
}