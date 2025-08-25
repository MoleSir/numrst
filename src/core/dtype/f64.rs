use crate::{Result, Storage};

use super::{DType, FloatDType, Scalar, WithDType};

impl WithDType for f64 {
    const DTYPE: DType = DType::F64;

    fn min_value() -> Self {
        <f64>::MIN
    }

    fn max_value() -> Self {
        <f64>::MAX
    }

    fn from_f64(v: f64) -> Self {
        v as f64
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn to_scalar(self) -> Scalar {
        Scalar::F64(self)
    }

    fn from_usize(v: usize) -> Self {
        v as f64
    }

    fn to_usize(self) -> usize {
        self as usize
    }

    fn dtype(&self) -> DType {
        DType::F64
    }

    fn to_storage(data: Vec<Self>) -> Result<Storage> {
        Ok(Storage::F64(data))
    }

    fn to_filled_storage(self, len: usize) -> Result<Storage> {
        Self::to_storage(vec![self; len])
    }
    
    fn to_range_storage(start: Self, end: Self) -> Result<Storage> {
        let mut vec = vec![];
        let mut v = start;
        while v < end {
            vec.push(v);
            v += 1.0;
        }
        Self::to_storage(vec)
    }
}

impl FloatDType for f64 {
}