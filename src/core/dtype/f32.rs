use crate::{Result, Storage};

use super::{DType, FloatDType, Scalar, WithDType};

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;

    fn min_value() -> Self {
        <f32>::MIN
    }

    fn max_value() -> Self {
        <f32>::MAX
    }

    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as f32
    }

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_scalar(self) -> Scalar {
        Scalar::F32(self)
    }

    fn dtype(&self) -> DType {
        DType::F32
    }

    fn to_storage(data: Vec<Self>) -> Result<Storage> {
        Ok(Storage::F32(data))
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

impl FloatDType for f32 {
}