use crate::{Result, Storage};

use super::{DType, IntDType, Scalar, WithDType};

impl WithDType for i32 {
    const DTYPE: DType = DType::I32;

    fn dtype(&self) -> DType {
        DType::I32
    }

    fn min_value() -> Self {
        i32::MIN
    }

    fn max_value() -> Self {
        i32::MAX
    }

    fn from_f64(v: f64) -> Self {
        v as i32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as i32
    }

    fn to_usize(self) -> usize {
        self as usize
    }
    
    fn to_scalar(self) -> Scalar {
        Scalar::I32(self)
    }

    fn to_storage(data: Vec<Self>) -> Result<Storage> {
        Ok(Storage::I32(data))
    }

    fn to_filled_storage(self, len: usize) -> Result<Storage> {
        Self::to_storage(vec![self; len])
    }
    
    fn to_range_storage(start: Self, end: Self) -> Result<Storage> {
        let vec: Vec<_> = (start..end).collect();
        Self::to_storage(vec)
    }
}

impl IntDType for i32 {
    fn abs(self) -> i32 {
        if self > 0 { self } else { -self }
    }

    fn neg(self) -> i32 {
        -self
    }
}