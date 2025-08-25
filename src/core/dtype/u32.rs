use crate::{Result, Storage};

use super::{DType, IntDType, Scalar, WithDType};

impl WithDType for u32 {
    const DTYPE: DType = DType::U32;

    fn dtype(&self) -> DType {
        DType::U32
    }

    fn min_value() -> Self {
        u32::MIN
    }

    fn max_value() -> Self {
        u32::MAX
    }

    fn from_f64(v: f64) -> Self {
        v as u32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as u32
    }

    fn to_usize(self) -> usize {
        self as usize
    }

    fn to_scalar(self) -> Scalar {
        Scalar::U32(self)
    }

    fn to_storage(data: Vec<Self>) -> Result<Storage> {
        Ok(Storage::U32(data))
    }

    fn to_filled_storage(self, len: usize) -> Result<Storage> {
        Self::to_storage(vec![self; len])
    }
    
    fn to_range_storage(start: Self, end: Self) -> Result<Storage> {
        let vec: Vec<_> = (start..end).collect();
        Self::to_storage(vec)
    }
}

impl IntDType for u32 {
    fn abs(self) -> u32 {
        self
    }

    fn neg(self) -> u32 {
        unimplemented!("no neg function for u32")
    }
}