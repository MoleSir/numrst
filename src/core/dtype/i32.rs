use crate::{Result, Storage};

use super::{DType, IntCategory, IntDType, NumDType, WithDType};

impl WithDType for i32 {
    const DTYPE: DType = DType::I32;

    fn dtype() -> DType {
        DType::I32
    }
}

impl NumDType for i32 {
    type Category = IntCategory;

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
    
    fn minimum(lhs: Self, rhs: Self) -> Self {
        if lhs > rhs { rhs } else { lhs }
    }

    fn maximum(lhs: Self, rhs: Self) -> Self {
        if lhs < rhs { rhs } else { lhs }
    }

    fn close(self, other: Self, _rtol: f64, _atol: f64) -> bool {
        self == other
    }

    fn to_range_storage(start: Self, end: Self) -> Result<Storage<Self>> {
        let vec: Vec<_> = (start..end).collect();
        Ok(Storage::new(vec))
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