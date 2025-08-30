use crate::{Result, Storage};
use super::{DType, IntCategory, IntDType, NumDType, SignedIntDType, WithDType};

impl WithDType for i16 {
    const DTYPE: DType = DType::I16;
}

impl NumDType for i16 {
    type Category = IntCategory;

    fn from_f64(v: f64) -> Self {
        v as i16
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as i16
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

impl IntDType for i16 {}

impl SignedIntDType for i16 {
    fn abs(self) -> i16 {
        if self > 0 { self } else { -self }
    }

    fn neg(self) -> i16 {
        -self
    }
}
