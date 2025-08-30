use crate::{Result, Storage};

use super::{DType, IntCategory, IntDType, NumDType, UnsignedIntDType, WithDType};

impl WithDType for u16 {
    const DTYPE: DType = DType::U16;
}

impl NumDType for u16 {
    type Category = IntCategory;

    fn from_f64(v: f64) -> Self {
        v as u16
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as u16
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

impl IntDType for u16 {}
impl UnsignedIntDType for u16 {}
