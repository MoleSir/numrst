use crate::{Result, Storage};
use super::{DType, IntCategory, IntDType, NumDType, UnsignedIntDType, WithDType};

impl WithDType for usize {
    const DTYPE: DType = DType::U32;
}

impl NumDType for usize {
    type Category = IntCategory;

    fn from_f64(v: f64) -> Self {
        v as usize
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v
    }

    fn to_usize(self) -> usize {
        self
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

impl IntDType for usize {}
impl UnsignedIntDType for usize {}
