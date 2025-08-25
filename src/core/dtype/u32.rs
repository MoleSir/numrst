use crate::{Result, Storage};

use super::{DType, IntDType, IntGroup, WithDType};

impl WithDType for u32 {
    const DTYPE: DType = DType::U32;
    type Group = IntGroup;

    fn dtype() -> DType {
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

    fn minimum(lhs: Self, rhs: Self) -> Self {
        if lhs > rhs { rhs } else { lhs }
    }

    fn maximum(lhs: Self, rhs: Self) -> Self {
        if lhs < rhs { rhs } else { lhs }
    }

    fn close(self, other: Self, _rtol: f64, _atol: f64) -> bool {
        self == other
    }

    fn to_storage(data: Vec<Self>) -> Result<Storage<Self>> {
        Ok(Storage::new(data))
    }

    fn to_filled_storage(self, len: usize) -> Result<Storage<Self>> {
        Self::to_storage(vec![self; len])
    }
    
    fn to_range_storage(start: Self, end: Self) -> Result<Storage<Self>> {
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