use crate::{Result, Storage};
use super::{DType, DTypeConvert, IntCategory, IntDType, NumDType, SignedIntDType, WithDType};

impl WithDType for i32 {
    const DTYPE: DType = DType::I32;
}

impl NumDType for i32 {
    type Category = IntCategory;

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

impl IntDType for i32 {}

impl SignedIntDType for i32 {
    fn abs(self) -> i32 {
        if self > 0 { self } else { -self }
    }

    fn neg(self) -> i32 {
        -self
    }
}

impl DTypeConvert<u8> for i32 { fn convert(self) -> u8 { self as u8 } }
impl DTypeConvert<i32> for i32 { fn convert(self) -> i32 { self } }
impl DTypeConvert<u32> for i32 { fn convert(self) -> u32 { self as u32 } }
impl DTypeConvert<usize> for i32 { fn convert(self) -> usize { self as usize } }
impl DTypeConvert<f32> for i32 { fn convert(self) -> f32 { self as f32 } }
impl DTypeConvert<f64> for i32 { fn convert(self) -> f64 { self as f64 } }
impl DTypeConvert<bool> for i32 { fn convert(self) -> bool { self != 0 } }
