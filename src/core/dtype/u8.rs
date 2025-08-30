use crate::{Result, Storage};

use super::{DType, DTypeConvert, IntCategory, IntDType, NumDType, UnsignedIntDType, WithDType};

impl WithDType for u8 {
    const DTYPE: DType = DType::U8;
}

impl NumDType for u8 {
    type Category = IntCategory;

    fn from_f64(v: f64) -> Self {
        v as u8
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as u8
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

impl IntDType for u8 {}
impl UnsignedIntDType for u8 {}

impl DTypeConvert<i32> for u8 { fn convert(self) -> i32 { self as i32 } }
impl DTypeConvert<u32> for u8 { fn convert(self) -> u32 { self as u32 } }
impl DTypeConvert<u8> for u8 { fn convert(self) -> u8 { self } }
impl DTypeConvert<usize> for u8 { fn convert(self) -> usize { self as usize } }
impl DTypeConvert<f32> for u8 { fn convert(self) -> f32 { self as f32 } }
impl DTypeConvert<f64> for u8 { fn convert(self) -> f64 { self as f64 } }
impl DTypeConvert<bool> for u8 { fn convert(self) -> bool { self != 0 } }