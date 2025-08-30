use crate::{Result, Storage};

use super::{DType, DTypeConvert, IntCategory, IntDType, NumDType, UnsignedIntDType, WithDType};

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

impl DTypeConvert<u8> for usize { fn convert(self) -> u8 { self as u8 } }
impl DTypeConvert<i32> for usize { fn convert(self) -> i32 { self as i32 } }
impl DTypeConvert<u32> for usize { fn convert(self) -> u32 { self as u32 } }
impl DTypeConvert<usize> for usize { fn convert(self) -> usize { self } }
impl DTypeConvert<f32> for usize { fn convert(self) -> f32 { self as f32 } }
impl DTypeConvert<f64> for usize { fn convert(self) -> f64 { self as f64 } }
impl DTypeConvert<bool> for usize { fn convert(self) -> bool { self != 0 } }