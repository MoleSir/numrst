use approx::relative_eq;

use crate::{Result, Storage};

use super::{DType, FloatDType, FloatGroup, WithDType};

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;
    type Group = FloatGroup;

    fn min_value() -> Self {
        <f32>::MIN
    }

    fn max_value() -> Self {
        <f32>::MAX
    }

    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(v: usize) -> Self {
        v as f32
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

    fn close(self, other: Self, rtol: f64, atol: f64) -> bool {
        relative_eq!(self, other, epsilon = atol as f32, max_relative = rtol as f32)
    }

    fn dtype() -> DType {
        DType::F32
    }

    fn to_storage(data: Vec<Self>) -> Result<Storage<Self>> {
        Ok(Storage::new(data))
    }

    fn to_filled_storage(self, len: usize) -> Result<Storage<Self>> {
        Self::to_storage(vec![self; len])
    }
    
    fn to_range_storage(start: Self, end: Self) -> Result<Storage<Self>> {
        let mut vec = vec![];
        let mut v = start;
        while v < end {
            vec.push(v);
            v += 1.0;
        }
        Self::to_storage(vec)
    }
}

impl FloatDType for f32 {
}