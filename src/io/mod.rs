mod npy;
mod nrst;
mod utils;
pub use npy::*;
pub use nrst::*;

use crate::{DType, NdArray, Result};

#[derive(Clone)]
pub enum DynamicNdArray {
    Bool(NdArray<bool>),
    U32(NdArray<u32>),
    I32(NdArray<i32>),
    USize(NdArray<usize>),
    F32(NdArray<f32>),
    F64(NdArray<f64>),
}

impl DynamicNdArray {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::U32(_) => DType::U32,
            Self::I32(_) => DType::I32,
            Self::USize(_) => DType::USize,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn bool(self) -> Result<NdArray<bool>> {
        if let Self::Bool(arr) = self {
            Ok(arr)
        } else {
            crate::bail!("Expect bool dtype, but got {:?}", self.dtype())
        }
    }

    pub fn u32(self) -> Result<NdArray<u32>> {
        if let Self::U32(arr) = self {
            Ok(arr)
        } else {
            crate::bail!("Expect u32 dtype, but got {:?}", self.dtype())
        }
    }

    pub fn i32(self) -> Result<NdArray<i32>> {
        if let Self::I32(arr) = self {
            Ok(arr)
        } else {
            crate::bail!("Expect i32 dtype, but got {:?}", self.dtype())
        }
    }

    pub fn usize(self) -> Result<NdArray<usize>> {
        if let Self::USize(arr) = self {
            Ok(arr)
        } else {
            crate::bail!("Expect usize dtype, but got {:?}", self.dtype())
        }
    }

    pub fn f32(self) -> Result<NdArray<f32>> {
        if let Self::F32(arr) = self {
            Ok(arr)
        } else {
            crate::bail!("Expect f32 dtype, but got {:?}", self.dtype())
        }
    }

    pub fn f64(self) -> Result<NdArray<f64>> {
        if let Self::F64(arr) = self {
            Ok(arr)
        } else {
            crate::bail!("Expect f64 dtype, but got {:?}", self.dtype())
        }
    }
}

impl DynamicNdArray {
    pub fn to_u32(self) -> NdArray<u32> {
        match self {
            Self::I32(arr) => arr.to_dtype(),
            Self::U32(arr) => arr,
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::Bool(arr) => arr.to_dtype(),
        }
    }

    pub fn to_i32(self) -> NdArray<i32> {
        match self {
            Self::I32(arr) => arr,
            Self::U32(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::Bool(arr) => arr.to_dtype(),
        }
    }

    pub fn to_usize(self) -> NdArray<usize> {
        match self {
            Self::I32(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
            Self::USize(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
        }
    }

    pub fn to_f32(self) -> NdArray<f32> {
        match self {
            Self::I32(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::F32(arr) => arr,
            Self::F64(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::Bool(arr) => arr.to_dtype(),
        }
    }

    pub fn to_f64(self) -> NdArray<f64> {
        match self {
            Self::I32(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr,
            Self::USize(arr) => arr.to_dtype(),
            Self::Bool(arr) => arr.to_dtype(),
        }
    }

    pub fn to_bool(self) -> NdArray<bool> {
        match self {
            Self::I32(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::Bool(arr) => arr,
        }
    }
}