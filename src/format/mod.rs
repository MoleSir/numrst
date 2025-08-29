mod npy;
mod nrst;
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

    /// 类型安全地获取 u32 NdArray
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
