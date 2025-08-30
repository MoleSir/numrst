mod npy;
mod nrst;
mod utils;
pub use npy::*;
pub use nrst::*;

use crate::{DType, NdArray, Result};

#[derive(Clone)]
pub enum DynamicNdArray {
    Bool(NdArray<bool>),
    U8(NdArray<u8>),
    I8(NdArray<i8>),
    U16(NdArray<u16>),
    I16(NdArray<i16>),
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
            Self::U8(_) => DType::U8,
            Self::I8(_) => DType::I8,
            Self::U16(_) => DType::U16,
            Self::I16(_) => DType::I16,
            Self::U32(_) => DType::U32,
            Self::I32(_) => DType::I32,
            Self::USize(_) => DType::USize,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }
}

macro_rules! impl_dynamic_ndarray_getter {
    ($($fn_name:ident => $variant:ident : $ty:ty),* $(,)?) => {
        impl DynamicNdArray {
            $(
                pub fn $fn_name(self) -> Result<NdArray<$ty>> {
                    if let Self::$variant(arr) = self {
                        Ok(arr)
                    } else {
                        crate::bail!("Expect {:?} dtype, but got {:?}", stringify!($variant), self.dtype())
                    }
                }
            )*
        }
    };
}

impl_dynamic_ndarray_getter! {
    bool => Bool: bool,
    u8 => U8: u8,
    i8 => I8: i8,
    u16 => U16: u16,
    i16 => I16: i16,
    u32 => U32: u32,
    i32 => I32: i32,
    usize => USize: usize,
    f32 => F32: f32,
    f64 => F64: f64,
}

impl DynamicNdArray {
    pub fn to_bool(self) -> NdArray<bool> {
        match self {
            Self::Bool(arr) => arr,
            Self::U8(arr) => arr.to_dtype(),
            Self::I8(arr) => arr.to_dtype(),
            Self::U16(arr) => arr.to_dtype(),
            Self::I16(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::I32(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
        }
    }

    pub fn to_u8(self) -> NdArray<u8> {
        match self {
            Self::U8(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
            Self::I8(arr) => arr.to_dtype(),
            Self::U16(arr) => arr.to_dtype(),
            Self::I16(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::I32(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
        }
    }

    pub fn to_i8(self) -> NdArray<i8> {
        match self {
            Self::I8(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
            Self::U8(arr) => arr.to_dtype(),
            Self::U16(arr) => arr.to_dtype(),
            Self::I16(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::I32(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
        }
    }

    pub fn to_u16(self) -> NdArray<u16> {
        match self {
            Self::U16(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
            Self::U8(arr) => arr.to_dtype(),
            Self::I8(arr) => arr.to_dtype(),
            Self::I16(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::I32(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
        }
    }

    pub fn to_i16(self) -> NdArray<i16> {
        match self {
            Self::I16(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
            Self::U8(arr) => arr.to_dtype(),
            Self::I8(arr) => arr.to_dtype(),
            Self::U16(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::I32(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
        }
    }

    pub fn to_u32(self) -> NdArray<u32> {
        match self {
            Self::U32(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
            Self::U8(arr) => arr.to_dtype(),
            Self::I8(arr) => arr.to_dtype(),
            Self::U16(arr) => arr.to_dtype(),
            Self::I16(arr) => arr.to_dtype(),
            Self::I32(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
        }
    }

    pub fn to_i32(self) -> NdArray<i32> {
        match self {
            Self::I32(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
            Self::U8(arr) => arr.to_dtype(),
            Self::I8(arr) => arr.to_dtype(),
            Self::U16(arr) => arr.to_dtype(),
            Self::I16(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
        }
    }

    pub fn to_usize(self) -> NdArray<usize> {
        match self {
            Self::USize(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
            Self::U8(arr) => arr.to_dtype(),
            Self::I8(arr) => arr.to_dtype(),
            Self::U16(arr) => arr.to_dtype(),
            Self::I16(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::I32(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
        }
    }

    pub fn to_f32(self) -> NdArray<f32> {
        match self {
            Self::F32(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
            Self::U8(arr) => arr.to_dtype(),
            Self::I8(arr) => arr.to_dtype(),
            Self::U16(arr) => arr.to_dtype(),
            Self::I16(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::I32(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::F64(arr) => arr.to_dtype(),
        }
    }

    pub fn to_f64(self) -> NdArray<f64> {
        match self {
            Self::F64(arr) => arr,
            Self::Bool(arr) => arr.to_dtype(),
            Self::U8(arr) => arr.to_dtype(),
            Self::I8(arr) => arr.to_dtype(),
            Self::U16(arr) => arr.to_dtype(),
            Self::I16(arr) => arr.to_dtype(),
            Self::U32(arr) => arr.to_dtype(),
            Self::I32(arr) => arr.to_dtype(),
            Self::USize(arr) => arr.to_dtype(),
            Self::F32(arr) => arr.to_dtype(),
        }
    }
}
