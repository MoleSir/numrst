use approx::relative_eq;

use crate::Result;

use super::Storage;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    I32,  // signed 32-bit
    U32,  // unsigned 32-bit
    F32,  // 32-bit float
    F64,  // 64-bit float
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::I32 => std::mem::size_of::<i32>(),
            DType::U32 => std::mem::size_of::<u32>(),
            DType::F32 => std::mem::size_of::<f32>(),
            DType::F64 => std::mem::size_of::<f64>(),
        }
    }

    pub fn is_int(&self) -> bool {
        matches!(self, DType::I32 | DType::U32)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
}

impl Scalar {
    pub fn zero(dtype: DType) -> Self {
        match dtype {
            DType::U32 => Scalar::U32(0),
            DType::I32 => Scalar::I32(0),
            DType::F32 => Scalar::F32(0.0),
            DType::F64 => Scalar::F64(0.0),
        }
    }

    pub fn one(dtype: DType) -> Self {
        match dtype {
            DType::U32 => Scalar::U32(1),
            DType::I32 => Scalar::I32(1),
            DType::F32 => Scalar::F32(1.0),
            DType::F64 => Scalar::F64(1.0),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Scalar::U32(_) => DType::U32,
            Scalar::I32(_) => DType::I32,
            Scalar::F32(_) => DType::F32,
            Scalar::F64(_) => DType::F64,
        }
    }

    pub fn to_f64(&self) -> f64 {
        match self {
            Scalar::U32(v) => *v as f64,
            Scalar::I32(v) => *v as f64,
            Scalar::F32(v) => *v as f64,
            Scalar::F64(v) => *v,
        }
    }

    pub fn allclose(&self, other: &Self, rtol: f64, atol: f64) -> bool {
        match (self, other) {
            (Scalar::U32(a), Scalar::U32(b)) => a == b,
            (Scalar::I32(a), Scalar::I32(b)) => a == b,
            (Scalar::F32(a), Scalar::F32(b)) => {
                relative_eq!(a, b, epsilon = atol as f32, max_relative = rtol as f32)
            }
            (Scalar::F64(a), Scalar::F64(b)) => {
                relative_eq!(a, b, epsilon = atol, max_relative = rtol)
            }
            _ => false,
        }
    }
}

pub trait WithDType:
    Sized
    + Copy
    + num_traits::NumAssign
    + std::cmp::PartialOrd
    + std::fmt::Display
    + 'static
    + Send
    + Sync
{
    const DTYPE: DType;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn to_scalar(self) -> Scalar;
    fn dtype(&self) -> DType;

    fn to_storage(data: Vec<Self>) -> Result<Storage>;
    fn to_fill_storage(self, len: usize) -> Result<Storage>;
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn from_f64(v: f64) -> Self {
                v as $ty
            }

            fn to_f64(self) -> f64 {
                self as f64
            }

            fn to_scalar(self) -> Scalar {
                Scalar::$dtype(self)
            }

            fn dtype(&self) -> DType {
                DType::$dtype
            }

            fn to_storage(data: Vec<Self>) -> Result<Storage> {
                Ok(Storage::$dtype(data))
            }

            fn to_fill_storage(self, len: usize) -> Result<Storage> {
                Self::to_storage(vec![self; len])
            }
        }
    }
}

with_dtype!(u32, U32);
with_dtype!(i32, I32);
with_dtype!(f32, F32);
with_dtype!(f64, F64);

pub trait IntDType: WithDType + num_traits::Bounded {
    fn is_true(&self) -> bool;
    fn as_usize(&self) -> usize;
}

impl IntDType for i32 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for u32 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

pub trait FloatDType: WithDType {}

impl FloatDType for f32 {}
impl FloatDType for f64 {}
