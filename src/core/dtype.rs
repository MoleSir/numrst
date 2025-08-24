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

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::I32 => write!(f, "32-bit signed"),
            Self::U32 => write!(f, "32-bit unsigned"),
            Self::F32 => write!(f, "32-bit float"),
            Self::F64 => write!(f, "64-bit float"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
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

    fn min_value() -> Self;
    fn max_value() -> Self;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn to_scalar(self) -> Scalar;
    fn dtype(&self) -> DType;
    
    fn to_storage(data: Vec<Self>) -> Result<Storage>;
    fn to_filled_storage(self, len: usize) -> Result<Storage>;
    fn to_range_storage(start: Self, end: Self) -> Result<Storage>;
}

macro_rules! with_int_dtype {
    ($ty:ty, $dtype:ident) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn min_value() -> Self {
                <$ty>::MIN
            }

            fn max_value() -> Self {
                <$ty>::MAX
            }

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

            fn to_filled_storage(self, len: usize) -> Result<Storage> {
                Self::to_storage(vec![self; len])
            }
            
            fn to_range_storage(start: Self, end: Self) -> Result<Storage> {
                let vec: Vec<_> = (start..end).collect();
                Self::to_storage(vec)
            }
        }

        impl IntDType for $ty {
            fn is_true(&self) -> bool {
                *self != 0
            }
            fn as_usize(&self) -> usize {
                *self as usize
            }
        }
    }
}

pub trait IntDType: WithDType + num_traits::Bounded {
    fn is_true(&self) -> bool;
    fn as_usize(&self) -> usize;
}

pub trait FloatDType: WithDType {}

with_int_dtype!(u32, U32);
with_int_dtype!(i32, I32);

macro_rules! with_float_dtype {
    ($ty:ty, $dtype:ident) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn min_value() -> Self {
                <$ty>::MIN
            }

            fn max_value() -> Self {
                <$ty>::MAX
            }

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

            fn to_filled_storage(self, len: usize) -> Result<Storage> {
                Self::to_storage(vec![self; len])
            }
            
            fn to_range_storage(start: Self, end: Self) -> Result<Storage> {
                let mut vec = vec![];
                let mut v = start;
                while v < end {
                    vec.push(v);
                    v += 1.0;
                }
                Self::to_storage(vec)
            }
        }

        impl FloatDType for $ty {
        }
    }
}

with_float_dtype!(f32, F32);
with_float_dtype!(f64, F64);