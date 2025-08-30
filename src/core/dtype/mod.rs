mod f32;
mod f64;
mod u32;
mod usize;
mod i32;
mod bool;
mod u8;
mod i8;
mod u16;
mod i16;

use crate::Result;
use super::Storage;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,  // boolean
    U8,    // unsigned 8-bit
    I8,    // signed 8-bit
    U16,   // unsigned 16-bit
    I16,   // signed 16-bit
    U32,   // unsigned 32-bit
    I32,   // signed 32-bit
    USize, // unsigned size
    F32,   // 32-bit float
    F64,   // 64-bit float
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::Bool => std::mem::size_of::<bool>(),
            DType::U8 => std::mem::size_of::<u8>(),
            DType::I8 => std::mem::size_of::<i8>(),
            DType::I16 => std::mem::size_of::<i16>(),
            DType::U16 => std::mem::size_of::<u16>(),
            DType::I32 => std::mem::size_of::<i32>(),
            DType::U32 => std::mem::size_of::<u32>(),
            DType::USize => std::mem::size_of::<usize>(),
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
            Self::Bool => write!(f, "boolean"),
            Self::U8   => write!(f, "uint8"),
            Self::I8   => write!(f, "int8"),
            Self::I16 => write!(f, "int16"),
            Self::U16 => write!(f, "uint16"),
            Self::I32 => write!(f, "int32"),
            Self::U32 => write!(f, "uint32"),
            Self::USize => write!(f, "usize"),
            Self::F32 => write!(f, "float32"),
            Self::F64 => write!(f, "float64"),
        }
    }
}

pub trait WithDType:
    Sized
    + Copy
    + std::cmp::PartialOrd
    + std::cmp::PartialEq
    + std::fmt::Display
    + 'static
    + Send
    + Sync
{
    const DTYPE: DType;
}

pub trait NumDType : 
    WithDType 
  + num_traits::Num    
  + num_traits::Bounded
  + std::iter::Sum
  + std::iter::Product
  + std::ops::AddAssign
{
    type Category: NumCategory;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_usize(v: usize) -> Self;
    fn to_usize(self) -> usize;

    fn minimum(lhs: Self, rhs: Self) -> Self;
    fn maximum(lhs: Self, rhs: Self) -> Self;
    fn close(self, other: Self, rtol: f64, atol: f64) -> bool;

    fn to_range_storage(start: Self, end: Self) -> Result<Storage<Self>>;
}

pub trait IntDType: 
    NumDType<Category = IntCategory>
    + num_traits::Bounded 
{
    fn is_true(self) -> bool {
        self != Self::zero()
    }
}

pub trait SignedIntDType : 
    IntDType 
  + num_traits::Signed 
{
    fn abs(self) -> Self;
    fn neg(self) -> Self;
}

pub trait UnsignedIntDType : 
    IntDType 
  + num_traits::Unsigned 
{
}

pub trait FloatDType: 
    NumDType<Category = FloatCategory>
    + num_traits::Float
{
}

pub trait NumCategory {}
pub struct IntCategory {}
pub struct FloatCategory {}

impl NumCategory for IntCategory {}
impl NumCategory for FloatCategory {}

pub trait DTypeConvert<To: WithDType>: WithDType {
    fn convert(self) -> To;
}

macro_rules! impl_dtype_convert_from {
    ($from:ty, { $($to:ty),* }) => {
        $(
            impl DTypeConvert<$to> for $from {
                #[inline]
                fn convert(self) -> $to {
                    self as $to
                }
            }
        )*
    };
}

impl_dtype_convert_from!(i8,  { i8, u8, i16, u16, i32, u32, usize, f32, f64 });
impl_dtype_convert_from!(u8,  { i8, u8, i16, u16, i32, u32, usize, f32, f64 });
impl_dtype_convert_from!(i16, { i8, u8, i16, u16, i32, u32, usize, f32, f64 });
impl_dtype_convert_from!(u16, { i8, u8, i16, u16, i32, u32, usize, f32, f64 });
impl_dtype_convert_from!(i32, { i8, u8, i16, u16, i32, u32, usize, f32, f64 });
impl_dtype_convert_from!(u32, { i8, u8, i16, u16, i32, u32, usize, f32, f64 });
impl_dtype_convert_from!(f32, { i8, u8, i16, u16, i32, u32, usize, f32, f64 });
impl_dtype_convert_from!(f64, { i8, u8, i16, u16, i32, u32, usize, f32, f64 });
impl_dtype_convert_from!(usize, { i8, u8, i16, u16, i32, u32, usize, f32, f64 });

macro_rules! impl_with_bool {
    ($($dtype:ty),*) => {
        $(
            impl DTypeConvert<bool> for $dtype {
                #[inline]
                fn convert(self) -> bool {
                    self != 0 as $dtype
                }
            }
            
            impl DTypeConvert<$dtype> for bool {
                #[inline]
                fn convert(self) -> $dtype {
                    if self { 1 as $dtype } else { 0 as $dtype }
                }
            }
        )*
    };
}

impl_with_bool!(f32, f64, i8, u8, i16, u16, i32, u32, usize);