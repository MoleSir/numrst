mod f32;
mod f64;
mod u32;
mod i32;
mod bool;
use crate::Result;
use super::Storage;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    Bool, // boolean
    I32,  // signed 32-bit
    U32,  // unsigned 32-bit
    F32,  // 32-bit float
    F64,  // 64-bit float
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::Bool => std::mem::size_of::<bool>(),
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
            Self::Bool => write!(f, "boolean"),
            Self::I32 => write!(f, "32-bit signed"),
            Self::U32 => write!(f, "32-bit unsigned"),
            Self::F32 => write!(f, "32-bit float"),
            Self::F64 => write!(f, "64-bit float"),
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
    fn dtype() -> DType;
    
    // fn to_storage(data: Vec<Self>) -> Result<Storage<Self>>;
    // fn to_filled_storage(self, len: usize) -> Result<Storage<Self>>;
}

pub trait NumDType : 
    WithDType 
  + num_traits::Num    
  + std::iter::Sum
  + std::iter::Product
  + std::ops::AddAssign

{
    type Category: NumCategory;

    fn min_value() -> Self;
    fn max_value() -> Self;

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
    NumDType
    + num_traits::Bounded 
{
    fn is_true(self) -> bool {
        self != Self::zero()
    }
    fn abs(self) -> Self;
    fn neg(self) -> Self;
}

pub trait FloatDType: 
    NumDType<Category = FloatCategory>
    + num_traits::Float
{
}

pub trait BoolDType:
    NumDType<Category = IntCategory>
{

}

pub trait NumCategory {}
pub struct IntCategory {}
pub struct FloatCategory {}

impl NumCategory for IntCategory {}
impl NumCategory for FloatCategory {}