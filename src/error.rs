use crate::{DType, Shape};


#[derive(Debug, thiserror::Error)]
pub enum Error {
    // === DType Errors ===
    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("dtype mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    DTypeMismatchBinaryOp {
        lhs: DType,
        rhs: DType,
        op: &'static str,
    },

    #[error("unsupported dtype {0:?} for op {1}")]
    UnsupportedDTypeForOp(DType, &'static str),

    // === Dimension Index Errors ===
    #[error("{op}: dimension index {dim} out of range for shape {shape:?}")]
    DimOutOfRange {
        shape: Shape,
        dim: i32,
        op: &'static str,
    },

    #[error("{op}: duplicate dim index {dims:?} for shape {shape:?}")]
    DuplicateDimIndex {
        shape: Shape,
        dims: Vec<usize>,
        op: &'static str,
    },

    // === Shape Errors ===
    #[error("unexpected rank, expected: {expected}, got: {got} ({shape:?})")]
    UnexpectedNumberOfDims {
        expected: usize,
        got: usize,
        shape: Shape,
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedShape {
        msg: String,
        expected: Shape,
        got: Shape,
    },

    #[error(
        "Shape mismatch, got buffer of size {buffer_size} which is compatible with shape {shape:?}"
    )]
    ShapeMismatch { buffer_size: usize, shape: Shape },

    #[error("shape mismatch in {op}, lhs: {lhs:?}, rhs: {rhs:?}")]
    ShapeMismatchBinaryOp {
        lhs: Shape,
        rhs: Shape,
        op: &'static str,
    },

    #[error("shape mismatch in cat for dim {dim}, shape for arg 1: {first_shape:?} shape for arg {n}: {nth_shape:?}")]
    ShapeMismatchCat {
        dim: usize,
        first_shape: Shape,
        n: usize,
        nth_shape: Shape,
    },

    #[error("Cannot divide tensor of shape {shape:?} equally along dim {dim} into {n_parts}")]
    ShapeMismatchSplit {
        shape: Shape,
        dim: usize,
        n_parts: usize,
    },

    #[error("{op} can only be performed on a single dimension")]
    OnlySingleDimension { op: &'static str, dims: Vec<usize> },

    #[error("empty tensor for {op}")]
    EmptyTensor { op: &'static str },

    // === Op Specific Errors ===
    #[error("narrow invalid args {msg}: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
    NarrowInvalidArgs {
        shape: Shape,
        dim: usize,
        start: usize,
        len: usize,
        msg: &'static str,
    },

    #[error("{op} invalid index {index} with dim size {size}")]
    InvalidIndex {
        op: &'static str,
        index: usize,
        size: usize,
    },

    #[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
    BroadcastIncompatibleShapes { src_shape: Shape, dst_shape: Shape },

    #[error("cannot set variable {msg}")]
    CannotSetVar { msg: &'static str },

    // Box indirection to avoid large variant.
    #[error("{op} only supports contiguous tensors")]
    RequiresContiguous { op: &'static str },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneTensor { op: &'static str },

    #[error("{op} expects at least two tensors")]
    OpRequiresAtLeastTwoTensors { op: &'static str },

    #[error("backward is not supported for {op}")]
    BackwardNotSupported { op: &'static str },

    // === Other Errors ===
    #[error("the candle crate has not been built with cuda support")]
    NotCompiledWithCudaSupport,

    #[error("the candle crate has not been built with metal support")]
    NotCompiledWithMetalSupport,

    #[error("cannot find tensor {path}")]
    CannotFindTensor { path: String },

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// Utf8 parse error.
    #[error(transparent)]
    FromUtf8(#[from] std::string::FromUtf8Error),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("{context}\n{inner}")]
    Context {
        inner: Box<Self>,
        context: String,
    },

    /// User generated error message
    #[error("{0}")]
    Msg(String),

    #[error("unwrap none")]
    UnwrapNone,
}
pub type Result<T> = std::result::Result<T, Error>;

pub trait Context: Sized {
    fn context(self, ctx: impl Into<String>) -> Self;
}

impl<T> Context for Result<T> {
    fn context(self, ctx: impl Into<String>) -> Self {
        match self {
            Ok(v) => Ok(v),
            Err(e) => Err(Error::Context { inner: Box::new(e), context: ctx.into() })
        }
    }
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()))
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()))
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()))
    };
}