use std::str::Utf8Error;
use zip::result::ZipError;

use crate::{io::{NpyError, NrstError}, linalg::LinalgError, DType, Range, Shape};

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
    #[error("Index '{index}' out of range at storage({storage_len}) in take method")]
    IndexOutOfRangeTake {
        storage_len: usize,
        index: usize,
    },

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

    #[error("try to repeat {repeats} for shape {shape}")]
    RepeatRankOutOfRange {
        repeats: Shape,
        shape: Shape,
    },

    // === Shape Errors ===
    #[error("unexpected element size in {op}, expected: {expected}, got: {got}")]
    ElementSizeMismatch {
        expected: usize,
        got: usize,
        op: &'static str
    },

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

    #[error("source ndarray shape {src:?} mismatch with condition shape {condition:?}")]
    ShapeMismatchFilter {
        src: Shape,
        condition: Shape, 
    },

    #[error("mask ndarray shape {mask:?} mismatch with {who} shape")]
    ShapeMismatchSelect {
        mask: Shape,
        who: &'static str,
    },

    #[error("dst ndarray shape {dst:?} mismatch with src ndarray {src} shape")]
    ShapeMismatchCopyFrom {
        dst: Shape,
        src: Shape,
    },

    // === Op Specific Errors ===
    #[error("narrow range invalid args {msg}: {shape:?}, dim: {dim}, range: {range}")]
    NarrowRangeInvalidArgs {
        shape: Shape,
        dim: usize,
        range: Range,
        msg: &'static str,
    },

    #[error("narrow invalid args {msg}: {shape:?}, dim: {dim}, start: {start}, len:{len}")]
    NarrowInvalidArgs {
        shape: Shape,
        dim: usize,
        start: usize,
        len: usize,
        msg: &'static str,
    },

    #[error("can squeeze {dim} dim of {shape:?}(not 1)")]
    SqueezeDimNot1 {
        shape: Shape,
        dim: usize,
    },

    #[error("cannot broadcast {src_shape:?} to {dst_shape:?}")]
    BroadcastIncompatibleShapes { src_shape: Shape, dst_shape: Shape },

    #[error("{op} expects at least one tensor")]
    OpRequiresAtLeastOneNdArray { op: &'static str },

    #[error("rand error because {0}")]
    Rand(String),

    #[error("ndarray is not a scalar")]
    NotScalar,

    // === View ===
    #[error("len mismatch with lhs {lhs} and rhs {rhs}")]
    LenMismatchVectorDot {
        lhs: usize,
        rhs: usize,
    },

    #[error("index {index} of out range in {len} len vector")]
    VectorIndexOutOfRange {
        len: usize,
        index: usize,
    },

    #[error("{position} index {index} of out range in {len} len matrix")]
    MatrixIndexOutOfRange {
        len: usize,
        index: usize,
        position: &'static str,
    },

    // === Transparent Errors ===
    #[error(transparent)]
    Linalg(#[from] LinalgError),

    #[error(transparent)]
    Npy(#[from] NpyError),

    #[error(transparent)]
    Nrst(#[from] NrstError),

    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// Utf8 parse error.
    #[error(transparent)]
    FromUtf8(#[from] std::string::FromUtf8Error),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Utf8(#[from] Utf8Error),

    #[error(transparent)]
    Zip(#[from] ZipError),

    // === Utils ===
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