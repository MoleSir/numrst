
#[derive(Debug, thiserror::Error)]
pub enum LinalgError {
    #[error("unexpect singular matrix")]
    SingularMatrix,

    #[error("solve a not square matrix {0:?}")]
    NonSquareMatrix((usize, usize)),

    #[error("solve output len mismatch: expect {0}, got {1}")]
    SolveOutputLenMismatch(usize, usize),

    #[error("vector len mismatch with {len1} and {len2} in {op}")]
    VectorLenMismatch {
        len1: usize, len2: usize, op: &'static str,
    },

    #[error("vector index {index} out of range {len} in {op}")]
    VectorIndexOutOfRange {
        index: usize, len: usize, op: &'static str,
    },

    #[error("matrix mul shape with {lhs:?} and {rhs:?}")]
    MatmulShapeMismatch { 
        lhs: (usize, usize), rhs: (usize, usize),
    },

    #[error("matrix mul vec shape with {shape:?} and {len:?}")]
    MatMulVecShapeMismatch { 
        shape: (usize, usize), len: usize,
    },

    #[error("vec mul matrix shape with {len:?} and {shape:?}")]
    VecMulMatShapeMismatch { 
        len: usize, shape: (usize, usize),
    },

    #[error("expect a square matrix but got {shape:?} in {op}")]
    ExpectMatrixSquare {
        shape: (usize, usize), op: &'static str, 
    },

    #[error("expect a positive definite matrix in {op}")]
    ExpectPositiveDefiniteMatrix { op: &'static str },

    #[error("expect a symmetric matrix in {op}")]
    ExpectSymmetricMatrix { op: &'static str },
}