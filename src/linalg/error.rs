
#[derive(Debug, thiserror::Error)]
pub enum LinalgError {
    #[error("unexpect singular matrix")]
    SingularMatrix,

    #[error("solve a not square matrix {0:?}")]
    NonSquareMatrix((usize, usize)),

    #[error("solve output len mismatch: expect {0}, got {1}")]
    SolveOutputLenMismatch(usize, usize)
}