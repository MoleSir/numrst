use crate::{linalg::LinalgError, FloatDType, NdArray, Result};

/// Check a is square, and y has same len with a 
pub fn check_solve_arg<'a, T: FloatDType>(a: &'a NdArray<T>, y: &'a NdArray<T>) -> Result<()> {
    let a = a.matrix_view()?;
    let y = y.vector_view()?;
    // Check size
    let (m, n) = a.shape();
    if m != n {
        Err(LinalgError::NonSquareMatrix(a.shape()))?;
    }
    if m != y.len() {
        Err(LinalgError::SolveOutputLenMismatch(m, y.len()))?;
    }

    Ok(())
}