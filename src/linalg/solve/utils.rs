use crate::{linalg::LinalgError, FloatDType, MatrixView, Result, ToMatrixView, ToVectorView, VectorView};


/// Check a is square, and y has same len with a 
pub fn check_solve_arg<'a, T, M, V>(a: &'a M, y: &'a V) -> Result<(MatrixView<'a, T>, VectorView<'a, T>)> 
where 
    T: FloatDType,
    M: ToMatrixView<T>,
    V: ToVectorView<T>,
{
    let a = a.to_matrix_view()?;
    let y = y.to_vector_view()?;
    // Check size
    let (m, n) = a.shape();
    if m != n {
        Err(LinalgError::NonSquareMatrix(a.shape()))?;
    }
    if m != y.len() {
        Err(LinalgError::SolveOutputLenMismatch(m, y.len()))?;
    }

    Ok((a, y))
}