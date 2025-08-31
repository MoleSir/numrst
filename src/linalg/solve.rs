use crate::{FloatDType, Result, ToMatrixView, ToVectorView, Vector};
use super::{chol_solve, lu_solve};

pub enum SolveMethod {
    LU,
    Cholesky,
}

pub fn solve<T, M, V>(a: M, y: V, method: SolveMethod) -> Result<Vector<T>>
where
    T: FloatDType,
    M: ToMatrixView<T>,
    V: ToVectorView<T>,
{
    match method {
        SolveMethod::LU => lu_solve(a, y),
        SolveMethod::Cholesky => chol_solve(a, y),
    }
}
