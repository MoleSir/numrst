use crate::{linalg, Context, Error, FloatDType, IndexOp, NdArray, Result};

/// Result of bidiagonalization of a matrix
pub struct BidiagonalResult<T: FloatDType> {
    /// Left orthogonal matrix `U` (n x n)
    pub u: NdArray<T>,
    /// Bidiagonal matrix `B` (n x m), only main diagonal and superdiagonal/non-zero
    pub b: NdArray<T>,
    /// Right orthogonal matrix `V` (m x m)
    pub v: NdArray<T>,
}

impl<T: FloatDType> BidiagonalResult<T> {
    /// Reconstruct the original matrix from its bidiagonal decomposition
    ///
    /// Computes:
    /// ```text
    /// A ≈ U * B * Vᵀ
    /// ```
    pub fn reconstruct(&self) -> Result<NdArray<T>> {
        let rec = self.u.matmul(&self.b)?.matmul(&self.v.transpose_last()?)?;
        Ok(rec)
    }
}

pub fn bidiagonal<T: FloatDType>(arr: &NdArray<T>) -> Result<BidiagonalResult<T>> {
    // let mut mat = arr.matrix_view_unsafe()?;
    let (rows, cols) = arr.dims2().context("check matrix")?;

    // !!! Very carefell to view
    let arr = if cols > rows {
        arr.transpose_last()?.copy()
    } else {
        arr.copy()
    };

    // (m, n) => m < n
    let (mat, flipped) = if cols > rows {
        (arr.matrix_view_unsafe()?, true)
    } else {
        (arr.matrix_view_unsafe()?, false)
    };

    let (m, n) = mat.shape(); // (m > n)

    let u_arr = NdArray::<T>::eye(m)?; // (m, m)
    let u = u_arr.matrix_view_unsafe().unwrap();
    let v_arr = NdArray::<T>::eye(n)?; // (n, n)
    let v = v_arr.matrix_view_unsafe().unwrap();

    unsafe {
        for k in 0..n {
            let col = mat.slice(k.., k)?;
            let h_holder = make_householder(col.iter())?; // (m-k, m-k)
            let h_holder = h_holder.matrix_view_unsafe().unwrap();
    
            {
                let mut lower_self_block = mat.slice(k..m, k..n)?; // (m-k, n-k)
                let transformed_self = h_holder.matmul(&lower_self_block).context("h_holder @ lower_self_block")?; // (m-k, n-k)
                lower_self_block.copy_from(&transformed_self.matrix_view_unsafe().unwrap())?;
    
                let mut lower_u_block = u.slice(0..m, k..m)?; // (m, m-k)
                let transformed_u = lower_u_block.matmul(&h_holder).context("lower_u_block @ h_holder")?;  // (m, m-k)
                lower_u_block.copy_from(&transformed_u.matrix_view_unsafe().unwrap())?;
            }
    
            if k < n - 2 {
                let row = mat.slice(k, k+1..)?;
                let row_h_holder = make_householder(row.iter())?; // (n-k-1, n-k-1)
                let row_h_holder = row_h_holder.matrix_view_unsafe().unwrap();
    
                {
                    let mut lower_self_block = mat.slice(k..m, k+1..n)?; // (m-k, n-k-1)
                    let transformed_self = lower_self_block.matmul(&row_h_holder).context("lower_self_block @ row_h_holder")?; // (m-k, n-k-1)
                    lower_self_block.copy_from(&transformed_self.matrix_view_unsafe().unwrap())?;
    
                    let mut lower_v_block = v.slice(0..n, k+1..n)?; // (n, n-k-1)
                    let transformed_v = lower_v_block.matmul(&row_h_holder).context("lower_v_block @ row_h_holder")?; // (n, n-k-1)
                    lower_v_block.copy_from(&transformed_v.matrix_view_unsafe().unwrap())?;
                }
            }
        }
        
        let new_arr = arr.index((0..n, 0..n))?; // (m, n) => (n, n)
        let u_arr = u_arr.index((0..m, 0..n))?; // (m, m) => (m, n)
        if flipped {
            Ok(BidiagonalResult {
                b: new_arr.transpose_last()?, 
                u: v_arr, 
                v: u_arr
            })
        } else {
            Ok(BidiagonalResult {
                b: new_arr,
                u: u_arr, 
                v: v_arr
            })
        }
    }
}

fn make_householder<T: FloatDType>(column: impl Iterator<Item = T>) -> Result<NdArray<T>> {
    let mut col_vec: Vec<T> = column.collect();
    let size = col_vec.len();

    if size == 0 {
        return Err(Error::Msg("Column for Householder transform cannot be empty.".into()));
    }

    let dot: T = col_vec.iter().map(|&x| x * x).sum();
    let denom = col_vec[0] + col_vec[0].signum() * dot.sqrt();

    if denom == T::zero() {
        return Err(Error::Msg("Cannot produce Householder transform: first entry is 0.".into()));
    }

    for x in &mut col_vec {
        *x /= denom;
    }
    col_vec[0] = T::one(); // 保证第一个元素为 1
    let v = NdArray::new(col_vec)?; // (size,)

    // v^T * v
    let v_norm_sq = linalg::dot(&v, &v)?;

    // 构造 I - 2 * v v^T / (v^T v)
    let eye = NdArray::<T>::eye(size)?;          // (size, size)
    let v_outer = linalg::outer(&v, &v)?;   // (size, size)
    v_outer.mul_assign(T::from(2.0).unwrap() / v_norm_sq)?;
    
    eye - v_outer
}

#[cfg(test)]
mod test_bidiagonal {
    use crate::{NdArray, linalg};
    use super::BidiagonalResult;

    #[test]
    fn test_bidiagonal_square() {
        let a = NdArray::new(&[
            [1., 2., 3., 4., 5.],
            [2., 4., 1., 2., 1.],
            [3., 1., 7., 1., 1.],
            [4., 2., 1., -1., 3.],
            [5., 1., 1., 3., 2.],
        ]).unwrap();

        let result = linalg::bidiagonal(&a).unwrap();
        validate_bidiag(&a, &result, true);
    }

    #[test]
    fn test_bidiagonal_tall() {
        let a = NdArray::new(&[
            [1., 2., 3.],
            [4., 5., 2.],
            [4., 1., 2.],
            [1., 3., 1.],
            [7., 1., 1.],
        ]).unwrap();

        let result = linalg::bidiagonal(&a).unwrap();
        validate_bidiag(&a, &result, true);
    }

    #[test]
    fn test_bidiagonal_wide() {
        let a = NdArray::new(&[
            [1., 2., 3., 4., 5.],
            [2., 4., 1., 2., 1.],
            [3., 1., 7., 1., 1.],
        ]).unwrap();

        let result = linalg::bidiagonal(&a).unwrap();
        validate_bidiag(&a, &result, false);
    }

    fn validate_bidiag(a: &NdArray<f64>, result: &BidiagonalResult<f64>, upper: bool) {
        let b = &result.b;

        let bv = b.matrix_view_unsafe().unwrap();
        let (m, n) = bv.shape();
        for i in 0..m {
            for j in 0..n {
                let allowed = if upper {
                    j == i || j == i + 1
                } else {
                    j == i || j + 1 == i
                };

                unsafe {
                    if !allowed {
                        assert!(bv.g(i, j).abs() < 1e-10, "B should be bidiagonal, but got {} at ({}, {})", bv.g(i, j), i, j);
                    }
                }
            }
        }

        let recovered = result.reconstruct().unwrap();
        assert!(recovered.allclose(a, 1e-8, 1e-8), "A and U*B*V^T mismatch");
    }

}
