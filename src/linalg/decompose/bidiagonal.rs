use crate::{linalg, Context, Error, FloatDType, IndexOp, NdArray, Result};

pub fn bidiagonal<T: FloatDType>(arr: &NdArray<T>) -> Result<(NdArray<T>, NdArray<T>, NdArray<T>)> {
    // let mut mat = arr.matrix_view()?;
    let (rows, cols) = arr.dims2().context("check matrix")?;
    let arr = arr.copy();
    
    // (m, n) => m < n
    let (mat, flipped) = if cols > rows {
        (arr.transpose_last()?, true)
    } else {
        (arr.clone(), false)
    };

    let (m, n) = mat.dims2().unwrap(); // (m > n)

    let u = NdArray::<T>::eye(m)?; // (m, m)
    let v = NdArray::<T>::eye(n)?; // (n, n)

    for k in 0..n {
        let h_holder = {
            let col = mat.index((k.., k))?;
            make_householder(col.iter())? // (m-k, m-k)
        };

        {
            let lower_self_block = mat.index((k..m, k..n))?; // (m-k, n-k)
            let transformed_self = linalg::matmul(&h_holder, &lower_self_block).context("h_holder @ lower_self_block")?; // (m-k, n-k)
            lower_self_block.copy_from(&transformed_self)?;

            let lower_u_block = u.index((0..m, k..m))?; // (m, m-k)
            let transformed_u = linalg::matmul(&lower_u_block, &h_holder).context("lower_u_block @ h_holder")?;  // (m, m-k)
            lower_u_block.copy_from(&transformed_u)?;
        }

        if k < n - 2 {
            let row = mat.index((k, k+1..))?;
            let row_h_holder = make_householder(row.iter())?; // (n-k-1, n-k-1)

            {
                let lower_self_block = mat.index((k..m, k+1..n))?; // (m-k, n-k-1)
                let transformed_self = linalg::matmul(&lower_self_block, &row_h_holder).context("lower_self_block @ row_h_holder")?; // (m-k, n-k-1)
                lower_self_block.copy_from(&transformed_self)?;

                let lower_v_block = v.index((0..n, k+1..n))?; // (n, n-k-1)
                let transformed_v = linalg::matmul(&lower_v_block, &row_h_holder).context("lower_v_block @ row_h_holder")?; // (n, n-k-1)
                lower_v_block.copy_from(&transformed_v)?;
            }
        }
    }

    let new_mat = mat.index((0..n, 0..n))?; // (m, n) => (n, n)
    let u = u.index((0..m, 0..n))?; // (m, m) => (m, n)
    if flipped {
        Ok((new_mat.transpose_last()?, v, u))
    } else {
        Ok((new_mat, u, v))
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

    fn validate_bidiag(a: &NdArray<f64>, b: &NdArray<f64>, u: &NdArray<f64>, v: &NdArray<f64>, upper: bool) {
        let bv = b.matrix_view().unwrap();
        let (m, n) = bv.shape();
        for i in 0..m {
            for j in 0..n {
                let allowed = if upper {
                    j == i || j == i + 1
                } else {
                    j == i || j + 1 == i
                };
                if !allowed {
                    assert!(bv.g(i, j).abs() < 1e-10, "B should be bidiagonal, but got {} at ({}, {})", bv.g(i, j), i, j);
                }
            }
        }

        let recovered = u.matmul(b).unwrap().matmul(&v.transpose_last().unwrap()).unwrap();
        assert!(recovered.allclose(a, 1e-8, 1e-8), "A and U*B*V^T mismatch");
    }

    #[test]
    fn test_bidiagonal_square() {
        let a = NdArray::new(&[
            [1., 2., 3., 4., 5.],
            [2., 4., 1., 2., 1.],
            [3., 1., 7., 1., 1.],
            [4., 2., 1., -1., 3.],
            [5., 1., 1., 3., 2.],
        ]).unwrap();

        let (b, u, v) = linalg::bidiagonal(&a).unwrap();
        validate_bidiag(&a, &b, &u, &v, true);
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

        let (b, u, v) = linalg::bidiagonal(&a).unwrap();
        validate_bidiag(&a, &b, &u, &v, true);
    }

    #[test]
    fn test_bidiagonal_wide() {
        let a = NdArray::new(&[
            [1., 2., 3., 4., 5.],
            [2., 4., 1., 2., 1.],
            [3., 1., 7., 1., 1.],
        ]).unwrap();

        let (b, u, v) = linalg::bidiagonal(&a).unwrap();
        validate_bidiag(&a, &b, &u, &v, false);
    }
}
