use crate::{linalg, FloatDType, NdArray, Result};
use super::utils;

pub fn qr_solve<T: FloatDType>(a: &NdArray<T>, y: &NdArray<T>) -> Result<NdArray<T>> {
    utils::check_solve_arg(&a, &y)?;
    let (q, r) = linalg::qr(a)?;
    let q = q.matrix_view()?;                // Q: (m, m)
    let r = r.matrix_view()?;                // R: (m, n)
    let y = y.vector_view().unwrap();

    let n = r.shape().1;                        // number of unknowns
    let rhs_arr = NdArray::<T>::zeros(n)?; // rhs = Q^T * y
    let mut rhs = rhs_arr.vector_view_mut().unwrap();

    // Step 2: compute Q^T * y
    for i in 0..n {
        let mut sum = T::zero();
        for j in 0..y.len() {
            sum += q.g(j, i) * y.g(j);
        }
        rhs.s(i, sum);
    }

    // Step 3: backward substitution R x = rhs
    let x_arr = NdArray::<T>::zeros(n)?;
    {
        let mut x = x_arr.vector_view_mut().unwrap();
        for i in (0..n).rev() {
            let mut sum = T::zero();
            for j in i+1..n {
                sum += r.g(i, j) * x.g(j);
            }
            x.s(i, (rhs.g(i) - sum) / r.g(i, i));
        }    
    }

    Ok(x_arr)
}   

#[cfg(test)]
mod test {
    use crate::{NdArray, linalg};

    #[test]
    fn test_qr_solve_simple() {
        // Ax = y, A = [[3, 1], [1, 2]], y = [9, 8], solution x = [2, 3]
        let a = NdArray::new(&[
            [3., 1.],
            [1., 2.],
        ]).unwrap();

        let y = NdArray::from_vec([9., 8.].to_vec(), 2).unwrap();

        let x = linalg::qr_solve(&a, &y).unwrap();
        let expected = NdArray::from_vec([2., 3.].to_vec(), 2).unwrap();

        assert!(x.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_qr_solve_identity() {
        // Identity matrix, solution should equal y
        let a = NdArray::<f64>::eye(4).unwrap();
        let y = NdArray::from_vec([1., 2., 3., 4.].to_vec(), 4).unwrap();

        let x = linalg::qr_solve(&a, &y).unwrap();
        assert!(x.allclose(&y, 1e-12, 1e-12));
    }
}
