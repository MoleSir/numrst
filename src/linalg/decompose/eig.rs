use crate::{linalg::{self, LinalgError}, view::MatrixViewUsf, FloatDType, NdArray, Result};

pub struct EigResult<T: FloatDType> {
    pub eig_values: Vec<T>,
    pub eig_vectors:  NdArray<T>,
}

impl<T: FloatDType> EigResult<T> {
    fn new(eig_values: Vec<T>, eig_vectors:  NdArray<T>,) -> Self {
        Self { eig_values, eig_vectors }
    }

    pub fn reconstruct(&self) -> Result<NdArray<T>> {
        let d = NdArray::diag(&self.eig_values).unwrap();
        let v = self.eig_vectors.matmul(&d)?;
        let v = v.matmul(&self.eig_vectors.transpose_last()?)?;
        Ok(v)
    }
}

pub fn eig_qr<T: FloatDType>(a: &NdArray<T>, max_iters: usize, tol: T) -> Result<EigResult<T>> {
    if !linalg::is_symmetric(a)? {
        return Err(LinalgError::ExpectSymmetricMatrix { op: "eig_jacobi" })?;
    }

    let n = a.dims2()?.0;
    let mut h = a.copy();
    let mut v = NdArray::<T>::eye(n)?;

    for _ in 0..max_iters {
        let (q, r) = linalg::qr(&h)?;
        h = r.matmul(&q)?;
        v = v.matmul(&q)?;

        // off-diagonal norm
        let mut off_diag_norm = T::zero();
        let hv = h.matrix_view_unsafe()?;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off_diag_norm += unsafe { hv.g(i,j).abs() };
                }
            }
        }


        if off_diag_norm < tol {
            break;
        }
    }

    let hv = h.matrix_view_unsafe().unwrap();
    let eig_values = (0..n).map(|i| unsafe { hv.g(i,i) }).collect::<Vec<_>>();
    Ok(EigResult { eig_values, eig_vectors: v })
}

/// Eig by jacobi
/// 
/// # References
/// - https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
/// - https://www.quantstart.com/articles/Jacobi-Method-in-Python-and-NumPy/
/// - https://oldsite.pup.ac.in/e-content/science/physics/mscphy58.pdf
/// 
pub fn eig_jacobi<T: FloatDType>(mat: &NdArray<T>, tol: T) -> Result<EigResult<T>> {
    unsafe fn max_elem<T: FloatDType>(mat: &MatrixViewUsf<'_, T>) -> (T, (usize, usize)) {
        unsafe {
            let (n, _) = mat.shape();
            let mut max_value = mat.g(0, 1).abs();
            let mut max_position = (0, 1);
            for i in 0..(n - 1) {
                for j in (i+1)..n {
                    if mat.g(i, j).abs() > max_value {
                        max_value = mat.g(i, j).abs();
                        max_position = (i, j);
                    }
                }
            }
            (max_value, max_position)
        }
    }

    if !linalg::is_symmetric(mat)? {
        return Err(LinalgError::ExpectSymmetricMatrix { op: "eig_jacobi" })?;
    }

    let mat_view = mat.matrix_view_unsafe()?;
    let (n, m) = mat_view.shape();
    if m != n {
        Err(LinalgError::ExpectMatrixSquare { shape: mat_view.shape(), op: "eig_jacobi" })?;
    }

    unsafe {
        let max_rot = 5 * n.pow(2);
        let mut a = mat_view.copy();
        let mut r = NdArray::<T>::eye(n)?;
    
        for _ in 0..max_rot {
            let av = a.matrix_view_unsafe().unwrap();
            let (max_value, (p, q)) = max_elem(&av);
            assert!(p != q);
            if max_value < tol {
                let eig_vals: Vec<T> = (0..n).map(|i| av.g(i,i)).collect();
    
                // sort by eig_vals
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&i, &j| eig_vals[j].abs().partial_cmp(&eig_vals[i].abs()).unwrap());
                let eig_vals_sorted: Vec<T> = idx.iter().map(|&i| eig_vals[i]).collect();
                let eig_vecs_sorted = r.copy();
                {
                    let mut mv = eig_vecs_sorted.matrix_view_unsafe().unwrap();
                    let rv = r.matrix_view_unsafe().unwrap();
    
                    for (new_j, &old_j) in idx.iter().enumerate() {
                        for i in 0..n {
                            mv.s(i, new_j, rv.g(i, old_j));
                        }
                    }
                }
            
                return Ok(EigResult::new(eig_vals_sorted, eig_vecs_sorted));
            }
    
            let (ai, ri) = jacobi_rotate(&a, p, q)?;
            a = ai;
            r = linalg::matmul(&r, &ri)?;
        }
    
        Err(LinalgError::JacobiMethodDidNotConverge)?
    }
}

fn jacobi_rotate<T: FloatDType>(arr: &NdArray<T>, p: usize, q: usize) -> Result<(NdArray<T>, NdArray<T>)> {
    let mat = arr.matrix_view_unsafe()?;
    let (m, _) = mat.shape();

    unsafe {
        if mat.g(p, q) == T::zero() {
            return Ok((arr.clone(), NdArray::<T>::eye(m)?));
        }
    
        let (a, b, ab) = (mat.g(p, p), mat.g(q, q), mat.g(p, q));
        let (c, s) = jacobi_calculate_cs(a, b, ab);
    
        let r = NdArray::<T>::eye(m)?;
        {
            let mut r = r.matrix_view_unsafe()?;
            r.s(p, p, c); 
            r.s(q, q, c); 
            r.s(p, q, s); 
            r.s(q, p, -s); 
        }
    
        let a = linalg::matmul(&r.transpose_last()?, arr)?;
        let a = linalg::matmul(&a, &r)?;
        
        Ok((a, r))
    }
}

fn jacobi_calculate_cs<T: FloatDType>(a: T, b: T, ab: T) -> (T, T) {
    let tau = (b - a) / (T::from_f64(2.) * ab);
    let t = if tau > T::zero() {
        T::one() / (tau + (T::one() + tau.powi(2)).sqrt())
    } else {
        -T::one() / (tau.abs() + (T::one() + tau.powi(2)).sqrt())
    };
    let c = T::one() / (T::one() + t.powi(2)).sqrt();
    let s = c * t;
    (c, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn test_eig_jacobi_identity() {
        let a = NdArray::<f64>::eye(3).unwrap();
        let result = eig_jacobi(&a, 1e-12).unwrap();

        for v in result.eig_values.iter() {
            assert!((v - 1.0f64).abs() < 1e-10);
        }

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8));
    }

    #[test]
    fn test_eig_jacobi_diagonal_matrix() {
        let a = NdArray::new(&[
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, -2.0],
        ]).unwrap();

        let result = eig_jacobi(&a, 1e-12).unwrap();

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8));
    }

    #[test]
    fn test_eig_jacobi_symmetric_offdiag() {
        let a = NdArray::new(&[
            [2.0, 1.0],
            [1.0, 2.0],
        ]).unwrap();
        let result = eig_jacobi(&a, 1e-12).unwrap();

        let mut sorted = result.eig_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0f64).abs() < 1e-6);
        assert!((sorted[1] - 3.0f64).abs() < 1e-6);

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8));
    }

    #[test]
    fn test_eig_jacobi_zero_matrix() {
        let a = NdArray::zeros((3, 3)).unwrap();
        let result = eig_jacobi(&a, 1e-12).unwrap();

        for v in result.eig_values.iter() {
            assert!(f64::abs(*v) < 1e-12);
        }

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8));
    }

    #[test]
    fn test_eig_jacobi_random_symmetric() {
        let a = NdArray::<f64>::randn(0.0, 1.0, (4, 4)).unwrap();
        let a_sym = (&a + &a.transpose_last().unwrap()).unwrap().mul(0.5).unwrap();

        let result = eig_jacobi(&a_sym, 1e-10).unwrap();
        let rec = result.reconstruct().unwrap();

        assert!(rec.allclose(&a_sym, 1e-6, 1e-6));
    }

    #[test]
    #[should_panic]
    fn test_eig_jacobi_non_square() {
        let a = NdArray::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]).unwrap();
    
        let _ = eig_jacobi(&a, 1e-12).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_eig_jacobi_non_symmetric() {
        let a = NdArray::new(&[
            [1.0, 2.0],
            [0.0, 3.0],
        ]).unwrap();
    
        let result = eig_jacobi(&a, 1e-12).unwrap();
        println!("{}", result.reconstruct().unwrap());
    }

    #[test]
    fn test_eig_jacobi_zeros() {
        let a = NdArray::<f64>::zeros((4, 4)).unwrap();
        let _ = eig_jacobi(&a, 1e-12).unwrap();
    }
    
    #[test]
    fn test_eig_qr_identity() {
        let a = NdArray::<f64>::eye(3).unwrap();
        let result = eig_qr(&a, 100, 1e-12).unwrap();

        for v in result.eig_values.iter() {
            assert!((v - 1.0f64).abs() < 1e-10);
        }

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8));
    }

    #[test]
    fn test_eig_qr_diagonal_matrix() {
        let a = NdArray::new(&[
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [0.0, 0.0, -2.0],
        ]).unwrap();

        let result = eig_qr(&a, 100, 1e-12).unwrap();

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8));
    }

    #[test]
    fn test_eig_qr_symmetric_offdiag() {
        let a = NdArray::new(&[
            [2.0, 1.0],
            [1.0, 2.0],
        ]).unwrap();
        let result = eig_qr(&a, 100, 1e-12).unwrap();

        let mut sorted = result.eig_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0f64).abs() < 1e-6);
        assert!((sorted[1] - 3.0f64).abs() < 1e-6);

        let rec = result.reconstruct().unwrap();
        assert!(rec.allclose(&a, 1e-8, 1e-8));
    }

    #[test]
    fn test_eig_qr_random_symmetric() {
        let a = NdArray::<f64>::randn(0.0, 1.0, (4, 4)).unwrap();
        let a_sym = (&a + &a.transpose_last().unwrap()).unwrap().mul(0.5).unwrap();

        let result = eig_qr(&a_sym, 200, 1e-10).unwrap();
        let rec = result.reconstruct().unwrap();

        assert!(rec.allclose(&a_sym, 1e-3, 1e-3));
    }

    #[test]
    #[should_panic]
    fn test_eig_qr_non_square() {
        let a = NdArray::new(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]).unwrap();

        let _ = eig_qr(&a, 100, 1e-12).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_eig_qr_non_symmetric() {
        let a = NdArray::new(&[
            [1.0, 2.0],
            [0.0, 3.0],
        ]).unwrap();

        let result = eig_qr(&a, 100, 1e-12).unwrap();
        println!("{}", result.reconstruct().unwrap());
    }
}
