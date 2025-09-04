use crate::{linalg, FloatDType, NdArray, Result};

pub fn det<T: FloatDType>(a: &NdArray<T>) -> Result<T> {
    linalg::check_square(a, "det")?;
    let plu = linalg::plu(a)?;

    let u = plu.u.matrix_view_unsafe()?;
    let mut det_u = T::one();
    for i in 0..u.shape().0 {
        det_u = det_u * u[(i, i)];
    }

    let det_p = det_permutation(&plu.p)?;

    Ok(det_p * det_u)
}

pub fn rank<T: FloatDType>(a: &NdArray<T>, tol: T) -> Result<usize> {
    linalg::check_square(a, "rank")?;
    let plu = linalg::plu(a)?;
    let u = plu.u.matrix_view_unsafe()?;

    let mut rank = 0;
    for i in 0..u.shape().0 {
        if u[(i, i)].abs() > tol {
            rank += 1;
        }
    }
    Ok(rank)
}

/// Compute determinant of permutation matrix P
fn det_permutation<T: FloatDType>(p: &NdArray<T>) -> Result<T> {
    let p = p.matrix_view_unsafe()?;
    let (n, m) = p.shape();
    assert_eq!(n, m);

    let mut perm = vec![0usize; n];
    for i in 0..n {
        for j in 0..n {
            if p[(i, j)] == T::one() {
                perm[i] = j;
                break;
            }
        }
    }

    let mut visited = vec![false; n];
    let mut sign = 1;
    for i in 0..n {
        if !visited[i] {
            let mut cycle_len = 0;
            let mut j = i;
            while !visited[j] {
                visited[j] = true;
                j = perm[j];
                cycle_len += 1;
            }
            if cycle_len % 2 == 0 {
                sign = -sign;
            }
        }
    }

    Ok(if sign == 1 { T::one() } else { -T::one() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn test_det_permutation_identity() -> Result<()> {
        let p = NdArray::<f64>::eye(3)?;
        let det_p = det_permutation(&p)?;
        assert_eq!(det_p, 1.0);
        Ok(())
    }

    #[test]
    fn test_det_permutation_swap() -> Result<()> {
        let p = NdArray::<f64>::from_vec(
            [0.0, 1.0, 0.0,
             1.0, 0.0, 0.0,
             0.0, 0.0, 1.0],
            (3, 3),
        )?;
        let det_p = det_permutation(&p)?;
        assert_eq!(det_p, -1.0);
        Ok(())
    }

    #[test]
    fn test_det_spd() -> Result<()> {
        let a = NdArray::<f64>::from_vec(
            [4.0, 2.0, 1.0,
             2.0, 5.0, 3.0,
             1.0, 3.0, 6.0],
            (3, 3),
        )?;
        let det_a = det(&a)?;
        assert!(det_a > 0.0);
        Ok(())
    }

    #[test]
    fn test_det_singular() -> Result<()> {
        let a = NdArray::<f64>::from_vec(
            [1.0, 2.0,
             2.0, 4.0],
            (2, 2),
        )?;
        let det_a = det(&a)?;
        assert!(det_a.abs() < 1e-12);

        let a = NdArray::<f64>::new(&[
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [0.5, 1.0, 22.0],
        ])?;
        let det_a = det(&a)?;
        assert!(det_a.abs() < 1e-12);

        Ok(())
    }

    #[test]
    fn test_det_normal() -> Result<()> {
        let a = NdArray::<f64>::from_vec(
            [2.0, -1.0, 0.0,
             -1.0, 2.0, -1.0,
             0.0, -1.0, 2.0],
            (3, 3),
        )?;
        let det_a = det(&a)?;
        assert!((det_a - 4.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn test_det_random() -> Result<()> {
        let a = NdArray::<f64>::randn(0., 1.0, (5, 5))?;
        let det_a = det(&a)?;
        println!("{}", det_a);
        Ok(())
    }

    #[test]
    fn test_rank() -> Result<()> {
        let a = NdArray::<f64>::from_vec(
            [1.0, 2.0,
             3.0, 4.0],
            (2, 2),
        )?;
        assert_eq!(rank(&a, 1e-12)?, 2);
    
        let b = NdArray::<f64>::from_vec(
            [1.0, 2.0,
             2.0, 4.0],
            (2, 2),
        )?;
        assert_eq!(rank(&b, 1e-12)?, 1);
    
        let c = NdArray::<f64>::zeros((3, 3))?;
        assert_eq!(rank(&c, 1e-12)?, 0);
    
        Ok(())
    }
}
