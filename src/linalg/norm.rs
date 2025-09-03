use std::borrow::Cow;
use crate::{DTypeConvert, FloatCategory, FloatDType, IntCategory, IntDType, NdArray, NumCategory, NumDType};

/// Enum representing different norm types for vectors and matrices
#[derive(Debug, Clone, Copy)]
pub enum Norm {
    /// L1 norm (sum of absolute values)
    /// \|x\|_1 = Σ_i |x_i|
    L1,

    /// L2 norm (Euclidean norm)
    /// \|x\|_2 = sqrt(Σ_i |x_i|^2)
    L2,

    /// Frobenius norm (for matrices)
    /// \|A\|_F = sqrt(Σ_{i,j} |A_ij|^2)
    Frobenius,

    /// Infinity norm (max absolute value)
    /// \|x\|_∞ = max_i |x_i|
    Infinity,

    /// Spectral norm (largest singular value of a matrix)
    /// \|A\|_2 = σ_max(A)
    Spectral,

    /// Nuclear norm (sum of singular values of a matrix)
    /// \|A\|_* = Σ_i σ_i(A)
    Nuclear,

    /// General Lp norm for vectors
    /// \|x\|_p = (Σ_i |x_i|^p)^(1/p)
    Lp(f64),
}

impl Norm {
    /// Optional: return a human-readable formula as string
    pub fn formula(&self) -> Cow<'_, str> {
        match self {
            Norm::L1 => r"||x||_1 = \sum_i |x_i|".into(),
            Norm::L2 => r"||x||_2 = \sqrt{\sum_i |x_i|^2}".into(),
            Norm::Frobenius => r"||A||_F = \sqrt{\sum_{i,j} |A_{ij}|^2}".into(),
            Norm::Infinity => r"||x||_\infty = \max_i |x_i|".into(),
            Norm::Spectral => r"||A||_2 = \sigma_\max(A)".into(),
            Norm::Nuclear => r"||A||_* = \sum_i \sigma_i(A)".into(),
            Norm::Lp(p) => format!("||x||_{} = (\\sum_i |x_i|^{})^{}", p, p, 1./p).into(),
        }
    }
}

/// Compute the specified norm of an NdArray
pub fn norm<T: NormOp>(arr: &NdArray<T>, norm: Norm) -> T::NormOutput {
    match norm {
        Norm::L1 => T::l1_norm(arr),
        Norm::L2 => T::l2_norm(arr),
        Norm::Frobenius => T::frobenius_norm(arr),
        Norm::Infinity => T::inf_norm(arr),
        Norm::Spectral => T::spectral_norm(arr),
        Norm::Nuclear => T::nuclear_norm(arr),
        Norm::Lp(p) => T::lp_norm(arr, p),
    }
}

/// Trait for computing various norms on vectors or matrices.
///
/// # Overview
/// This trait provides a unified interface to compute common norms,
/// including Lp norms, infinity norm, spectral norm, nuclear norm,
/// L1, L2, and Frobenius norms. It is generic over the element type `Self`
/// and produces results of type `Output` (usually a floating-point type).
pub trait NormOp: NumDType {
    /// Output type of the norm computation (usually floating point)
    type NormOutput: FloatDType;

    fn lp_norm(arr: &NdArray<Self> , p: f64) -> Self::NormOutput;

    fn inf_norm(arr: &NdArray<Self>) -> Self::NormOutput;

    fn spectral_norm(arr: &NdArray<Self>) -> Self::NormOutput;

    fn nuclear_norm(arr: &NdArray<Self>) -> Self::NormOutput;

    fn l1_norm(arr: &NdArray<Self>) -> Self::NormOutput {
        Self::lp_norm(arr, 1.0)
    }

    fn l2_norm(arr: &NdArray<Self>) -> Self::NormOutput {
        Self::lp_norm(arr, 2.0)
    }

    fn frobenius_norm(arr: &NdArray<Self>) -> Self::NormOutput {
        Self::lp_norm(arr, 2.0)
    }
}

pub trait NormOpCategory<T: NumDType, C: NumCategory = <T as NumDType>::Category> {
    type NormOutput: FloatDType;
    fn lp_norm(arr: &NdArray<T>, p: f64) -> Self::NormOutput;
    fn inf_norm(arr: &NdArray<T>) -> Self::NormOutput;
    fn spectral_norm(arr: &NdArray<T>) -> Self::NormOutput;
    fn nuclear_norm(arr: &NdArray<T>) -> Self::NormOutput;

}

impl<T: IntDType + DTypeConvert<f64>> NormOpCategory<T, IntCategory> for T {
    type NormOutput = f64;

    fn lp_norm(arr: &NdArray<Self>, p: f64) -> f64 {
        let sum = arr.iter().map(|v| T::to_f64(v).abs().powf(p)).sum::<f64>();
        sum.powf(1. / p)
    }

    fn inf_norm(arr: &NdArray<Self>) -> f64 {
        T::to_f64(arr.iter().max().unwrap_or(T::zero()))
    }

    fn spectral_norm(arr: &NdArray<Self>) -> f64 {
        <f64 as NormOp>::spectral_norm(&arr.to_dtype::<f64>())
    }

    fn nuclear_norm(arr: &NdArray<Self>) -> f64 {
        <f64 as NormOp>::nuclear_norm(&arr.to_dtype::<f64>())
    }
}

impl<T: FloatDType> NormOpCategory<T, FloatCategory> for T {
    type NormOutput = T;

    fn lp_norm(arr: &NdArray<Self>, p: f64) -> T {
        let p = T::from_f64(p);
        let sum = arr.iter().map(|v| v.abs().powf(p)).sum::<T>();
        sum.powf(T::one() / p)
    }

    fn inf_norm(arr: &NdArray<Self>) -> T {
        arr.iter()
            .map(|v| v.abs())
            .fold(T::zero(), |acc, x| acc.max(x))
    }

    fn spectral_norm(arr: &NdArray<Self>) -> T {
        if arr.element_count() == 0 {
            return T::zero();
        }
            
        if arr.rank() == 1 {
            // treat as vector
            return <Self as NormOp>::lp_norm(arr, 2.0);
        }

        // matrix: compute largest singular value
        let svd = crate::linalg::svd(arr).unwrap();
        svd.sigmas.iter().cloned().fold(T::zero(), |a, b| a.max(b))
    }

    fn nuclear_norm(arr: &NdArray<Self>) -> T {
        if arr.element_count() == 0 {
            return T::zero();
        }
        if arr.rank() == 1 {
            return arr.iter().map(|v| v.abs()).sum::<T>();
        }
        let svd = crate::linalg::svd(arr).unwrap();
        svd.sigmas.iter().cloned().sum()
    }
}


impl<T: NumDType> NormOp for T 
where
    Self: NormOpCategory<T>
{
    type NormOutput = <T as NormOpCategory<T>>::NormOutput;

    fn lp_norm(arr: &NdArray<Self>, p: f64) -> Self::NormOutput {
        <Self as NormOpCategory<Self>>::lp_norm(arr, p)
    }

    fn inf_norm(arr: &NdArray<Self>) -> Self::NormOutput {
        <Self as NormOpCategory<Self>>::inf_norm(arr)
    }

    fn spectral_norm(arr: &NdArray<Self>) -> Self::NormOutput {
        <Self as NormOpCategory<Self>>::spectral_norm(arr)

    }

    fn nuclear_norm(arr: &NdArray<Self>) -> Self::NormOutput {
        <Self as NormOpCategory<Self>>::nuclear_norm(arr)
    }
} 

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NdArray, Result};

    #[test]
    fn test_norm_float() -> Result<()> {
        let a = NdArray::new(&[1.0f32, -2.0, 3.0])?;
        let m = NdArray::new(&[
            [1.0f32, 2.0],
            [-3.0, 4.0]
        ])?;

        // L1 norm
        let l1_v = norm(&a, Norm::L1);
        assert!((l1_v - 6.0).abs() < 1e-6);

        let l1_m = norm(&m, Norm::L1);
        assert!((l1_m - 10.0).abs() < 1e-6);

        // L2 norm
        let l2_v = norm(&a, Norm::L2);
        assert!((l2_v - (1.0f32 + 4.0 + 9.0).sqrt()).abs() < 1e-6);

        let l2_m = norm(&m, Norm::L2);
        assert!((l2_m - f32::sqrt(1.0 + 4.0 + 9.0 + 16.0)).abs() < 1e-6);

        // Infinity norm
        let inf_v = norm(&a, Norm::Infinity);
        assert!((inf_v - 3.0).abs() < 1e-6);

        let inf_m = norm(&m, Norm::Infinity);
        assert!((inf_m - 4.0).abs() < 1e-6);

        // Lp norm with p = 3
        let lp_v = norm(&a, Norm::Lp(3.0));
        assert!((lp_v - (1f32.powi(3) + 8.0 + 27.0).powf(1.0/3.0)).abs() < 1e-6);

        // Frobenius norm (matrix)
        let frob = norm(&m, Norm::Frobenius);
        assert!((frob - f32::sqrt(1.0 + 4.0 + 9.0 + 16.0)).abs() < 1e-6);

        let _ = norm(&m, Norm::Spectral);
        let _ = norm(&m, Norm::Nuclear);

        Ok(())
    }


    #[test]
    fn test_int_norms() -> Result<()> {
        let a = NdArray::new(&[1i32, -2, 3, -4, 5])?;

        // L1 norm: sum of absolute values
        let l1 = norm(&a, Norm::L1);
        assert_eq!(l1, 15.0); // 1+2+3+4+5=15

        // L2 norm: sqrt(sum of squares)
        let l2 = norm(&a, Norm::L2);
        assert!((l2 - f64::sqrt(1.0+4.0+9.0+16.0+25.0)).abs() < 1e-10);

        // Infinity norm: max absolute value
        let inf = norm(&a, Norm::Infinity);
        assert_eq!(inf, 5.0);

        // Lp norm with p=3
        let lp3 = norm(&a, Norm::Lp(3.0));
        let expected_lp3 = (1f64.powi(3) + 2f64.powi(3) + 3f64.powi(3) + 4f64.powi(3) + 5f64.powi(3)).powf(1.0/3.0);
        assert!((lp3 - expected_lp3).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_int_empty_array() -> Result<()> {
        let a: NdArray<i32> = NdArray::zeros(0)?;
        let l1 = norm(&a, Norm::L1);
        assert_eq!(l1, 0.0);

        let l2 = norm(&a, Norm::L2);
        assert_eq!(l2, 0.0);

        Ok(())
    }
}
