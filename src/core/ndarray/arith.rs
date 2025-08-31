use crate::{Error, FloatDType, NumDType, Result, Shape, Storage, WithDType};
use super::NdArray;

//////////////////////////////////////////////////////////////////////////////
///        Binary(Assign) Op with NdArray and NdArray / scalar
//////////////////////////////////////////////////////////////////////////////

impl<T: WithDType> NdArray<T> {
    fn same_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<&Shape> {
        let lhs = self.shape();
        let rhs = rhs.shape();
        if lhs != rhs {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op,
            })
        } else {
            Ok(lhs)
        }
    }
}

pub trait NdArrayBinaryOpRhs<T: WithDType> {
    fn op<U, F>(lhs: &NdArray<T>, rhs: Self, f: F, op_name: &'static str) -> Result<NdArray<U>>
    where 
        U: WithDType, 
        F: FnMut(T, T) -> U;

    fn assign_op<F>(lhs: &NdArray<T>, rhs: Self, f: F, op_name: &'static str) -> Result<()>
    where 
        F: FnMut(T, T) -> T;
}

impl<T: WithDType> NdArrayBinaryOpRhs<T> for T {
    fn op<U, F>(lhs: &NdArray<T>, rhs: T, mut f: F, _op_name: &'static str) -> Result<NdArray<U>>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        let storage = lhs.storage();
        let vec = storage.data();
        let output: Vec<_> = lhs.layout().storage_indices()
            .map(|i| f(vec[i], rhs))
            .collect();
    
        let storage = Storage::<U>::new(output);
        Ok(NdArray::<U>::from_storage(storage, lhs.shape()))
    }

    fn assign_op<F>(lhs: &NdArray<T>, rhs: T, mut f: F, _op_name: &'static str) -> Result<()>
        where 
            F: FnMut(T, T) -> T
    {
        let mut storage = lhs.0.storage.write().unwrap();
        let data = storage.data_mut();

        lhs.layout().storage_indices()
            .for_each(|i| data[i] = f(data[i], rhs));

        Ok(())
    }
} 

impl<T: WithDType> NdArrayBinaryOpRhs<T> for NdArray<T> {
    fn op<U, F>(lhs: &NdArray<T>, rhs: NdArray<T>, f: F, op_name: &'static str) -> Result<NdArray<U>>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        <&NdArray<T> as NdArrayBinaryOpRhs<T>>::op(lhs, &rhs, f, op_name)
    }

    fn assign_op<F>(lhs: &NdArray<T>, rhs: Self, f: F, op_name: &'static str) -> Result<()>
        where 
            F: FnMut(T, T) -> T 
    {
        <&NdArray<T> as NdArrayBinaryOpRhs<T>>::assign_op(lhs, &rhs, f, op_name)
    }
}

impl<T: WithDType> NdArrayBinaryOpRhs<T> for &NdArray<T> {
    fn op<U, F>(lhs: &NdArray<T>, rhs: &NdArray<T>, mut f: F, op_name: &'static str) -> Result<NdArray<U>>
        where 
            U: WithDType, 
            F: FnMut(T, T) -> U 
    {
        let shape = NdArray::<T>::same_shape_binary_op(lhs, rhs, op_name)?;
        let lhs_storage = lhs.storage();
        let rhs_storage = rhs.storage();
        let lhs_layout = lhs.layout();
        let rhs_layout = rhs.layout();

        assert_eq!(lhs_layout.dims(), rhs_layout.dims(), "lhs dims != rhs dim2");

        let lhs = lhs_storage.data();
        let rhs = rhs_storage.data();
        
        let output: Vec<_> = lhs_layout.storage_indices().zip(rhs_layout.storage_indices())
            .map(|(lhs_index, rhs_index)| f(lhs[lhs_index], rhs[rhs_index]))
            .collect();
        
        let storage = Storage::<U>::new(output);
        Ok(NdArray::<U>::from_storage(storage, shape))
    }

    fn assign_op<F>(lhs: &NdArray<T>, rhs: &NdArray<T>, mut f: F, op_name: &'static str) -> Result<()>
        where 
            F: FnMut(T, T) -> T
    {
        NdArray::<T>::same_shape_binary_op(lhs, rhs, op_name)?;

        let mut lhs_storage = lhs.0.storage.write().unwrap();
        let rhs_storage = rhs.storage();
        let lhs_layout = lhs.layout();
        let rhs_layout = rhs.layout();

        assert_eq!(lhs_layout.dims(), rhs_layout.dims(), "lhs dims != rhs dim2");
        let lhs = lhs_storage.data_mut();
        let rhs = rhs_storage.data();
        
        for (lhs_index, rhs_index) in lhs_layout.storage_indices().zip(rhs_layout.storage_indices()) {
            lhs[lhs_index] = f(lhs[lhs_index], rhs[rhs_index]) ;
        }

        Ok(())    
    }
}

macro_rules! binary_op_impl {
    ($fn_name:ident) => {
        pub fn $fn_name(&self, rhs: impl NdArrayBinaryOpRhs<T>) -> Result<Self> {
            NdArrayBinaryOpRhs::<T>::op(self, rhs, T::$fn_name, stringify!(fn_name))
        }
    };
}

impl<T: NumDType> NdArray<T> {
    binary_op_impl!(add);
    binary_op_impl!(mul);
    binary_op_impl!(sub);
    binary_op_impl!(div);
    binary_op_impl!(minimum);
    binary_op_impl!(maximum);
}

macro_rules! assign_op_impl {
    ($fn_name:ident, $op:ident) => {
        pub fn $fn_name(&self, rhs: impl NdArrayBinaryOpRhs<T>) -> Result<()> {
            NdArrayBinaryOpRhs::<T>::assign_op(self, rhs, T::$op, stringify!(fn_name))
        }
    };
}

impl<T: NumDType> NdArray<T> {
    assign_op_impl!(add_assign, add);
    assign_op_impl!(mul_assign, mul);
    assign_op_impl!(sub_assign, sub);
    assign_op_impl!(div_assign, div);
    assign_op_impl!(minimum_assign, minimum);
    assign_op_impl!(maximum_assign, maximum);
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

impl<T: NumDType> NdArray<T> {
    pub fn eq(&self, rhs: impl NdArrayBinaryOpRhs<T>) -> Result<NdArray<bool>> {
        self.cmp(rhs, CmpOp::Eq)
    }

    pub fn ne(&self, rhs: impl NdArrayBinaryOpRhs<T>) -> Result<NdArray<bool>> {
        self.cmp(rhs, CmpOp::Ne)
    }

    pub fn le(&self, rhs: impl NdArrayBinaryOpRhs<T>) -> Result<NdArray<bool>> {
        self.cmp(rhs, CmpOp::Le)
    }

    pub fn ge(&self, rhs: impl NdArrayBinaryOpRhs<T>) -> Result<NdArray<bool>> {
        self.cmp(rhs, CmpOp::Ge)
    }

    pub fn lt(&self, rhs: impl NdArrayBinaryOpRhs<T>) -> Result<NdArray<bool>> {
        self.cmp(rhs, CmpOp::Lt)
    }

    pub fn gt(&self, rhs: impl NdArrayBinaryOpRhs<T>) -> Result<NdArray<bool>> {
        self.cmp(rhs, CmpOp::Gt)
    }

    pub fn cmp(&self, rhs: impl NdArrayBinaryOpRhs<T>, op: CmpOp) -> Result<NdArray<bool>> {
        match op {
            CmpOp::Eq => NdArrayBinaryOpRhs::<T>::op(self, rhs, |a, b| a == b, "eq"),
            CmpOp::Ne => NdArrayBinaryOpRhs::<T>::op(self, rhs, |a, b| a != b, "nq"),
            CmpOp::Le => NdArrayBinaryOpRhs::<T>::op(self, rhs, |a, b| a <= b, "le"),
            CmpOp::Ge => NdArrayBinaryOpRhs::<T>::op(self, rhs, |a, b| a >= b, "ge"),
            CmpOp::Lt => NdArrayBinaryOpRhs::<T>::op(self, rhs, |a, b| a <  b, "lt"),
            CmpOp::Gt => NdArrayBinaryOpRhs::<T>::op(self, rhs, |a, b| a >  b, "gt"),
        }
    } 
}

impl NdArray<bool> {
    pub fn and(&self, rhs: impl NdArrayBinaryOpRhs<bool>) -> Result<NdArray<bool>> {
        NdArrayBinaryOpRhs::<bool>::op(self, rhs, |a, b| a & b, "and")
    }

    pub fn or(&self, rhs: impl NdArrayBinaryOpRhs<bool>) -> Result<NdArray<bool>> {
        NdArrayBinaryOpRhs::<bool>::op(self, rhs, |a, b| a | b, "or")
    }

    pub fn xor(&self, rhs: impl NdArrayBinaryOpRhs<bool>) -> Result<NdArray<bool>> {
        NdArrayBinaryOpRhs::<bool>::op(self, rhs, |a, b| a ^ b, "xor")
    }

    pub fn not(&self) -> NdArray<bool> {
        if self.element_count() == 0 {
            return self.clone();
        }
        let storage = self.unary_op(|v| !v);
        Self::from_storage(storage, self.shape())
    }
}

//////////////////////////////////////////////////////////////////////////////
///        Unary Op / Unary Assign Op  for NdArray
//////////////////////////////////////////////////////////////////////////////

impl<T: WithDType> NdArray<T> {
    fn unary_op<U, F>(&self, mut f: F) -> Storage<U> 
    where
        U: WithDType,
        F: FnMut(T) -> U
    {
        let storage = self.storage();
        let vec = storage.data();
        let mut output = vec![];
        for index in self.layout().storage_indices() {
            output.push( f(vec[index]) );
        }
        
        Storage::new(output)
    }

    fn unary_assign_op<F>(&self, mut f: F) 
    where
        F: FnMut(T) -> T
    {
        let mut storage = self.0.storage.write().unwrap();
        let vec = storage.data_mut();
        for index in self.layout().storage_indices() {
            vec[index] = f(vec[index]);
        }
    }
}

impl<T: NumDType> NdArray<T> {
    pub fn affine(&self, mul: T, add: T) -> Result<Self> {
        if self.element_count() == 0 {
            return Ok(self.clone());
        }
        let storage = self.unary_op(|v| v * mul + add);
        Ok(Self::from_storage(storage, self.shape()))
    }

    pub fn affine_assign(&self, mul: T, add: T) -> Result<()> {
        if self.element_count() == 0 {
            return Ok(());
        }
        self.unary_assign_op(|v| v * mul + add);
        Ok(())
    }
}

macro_rules! float_unary_op_impl {
    ($fn_name:ident) => {
        pub fn $fn_name(&self) -> Self {
            if self.element_count() == 0 {
                return self.clone();
            }
            let storage = self.unary_op(F::$fn_name);
            Self::from_storage(storage, self.shape())
        }
    };
}

macro_rules! float_unary_assign_op_impl {
    ($fn_name:ident, $op:ident) => {
        pub fn $fn_name(&self) {
            if self.element_count() == 0 {
                return;
            }
            self.unary_assign_op(F::$op);
        }
    };
}

impl<F: FloatDType> NdArray<F> {
    float_unary_op_impl!(exp);
    float_unary_op_impl!(sin);
    float_unary_op_impl!(cos);
    float_unary_op_impl!(tanh);
    float_unary_op_impl!(sqrt);
    float_unary_op_impl!(floor);
    float_unary_op_impl!(ceil);
    float_unary_op_impl!(round);
    float_unary_op_impl!(abs);
    float_unary_op_impl!(neg);
    float_unary_op_impl!(ln);
    float_unary_op_impl!(recip);

    float_unary_assign_op_impl!(exp_assign, exp);
    float_unary_assign_op_impl!(sin_assign, sin);
    float_unary_assign_op_impl!(cos_assign, cos);
    float_unary_assign_op_impl!(sqrt_assign, sqrt);
    float_unary_assign_op_impl!(tanh_assign, tanh);
    float_unary_assign_op_impl!(floor_assign, floor);
    float_unary_assign_op_impl!(ceil_assign, ceil);
    float_unary_assign_op_impl!(round_assign, round);
    float_unary_assign_op_impl!(abs_assign, abs);
    float_unary_assign_op_impl!(neg_assign, neg);
    float_unary_assign_op_impl!(ln_assign, ln);
    float_unary_assign_op_impl!(recip_assign, recip);
    
}

use std::ops::{Add, Sub, Mul, Div};

impl<T: NumDType, R: NdArrayBinaryOpRhs<T>> Add<R> for &NdArray<T> {
    type Output = Result<NdArray<T>>;
    fn add(self, rhs: R) -> Self::Output {
        NdArray::add(self, rhs)
    }
}

impl<T: NumDType, R: NdArrayBinaryOpRhs<T>> Add<R> for NdArray<T> {
    type Output = Result<NdArray<T>>;
    fn add(self, rhs: R) -> Self::Output {
        NdArray::add(&self, rhs)
    }
}

impl<T: NumDType, R: NdArrayBinaryOpRhs<T>> Sub<R> for &NdArray<T> {
    type Output = Result<NdArray<T>>;
    fn sub(self, rhs: R) -> Self::Output {
        NdArray::sub(self, rhs)
    }
}

impl<T: NumDType, R: NdArrayBinaryOpRhs<T>> Sub<R> for NdArray<T> {
    type Output = Result<NdArray<T>>;
    fn sub(self, rhs: R) -> Self::Output {
        NdArray::sub(&self, rhs)
    }
}

impl<T: NumDType, R: NdArrayBinaryOpRhs<T>> Mul<R> for &NdArray<T> {
    type Output = Result<NdArray<T>>;
    fn mul(self, rhs: R) -> Self::Output {
        NdArray::mul(self, rhs)
    }
}

impl<T: NumDType, R: NdArrayBinaryOpRhs<T>> Mul<R> for NdArray<T> {
    type Output = Result<NdArray<T>>;
    fn mul(self, rhs: R) -> Self::Output {
        NdArray::mul(&self, rhs)
    }
}

impl<T: NumDType, R: NdArrayBinaryOpRhs<T>> Div<R> for &NdArray<T> {
    type Output = Result<NdArray<T>>;
    fn div(self, rhs: R) -> Self::Output {
        NdArray::div(self, rhs)
    }
}

impl<T: NumDType, R: NdArrayBinaryOpRhs<T>> Div<R> for NdArray<T> {
    type Output = Result<NdArray<T>>;
    fn div(self, rhs: R) -> Self::Output {
        NdArray::div(&self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_log() {
        let a = NdArray::new(&[0.0f32, 1.0, 2.0]).unwrap();
        let exp_a = a.exp();
        let log_a = exp_a.ln();
        assert!(a.allclose(&log_a, 1e-5, 1e-8));
    }

    #[test]
    fn test_trig() {
        let a = NdArray::new(&[0.0f32, std::f32::consts::FRAC_PI_2]).unwrap();
        let sin_a = a.sin();
        let cos_a = a.cos();

        let expected_sin = NdArray::new(&[0.0f32, 1.0]).unwrap();
        let expected_cos = NdArray::new(&[1.0f32, 0.0]).unwrap();

        println!("{:?}", cos_a.iter().collect::<Vec<_>>());

        assert!(sin_a.allclose(&expected_sin, 1e-5, 1e-8));
        assert!(cos_a.allclose(&expected_cos, 1e-5, 8e-8));
    }

    #[test]
    fn test_abs_neg() {
        let a = NdArray::new(&[-1.0f32, 0.0, 2.0]).unwrap();
        let abs_a = a.abs();
        let neg_a = a.neg();

        let expected_abs = NdArray::new(&[1.0f32, 0.0, 2.0]).unwrap();
        let expected_neg = NdArray::new(&[1.0f32, 0.0, -2.0]).unwrap();

        assert!(abs_a.allclose(&expected_abs, 1e-6, 1e-6));
        assert!(neg_a.allclose(&expected_neg, 1e-6, 1e-6));
    }

    #[test]
    fn test_floor_ceil_round() {
        let a = NdArray::new(&[1.2f32, 2.7, -1.3]).unwrap();
        let floor_a = a.floor();
        let ceil_a = a.ceil();
        let round_a = a.round();

        let expected_floor = NdArray::new(&[1.0f32, 2.0, -2.0]).unwrap();
        let expected_ceil = NdArray::new(&[2.0f32, 3.0, -1.0]).unwrap();
        let expected_round = NdArray::new(&[1.0f32, 3.0, -1.0]).unwrap();

        assert!(floor_a.allclose(&expected_floor, 1e-6, 1e-6));
        assert!(ceil_a.allclose(&expected_ceil, 1e-6, 1e-6));
        assert!(round_a.allclose(&expected_round, 1e-6, 1e-6));
    }

    #[test]
    fn test_floor_recip() {
        let a = NdArray::new(&[1.2f32, 2.7, -1.3]).unwrap();
        let recip_a = a.recip();
        let expected = NdArray::new(&[1.2f32.recip(), 2.7f32.recip(), -1.3f32.recip(),]).unwrap();

        assert!(recip_a.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_add_basic() {
        let a = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = NdArray::new(&[4.0f32, 5.0, 6.0]).unwrap();
        let c = NdArray::add(&a, &b).unwrap();
        let expected = NdArray::new(&[5.0f32, 7.0, 9.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_sub_basic() {
        let a = NdArray::new(&[10.0f32, 20.0, 30.0]).unwrap();
        let b = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let c = NdArray::sub(&a, &b).unwrap();
        let expected = NdArray::new(&[9.0f32, 18.0, 27.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_mul_basic() {
        let a = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = NdArray::new(&[2.0f32, 3.0, 4.0]).unwrap();
        let c = NdArray::mul(&a, &b).unwrap();
        let expected = NdArray::new(&[2.0f32, 6.0, 12.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_div_basic() {
        let a = NdArray::new(&[4.0f32, 9.0, 16.0]).unwrap();
        let b = NdArray::new(&[2.0f32, 3.0, 4.0]).unwrap();
        let c = NdArray::div(&a, &b).unwrap();
        let expected = NdArray::new(&[2.0f32, 3.0, 4.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_min_max_basic() {
        let a = NdArray::new(&[1.0f32, 5.0, 3.0]).unwrap();
        let b = NdArray::new(&[2.0f32, 4.0, 6.0]).unwrap();
        let min_res = NdArray::minimum(&a, &b).unwrap();
        let max_res = NdArray::maximum(&a, &b).unwrap();
        let expected_min = NdArray::new(&[1.0f32, 4.0, 3.0]).unwrap();
        let expected_max = NdArray::new(&[2.0f32, 5.0, 6.0]).unwrap();
        assert!(min_res.allclose(&expected_min, 1e-6, 1e-6));
        assert!(max_res.allclose(&expected_max, 1e-6, 1e-6));
    }

    #[test]
    fn test_add_sub_mul_div_assign() {
        let a = NdArray::new(&[1., 2., 3.]).unwrap();
        let b = NdArray::new(&[4., 5., 6.]).unwrap();

        a.add_assign(&b).unwrap();
        assert!(a.allclose(&NdArray::new(&[5., 7., 9.]).unwrap(), 1e-6, 1e-6));

        a.sub_assign(&b).unwrap();
        assert!(a.allclose(&NdArray::new(&[1., 2., 3.]).unwrap(), 1e-6, 1e-6));

        a.mul_assign(&b).unwrap();
        assert!(a.allclose(&NdArray::new(&[4., 10., 18.]).unwrap(), 1e-6, 1e-6));

        a.div_assign(&b).unwrap();
        assert!(a.allclose(&NdArray::new(&[1., 2., 3.]).unwrap(), 1e-6, 1e-6));
    }

    #[test]
    fn test_min_max_assign() {
        let a = NdArray::new(&[1., 5., 3.]).unwrap();
        let b = NdArray::new(&[4., 2., 6.]).unwrap();

        a.minimum_assign(&b).unwrap();
        assert!(a.allclose(&NdArray::new(&[1., 2., 3.]).unwrap(), 1e-6, 1e-6));

        let a2 = NdArray::new(&[1., 5., 3.]).unwrap();
        a2.maximum_assign(&b).unwrap();
        assert!(a2.allclose(&NdArray::new(&[4., 5., 6.]).unwrap(), 1e-6, 1e-6));
    }

    #[test]
    fn test_comparisons() {
        let a = NdArray::new(&[1, 2, 3]).unwrap();
        let b = NdArray::new(&[1, 0, 3]).unwrap();

        assert_eq!(a.eq(&b).unwrap().to_vec(), [true, false, true]);
        assert_eq!(a.ne(&b).unwrap().to_vec(), [false, true, false]);
        assert_eq!(a.lt(&b).unwrap().to_vec(), [false, false, false]);
        assert_eq!(a.le(&b).unwrap().to_vec(), [true, false, true]);
        assert_eq!(a.gt(&b).unwrap().to_vec(), [false, true, false]);
        assert_eq!(a.ge(&b).unwrap().to_vec(), [true, true, true]);
    }

    #[test]
    fn test_add_mul_2d_3d() {
        let a = NdArray::new(&[[1.0f32, 2.0], [3.0, 4.0]]).unwrap();
        let b = NdArray::new(&[[5.0f32, 6.0], [7.0, 8.0]]).unwrap();
        let c = NdArray::add(&a, &b).unwrap();
        let expected = NdArray::new(&[[6., 8.], [10., 12.]]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));

        let a3 = NdArray::new(&[
            [[1., 2.], [3., 4.]],
            [[5., 6.], [7., 8.]],
        ]).unwrap();
        let b3 = NdArray::new(&[
            [[2., 0.5], [1., 2.]],
            [[0.5, 2.], [1.5, 1.]],
        ]).unwrap();
        let c3 = NdArray::mul(&a3, &b3).unwrap();
        let expected3 = NdArray::new(&[
            [[2., 1.], [3., 8.]],
            [[2.5, 12.], [10.5, 8.]],
        ]).unwrap();
        assert!(c3.allclose(&expected3, 1e-6, 1e-6));
    }

    #[test]
    fn test_div_high_dim() {
        let a = NdArray::full((2, 2, 2, 2), 8.0f32).unwrap();
        let b = NdArray::full((2, 2, 2, 2), 2.0f32).unwrap();
        let c = NdArray::div(&a, &b).unwrap();
        let expected = NdArray::full((2, 2, 2, 2), 4.0f32).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_affine_and_affine_assign() {
        let a = NdArray::<f64>::ones((3, 3)).unwrap();
        let b = a.affine(3., 2.).unwrap();
        let expected = NdArray::new(&[[5., 5., 5.],[5.,5.,5.],[5.,5.,5.]]).unwrap();
        assert!(b.allclose(&expected, 1e-6, 1e-6));

        let a2 = NdArray::<f64>::ones((3, 3)).unwrap();
        a2.affine_assign(3., 2.).unwrap();
        assert!(a2.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_add_scalar() {
        let a = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = 10.0f32;
        let c = NdArray::add(&a, b).unwrap();
        let expected = NdArray::new(&[11.0f32, 12.0, 13.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_sub_scalar() {
        let a = NdArray::new(&[10.0f32, 20.0, 30.0]).unwrap();
        let b = 5.0f32;
        let c = NdArray::sub(&a, b).unwrap();
        let expected = NdArray::new(&[5.0f32, 15.0, 25.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_mul_scalar() {
        let a = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = 2.0f32;
        let c = NdArray::mul(&a, b).unwrap();
        let expected = NdArray::new(&[2.0f32, 4.0, 6.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_div_scalar() {
        let a = NdArray::new(&[4.0f32, 9.0, 16.0]).unwrap();
        let b = 2.0f32;
        let c = NdArray::div(&a, b).unwrap();
        let expected = NdArray::new(&[2.0f32, 4.5, 8.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_minimum_scalar() {
        let a = NdArray::new(&[1.0f32, 5.0, 3.0]).unwrap();
        let b = 4.0f32;
        let c = NdArray::minimum(&a, b).unwrap();
        let expected = NdArray::new(&[1.0f32, 4.0, 3.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_maximum_scalar() {
        let a = NdArray::new(&[1.0f32, 5.0, 3.0]).unwrap();
        let b = 4.0f32;
        let c = NdArray::maximum(&a, b).unwrap();
        let expected = NdArray::new(&[4.0f32, 5.0, 4.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_add_assign_scalar() {
        let a = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = 5.0f32;
        a.add_assign(b).unwrap();
        let expected = NdArray::new(&[6.0f32, 7.0, 8.0]).unwrap();
        assert!(a.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_sub_assign_scalar() {
        let a = NdArray::new(&[10.0f32, 20.0, 30.0]).unwrap();
        let b = 5.0f32;
        a.sub_assign(b).unwrap();
        let expected = NdArray::new(&[5.0f32, 15.0, 25.0]).unwrap();
        assert!(a.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_mul_assign_scalar() {
        let a = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = 3.0f32;
        a.mul_assign(b).unwrap();
        let expected = NdArray::new(&[3.0f32, 6.0, 9.0]).unwrap();
        assert!(a.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_div_assign_scalar() {
        let a = NdArray::new(&[8.0f32, 12.0, 20.0]).unwrap();
        let b = 2.0f32;
        a.div_assign(b).unwrap();
        let expected = NdArray::new(&[4.0f32, 6.0, 10.0]).unwrap();
        assert!(a.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_minimum_assign_scalar() {
        let a = NdArray::new(&[1.0f32, 5.0, 3.0]).unwrap();
        let b = 4.0f32;
        a.minimum_assign(b).unwrap();
        let expected = NdArray::new(&[1.0f32, 4.0, 3.0]).unwrap();
        assert!(a.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_maximum_assign_scalar() {
        let a = NdArray::new(&[1.0f32, 5.0, 3.0]).unwrap();
        let b = 4.0f32;
        a.maximum_assign(b).unwrap();
        let expected = NdArray::new(&[4.0f32, 5.0, 4.0]).unwrap();
        assert!(a.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_eq_ne_scalar() {
        let a = NdArray::new(&[1, 2, 3]).unwrap();
        let b = 2;

        // NdArray vs scalar
        let eq_res = a.eq(b).unwrap();
        let expected_eq = NdArray::new(&[false, true, false]).unwrap();
        assert_eq!(eq_res.to_vec(), expected_eq.to_vec());

        let ne_res = a.ne(b).unwrap();
        let expected_ne = NdArray::new(&[true, false, true]).unwrap();
        assert_eq!(ne_res.to_vec(), expected_ne.to_vec());
    }

    #[test]
    fn test_lt_le_gt_ge_scalar() {
        let a = NdArray::new(&[1, 2, 3]).unwrap();
        let b = 2;

        let lt_res = a.lt(b).unwrap();
        assert_eq!(lt_res.to_vec(), [true, false, false]);

        let le_res = a.le(b).unwrap();
        assert_eq!(le_res.to_vec(), [true, true, false]);

        let gt_res = a.gt(b).unwrap();
        assert_eq!(gt_res.to_vec(), [false, false, true]);

        let ge_res = a.ge(b).unwrap();
        assert_eq!(ge_res.to_vec(), [false, true, true]);
    }

    #[test]
    fn test_eq_ne_ndarray() {
        let a = NdArray::new(&[1, 2, 3]).unwrap();
        let b = NdArray::new(&[1, 0, 3]).unwrap();

        let eq_res = a.eq(&b).unwrap();
        assert_eq!(eq_res.to_vec(), [true, false, true]);

        let ne_res = a.ne(&b).unwrap();
        assert_eq!(ne_res.to_vec(), [false, true, false]);
    }

    #[test]
    fn test_lt_le_gt_ge_ndarray() {
        let a = NdArray::new(&[1, 2, 3]).unwrap();
        let b = NdArray::new(&[2, 2, 1]).unwrap();

        let lt_res = a.lt(&b).unwrap();
        assert_eq!(lt_res.to_vec(), [true, false, false]);

        let le_res = a.le(&b).unwrap();
        assert_eq!(le_res.to_vec(), [true, true, false]);

        let gt_res = a.gt(&b).unwrap();
        assert_eq!(gt_res.to_vec(), [false, false, true]);

        let ge_res = a.ge(&b).unwrap();
        assert_eq!(ge_res.to_vec(), [false, true, true]);
    }

    #[test]
    fn test_comparison_2d() {
        let a = NdArray::new(&[[1, 2], [3, 4]]).unwrap();
        let b = NdArray::new(&[[2, 2], [1, 5]]).unwrap();

        let eq_res = a.eq(&b).unwrap();
        let expected_eq = NdArray::new(&[[false, true], [false, false]]).unwrap();
        assert_eq!(eq_res.to_vec(), expected_eq.to_vec());

        let gt_res = a.gt(&b).unwrap();
        let expected_gt = NdArray::new(&[[false, false], [true, false]]).unwrap();
        assert_eq!(gt_res.to_vec(), expected_gt.to_vec());

        // NdArray vs scalar
        let le_res = a.le(3).unwrap();
        let expected_le = NdArray::new(&[[true, true], [true, false]]).unwrap();
        assert_eq!(le_res.to_vec(), expected_le.to_vec());
    }

    #[test]
    fn test_std_ops() {
        let a = NdArray::new(&[[1., 2.], [3., 4.]]).unwrap();
        let b = NdArray::new(&[[2., 2.], [1., 5.]]).unwrap();
        let _ = (a + b).unwrap();

        let a = NdArray::new(&[[1., 2.], [3., 4.]]).unwrap();
        let b = NdArray::new(&[[2., 2.], [1., 5.]]).unwrap();
        let _ = (&a + &b).unwrap();

        let a = NdArray::new(&[[1., 2.], [3., 4.]]).unwrap();
        let b = NdArray::new(&[[2., 2.], [1., 5.]]).unwrap();
        let _ = (a + &b).unwrap();

        let a = NdArray::new(&[[1., 2.], [3., 4.]]).unwrap();
        let b = NdArray::new(&[[2., 2.], [1., 5.]]).unwrap();
        let _ = (&a + b).unwrap();
    }
}
