use crate::{Error, FloatDType, Layout, Result, Shape, Storage, WithDType};
use super::NdArray;

macro_rules! binary_op_impl {
    ($fn_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            if shape.element_count() == 0 {
                return Ok(self.clone());
            }
            let storage = Self::binary_op(
                &self.storage(), &rhs.storage(), self.layout(), rhs.layout(), T::$fn_name
            );
            
            Ok(Self::from_storage(storage, shape))
        }
    };
}

impl<T: WithDType> NdArray<T> {
    binary_op_impl!(add);
    binary_op_impl!(mul);
    binary_op_impl!(sub);
    binary_op_impl!(div);
    binary_op_impl!(minimum);
    binary_op_impl!(maximum);

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

    pub fn binary_op<U, F>(lhs: &Storage<T>, rhs: &Storage<T>, lhs_layout: &Layout, rhs_layout: &Layout, mut f: F) -> Storage<U> 
    where 
        U: WithDType, 
        F: FnMut(T, T) -> U
    {
        assert_eq!(lhs_layout.dims(), rhs_layout.dims(), "lhs dims != rhs dim2");
        let lhs = lhs.data();
        let rhs = rhs.data();
        
        let mut output = vec![];
        for (lhs_index, rhs_index) in lhs_layout.to_index().zip(rhs_layout.to_index()) {
            output.push( f(lhs[lhs_index], rhs[rhs_index]) );
        }
        
        Storage::<U>::new(output)
    }
}

impl<T: WithDType> NdArray<T> {
    fn unary_op<U, F>(s: &Storage<T>, layout: &Layout, mut f: F) -> Storage<U> 
    where
        U: WithDType,
        F: FnMut(T) -> U
    {
        let vec = s.data();
        let mut output = vec![];
        for index in layout.to_index() {
            output.push( f(vec[index]) );
        }
        
        Storage::new(output)
    }
}

macro_rules! float_unary_op_impl {
    ($fn_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            if self.element_count() == 0 {
                return Ok(self.clone());
            }
            let storage = Self::unary_op(&self.storage(), self.layout(), F::$fn_name);
            Ok(Self::from_storage(storage, self.shape()))
        }
    };
}

impl<F: FloatDType> NdArray<F> {
    float_unary_op_impl!(exp);
    float_unary_op_impl!(sin);
    float_unary_op_impl!(cos);
    float_unary_op_impl!(sqrt);
    float_unary_op_impl!(tanh);
    float_unary_op_impl!(floor);
    float_unary_op_impl!(ceil);
    float_unary_op_impl!(round);
    float_unary_op_impl!(abs);
    float_unary_op_impl!(neg);
    float_unary_op_impl!(ln);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_log() {
        let a = NdArray::new(&[0.0f32, 1.0, 2.0]).unwrap();
        let exp_a = a.exp().unwrap();
        let log_a = exp_a.ln().unwrap();
        assert!(a.allclose(&log_a, 1e-5, 1e-8));
    }

    #[test]
    fn test_trig() {
        let a = NdArray::new(&[0.0f32, std::f32::consts::FRAC_PI_2]).unwrap();
        let sin_a = a.sin().unwrap();
        let cos_a = a.cos().unwrap();

        let expected_sin = NdArray::new(&[0.0f32, 1.0]).unwrap();
        let expected_cos = NdArray::new(&[1.0f32, 0.0]).unwrap();

        println!("{:?}", cos_a.iter().collect::<Vec<_>>());

        assert!(sin_a.allclose(&expected_sin, 1e-5, 1e-8));
        assert!(cos_a.allclose(&expected_cos, 1e-5, 8e-8));
    }

    #[test]
    fn test_abs_neg() {
        let a = NdArray::new(&[-1.0f32, 0.0, 2.0]).unwrap();
        let abs_a = a.abs().unwrap();
        let neg_a = a.neg().unwrap();

        let expected_abs = NdArray::new(&[1.0f32, 0.0, 2.0]).unwrap();
        let expected_neg = NdArray::new(&[1.0f32, 0.0, -2.0]).unwrap();

        assert!(abs_a.allclose(&expected_abs, 1e-6, 1e-6));
        assert!(neg_a.allclose(&expected_neg, 1e-6, 1e-6));
    }

    #[test]
    fn test_floor_ceil_round() {
        let a = NdArray::new(&[1.2f32, 2.7, -1.3]).unwrap();
        let floor_a = a.floor().unwrap();
        let ceil_a = a.ceil().unwrap();
        let round_a = a.round().unwrap();

        let expected_floor = NdArray::new(&[1.0f32, 2.0, -2.0]).unwrap();
        let expected_ceil = NdArray::new(&[2.0f32, 3.0, -1.0]).unwrap();
        let expected_round = NdArray::new(&[1.0f32, 3.0, -1.0]).unwrap();

        assert!(floor_a.allclose(&expected_floor, 1e-6, 1e-6));
        assert!(ceil_a.allclose(&expected_ceil, 1e-6, 1e-6));
        assert!(round_a.allclose(&expected_round, 1e-6, 1e-6));
    }
    
    #[test]
    fn test_add() {
        let a = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = NdArray::new(&[4.0f32, 5.0, 6.0]).unwrap();
        let c = NdArray::add(&a, &b).unwrap();
        let expected = NdArray::new(&[5.0f32, 7.0, 9.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_sub() {
        let a = NdArray::new(&[10.0f32, 20.0, 30.0]).unwrap();
        let b = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let c = NdArray::sub(&a, &b).unwrap();
        let expected = NdArray::new(&[9.0f32, 18.0, 27.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_mul() {
        let a = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let b = NdArray::new(&[2.0f32, 3.0, 4.0]).unwrap();
        let c = NdArray::mul(&a, &b).unwrap();
        let expected = NdArray::new(&[2.0f32, 6.0, 12.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_div() {
        let a = NdArray::new(&[4.0f32, 9.0, 16.0]).unwrap();
        let b = NdArray::new(&[2.0f32, 3.0, 4.0]).unwrap();
        let c = NdArray::div(&a, &b).unwrap();
        let expected = NdArray::new(&[2.0f32, 3.0, 4.0]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_min_max() {
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
    fn test_add_2d() {
        let a = NdArray::new(&[[1.0f32, 2.0], [3.0, 4.0]]).unwrap();
        let b = NdArray::new(&[[5.0f32, 6.0], [7.0, 8.0]]).unwrap();
        let c = NdArray::add(&a, &b).unwrap();
        let expected = NdArray::new(&[[6.0f32, 8.0], [10.0, 12.0]]).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_mul_3d() {
        let a = NdArray::new(&[
            [[1.0f32, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]).unwrap();

        let b = NdArray::new(&[
            [[2.0f32, 0.5], [1.0, 2.0]],
            [[0.5, 2.0], [1.5, 1.0]],
        ]).unwrap();

        let c = NdArray::mul(&a, &b).unwrap();

        let expected = NdArray::new(&[
            [[2.0f32, 1.0], [3.0, 8.0]],
            [[2.5, 12.0], [10.5, 8.0]],
        ]).unwrap();

        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_div_high_dim() {
        // shape (2, 2, 2, 2)
        let a = NdArray::fill((2, 2, 2, 2), 8.0f32).unwrap();
        let b = NdArray::fill((2, 2, 2, 2), 2.0f32).unwrap();
        let c = NdArray::div(&a, &b).unwrap();
        let expected = NdArray::fill((2, 2, 2, 2), 4.0f32).unwrap();
        assert!(c.allclose(&expected, 1e-6, 1e-6));
    }
}
