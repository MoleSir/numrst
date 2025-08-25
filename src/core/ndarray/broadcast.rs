use std::sync::Arc;
use crate::{NumDType, Result, Shape, WithDType};
use super::{NdArray, NdArrayId, NdArrayImpl};

impl<T: WithDType> NdArray<T> {
    /// Broadcast the input tensor to the target shape. This returns an error if the input shape is
    /// not compatible with the target shape.
    ///
    /// If the input shape is `i_1, i_2, ... i_k`, the target shape has to have `k` dimensions or
    /// more and shape `j_1, ..., j_l, t_1, t_2, ..., t_k`. The dimensions `j_1` to `j_l` can have
    /// any value, the dimension `t_a` must be equal to `i_a` if `i_a` is different from 1. If
    /// `i_a` is equal to 1, any value can be used.
    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let ndarry_ = NdArrayImpl {
            id: NdArrayId::new(),
            storage: self.0.storage.clone(),
            layout: self.layout().broadcast_as(shape)?,
            dtype: self.dtype(),
        };
        Ok(NdArray(Arc::new(ndarry_)))
    }
}

macro_rules! broadcast_binary_op {
    ($fn_name:ident, $inner_fn_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let lhs = self;
            let shape = lhs
                .shape()
                .broadcast_shape_binary_op(rhs.shape(), stringify!($fn_name))?;
            let l_broadcast = shape != *lhs.shape();
            let r_broadcast = shape != *rhs.shape();
            match (l_broadcast, r_broadcast) {
                (true, true) => lhs
                    .broadcast_as(&shape)?
                    .$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (false, true) => lhs.$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (true, false) => lhs.broadcast_as(&shape)?.$inner_fn_name(rhs),
                (false, false) => lhs.$inner_fn_name(rhs),
            }
        }
    };
}

macro_rules! broadcast_cmp_op {
    ($fn_name:ident, $inner_fn_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<NdArray<bool>> {
            let lhs = self;
            let shape = lhs
                .shape()
                .broadcast_shape_binary_op(rhs.shape(), stringify!($fn_name))?;
            let l_broadcast = shape != *lhs.shape();
            let r_broadcast = shape != *rhs.shape();
            match (l_broadcast, r_broadcast) {
                (true, true) => lhs
                    .broadcast_as(&shape)?
                    .$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (false, true) => lhs.$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (true, false) => lhs.broadcast_as(&shape)?.$inner_fn_name(rhs),
                (false, false) => lhs.$inner_fn_name(rhs),
            }
        }
    };
}

impl<T: NumDType> NdArray<T> {
    broadcast_binary_op!(broadcast_add, add);
    broadcast_binary_op!(broadcast_mul, mul);
    broadcast_binary_op!(broadcast_sub, sub);
    broadcast_binary_op!(broadcast_div, div);
    broadcast_binary_op!(broadcast_maximum, maximum);
    broadcast_binary_op!(broadcast_minimum, minimum);
    broadcast_cmp_op!(broadcast_eq, eq);
    broadcast_cmp_op!(broadcast_ne, ne);
    broadcast_cmp_op!(broadcast_lt, lt);
    broadcast_cmp_op!(broadcast_le, le);
    broadcast_cmp_op!(broadcast_gt, gt);
    broadcast_cmp_op!(broadcast_ge, ge);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_add_scalar() {
        let a = NdArray::new(&[1., 2., 3.]).unwrap();
        let b = NdArray::new(&[10.]).unwrap(); // 标量 (1,)
        let res = a.broadcast_add(&b).unwrap();

        let expected = NdArray::new(&[11., 12., 13.]).unwrap();
        assert!(res.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_broadcast_add_vector_to_matrix() {
        let a = NdArray::new(&[[1., 2., 3.], [4., 5., 6.]]).unwrap(); // (2,3)
        let b = NdArray::new(&[10., 20., 30.]).unwrap(); // (3,)
        let res = a.broadcast_add(&b).unwrap();

        let expected = NdArray::new(&[[11., 22., 33.], [14., 25., 36.]]).unwrap();
        assert!(res.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_broadcast_mul() {
        let a = NdArray::new(&[[1., 2.], [3., 4.]]).unwrap(); // (2,2)
        let b = NdArray::new(&[10.]).unwrap(); // (1,)
        let res = a.broadcast_mul(&b).unwrap();

        let expected = NdArray::new(&[[10., 20.], [30., 40.]]).unwrap();
        assert!(res.allclose(&expected, 1e-6, 1e-6));
    }

    #[test]
    fn test_broadcast_maximum_minimum() {
        let a = NdArray::new(&[1., 5., 3.]).unwrap();
        let b = NdArray::new(&[2., 2., 2.]).unwrap();

        let max_res = a.broadcast_maximum(&b).unwrap();
        let min_res = a.broadcast_minimum(&b).unwrap();

        let expected_max = NdArray::new(&[2., 5., 3.]).unwrap();
        let expected_min = NdArray::new(&[1., 2., 2.]).unwrap();

        assert!(max_res.allclose(&expected_max, 1e-6, 1e-6));
        assert!(min_res.allclose(&expected_min, 1e-6, 1e-6));
    }

    #[test]
    fn test_broadcast_comparisons() {
        let a = NdArray::new(&[1., 2., 3.]).unwrap();
        let b = NdArray::new(&[2.]).unwrap();

        let eq = a.broadcast_eq(&b).unwrap();
        let lt = a.broadcast_lt(&b).unwrap();
        let le = a.broadcast_le(&b).unwrap();
        let gt = a.broadcast_gt(&b).unwrap();
        let ge = a.broadcast_ge(&b).unwrap();
        let ne = a.broadcast_ne(&b).unwrap();

        assert_eq!(eq.to_vec(), [false, true, false]);
        assert_eq!(lt.to_vec(), [true, false, false]);
        assert_eq!(le.to_vec(), [true, true, false]);
        assert_eq!(gt.to_vec(), [false, false, true]);
        assert_eq!(ge.to_vec(), [false, true, true]);
        assert_eq!(ne.to_vec(), [true, false, true]);
    }

    #[test]
    fn test_broadcast_div() {
        let a = NdArray::new(&[10., 20., 30.]).unwrap();
        let b = NdArray::new(&[10.]).unwrap();
        let res = a.broadcast_div(&b).unwrap();

        let expected = NdArray::new(&[1., 2., 3.]).unwrap();
        assert!(res.allclose(&expected, 1e-6, 1e-6));
    }
}
