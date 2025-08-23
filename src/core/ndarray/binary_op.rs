use crate::{Error, Layout, Result, Shape, Storage};
use super::NdArray;

macro_rules! binary_op_impl {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            if shape.element_count() == 0 {
                return Ok(self.clone());
            }
            let storage = Self::binary_op::<$op_name>(
                &self.storage(), &rhs.storage(), self.layout(), rhs.layout()
            )?;
            
            Ok(Self::from_storage(storage, shape))
        }
    };
}

impl NdArray {
    binary_op_impl!(add, Add);
    binary_op_impl!(mul, Mul);
    binary_op_impl!(sub, Sub);
    binary_op_impl!(div, Div);
    binary_op_impl!(min, Minimum);
    binary_op_impl!(max, Maximum);

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

    pub fn binary_op<B: BinaryOp>(lhs: &Storage, rhs: &Storage, lhs_layout: &Layout, rhs_layout: &Layout) -> Result<Storage> {
        match (lhs, rhs) {
            (Storage::F32(lhs), Storage::F32(rhs)) => {
                let data = Self::binary_map(lhs, rhs, lhs_layout, rhs_layout, B::f32);
                Ok(Storage::F32(data))
            }
            (Storage::F64(lhs), Storage::F64(rhs)) => {
                let data = Self::binary_map(lhs, rhs, lhs_layout, rhs_layout, B::f64);
                Ok(Storage::F64(data))
            }
            (Storage::U32(lhs), Storage::U32(rhs)) => {
                let data = Self::binary_map(lhs, rhs, lhs_layout, rhs_layout, B::u32);
                Ok(Storage::U32(data))
            }
            (Storage::I32(lhs), Storage::I32(rhs)) => {
                let data = Self::binary_map(lhs, rhs, lhs_layout, rhs_layout, B::i32);
                Ok(Storage::I32(data))
            }
            _ => {
                // This should be covered by the dtype check above.
                Err(Error::DTypeMismatchBinaryOp {
                    lhs: lhs.dtype(),
                    rhs: rhs.dtype(),
                    op: B::NAME,
                })
            }
        }
    }
    

    fn binary_map<T, U, F>(lhs: &[T], rhs: &[T], lhs_layout: &Layout, rhs_layout: &Layout, mut f: F) -> Vec<U> 
    where T: Copy,
        U: Copy, 
        F: FnMut(T, T) -> U
    {
        assert_eq!(lhs_layout.dims(), rhs_layout.dims(), "lhs dims != rhs dim2");

        let mut output = vec![];
        for (lhs_index, rhs_index) in lhs_layout.to_index().zip(rhs_layout.to_index()) {
            output.push( f(lhs[lhs_index], rhs[rhs_index]) );
        }
        output
    }
}

pub trait BinaryOp {
    const NAME: &'static str;
    fn i32(v1: i32, v2: i32) -> i32;
    fn u32(v1: u32, v2: u32) -> u32;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;    
}

struct Add;
struct Div;
struct Mul;
struct Sub;
struct Maximum;
struct Minimum;

macro_rules! binary_op {
    ($op:ident, $name:literal, $func:tt) => {
        impl BinaryOp for $op {        
            const NAME: &'static str = $name;

            fn u32(v1: u32, v2: u32) -> u32 {
                v1 $func v2
            }
        
            fn i32(v1: i32, v2: i32) -> i32 {
                v1 $func v2
            }
        
            fn f32(v1: f32, v2: f32) -> f32 {
                v1 $func v2
            }
        
            fn f64(v1: f64, v2: f64) -> f64 {
                v1 $func v2
            }
        }
    };
}

macro_rules! binary_op_minmax {
    ($op:ident, $name:literal, $func:tt) => {
        impl BinaryOp for $op {        
            const NAME: &'static str = $name;

            fn u32(v1: u32, v2: u32) -> u32 {
                if v1 $func v2 { v2 } else { v1 }
            }
        
            fn i32(v1: i32, v2: i32) -> i32 {
                if v1 $func v2 { v2 } else { v1 }
            }
        
            fn f32(v1: f32, v2: f32) -> f32 {
                if v1 $func v2 { v2 } else { v1 }
            }
        
            fn f64(v1: f64, v2: f64) -> f64 {
                if v1 $func v2 { v2 } else { v1 }
            }
        }
    };
}

binary_op!(Add, "add", +);
binary_op!(Sub, "sub", -);
binary_op!(Mul, "mul", *);
binary_op!(Div, "div", /);
binary_op_minmax!(Minimum, "min", >);
binary_op_minmax!(Maximum, "max", <);

#[allow(unused)]
#[cfg(test)]
mod tests {
    use super::*;
    
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

        let min_res = NdArray::min(&a, &b).unwrap();
        let max_res = NdArray::max(&a, &b).unwrap();

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
