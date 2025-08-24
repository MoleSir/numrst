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
    binary_op_impl!(minimum, Minimum);
    binary_op_impl!(maximum, Maximum);

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
binary_op_minmax!(Minimum, "minimum", >);
binary_op_minmax!(Maximum, "maximum", <);

macro_rules! unary_op_impl {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            if self.element_count() == 0 {
                return Ok(self.clone());
            }
            let storage = Self::unary_op::<$op_name>(&self.storage(), self.layout())?;
            Ok(Self::from_storage(storage, self.shape()))
        }
        
    };
}

impl NdArray {
    unary_op_impl!(exp, Exp);
    unary_op_impl!(log, Log);
    unary_op_impl!(sin, Sin);
    unary_op_impl!(cos, Cos);
    unary_op_impl!(abs, Abs);
    unary_op_impl!(neg, Neg);
    unary_op_impl!(recip, Recip);
    unary_op_impl!(sqr, Sqr);
    unary_op_impl!(sqrt, Sqrt);
    unary_op_impl!(gelu, Gelu);
    unary_op_impl!(erf, Erf);
    unary_op_impl!(relu, Relu);
    unary_op_impl!(tanh, Tanh);
    unary_op_impl!(floor, Floor);
    unary_op_impl!(ceil, Ceil);
    unary_op_impl!(round, Round);

    pub fn unary_op<B: UnaryOp>(storage: &Storage, layout: &Layout) -> Result<Storage> {
        match storage {
            Storage::U32(vec) => {
                let output = Self::unary_map(vec, layout, B::u32);
                Ok(Storage::U32(output))
            }
            Storage::I32(vec) => {
                let output = Self::unary_map(vec, layout, B::i32);
                Ok(Storage::I32(output))
            }
            Storage::F32(vec) => {
                let output = Self::unary_map(vec, layout, B::f32);
                Ok(Storage::F32(output))
            }
            Storage::F64(vec) => {
                let output = Self::unary_map(vec, layout, B::f64);
                Ok(Storage::F64(output))
            }
        }
    }

    fn unary_map<T, U, F>(vec: &[T], layout: &Layout, mut f: F) -> Vec<U> 
    where T: Copy,
        U: Copy,
        F: FnMut(T) -> U
    {
        let mut output = vec![];
        for index in layout.to_index() {
            output.push( f(vec[index]) );
        }
        output
    }
}

pub trait UnaryOp {
    const NAME: &'static str;
    fn i32(v: i32) -> i32;
    fn u32(v: u32) -> u32;
    fn f32(v: f32) -> f32;
    fn f64(v: f64) -> f64;
}

struct Exp;
struct Log;
struct Sin;
struct Cos;
struct Recip;
struct Sqr;
struct Sqrt;
struct Gelu;
struct Erf;
struct Relu;
struct Tanh;
struct Floor;
struct Ceil;
struct Round;

struct Abs;
struct Neg;

macro_rules! unary_op {
    ($op:ident, $name:literal, $a: ident, $e:expr) => {
        impl UnaryOp for $op {
            const NAME: &'static str = $name;

            fn f32($a: f32) -> f32 {
                $e
            }
        
            fn f64($a: f64) -> f64 {
                $e
            }
    
            fn u32(_: u32) -> u32 {
                unimplemented!("no {} function for u32", $name)
            }
        
            fn i32(_: i32) -> i32 {
                unimplemented!("no {} function for i32", $name)
            }
        }
    };
}

unary_op!(Exp, "exp", v, v.exp());
unary_op!(Log, "log", v, v.ln());
unary_op!(Sin, "sin", v, v.sin());
unary_op!(Cos, "cos", v, v.cos());
unary_op!(Tanh, "tanh", v, v.tanh());
unary_op!(Recip, "recip", v, v.recip());
unary_op!(Sqr, "sqr", v, v * v);
unary_op!(Sqrt, "sqrt", v, v.sqrt());
unary_op!(Floor, "floor", v, v.floor());
unary_op!(Ceil, "ceil", v, v.ceil());
unary_op!(Round, "round", v, v.round());
unary_op!(Relu, "relu", v, if v > 0.0 { v } else { 0.0 });

impl UnaryOp for Erf {
    const NAME: &'static str = "erf";

    fn f32(v: f32) -> f32 {
        libm::erff(v)
    }

    fn f64(v: f64) -> f64 {
        libm::erf(v)
    }

    fn u32(_: u32) -> u32 {
        unimplemented!("no erf function for u32")
    }

    fn i32(_: i32) -> i32 {
        unimplemented!("no erf function for i32")
    }
}

impl UnaryOp for Gelu {
    const NAME: &'static str = "gelu";

    fn f32(v: f32) -> f32 {
        let c: f32 = (2.0 / std::f32::consts::PI).sqrt();
        0.5 * v * (1.0 + (c * (v + 0.044715 * v.powi(3))).tanh())
    }

    fn f64(v: f64) -> f64 {
        let c: f64 = (2.0 / std::f64::consts::PI).sqrt();
        0.5 * v * (1.0 + (c * (v + 0.044715 * v.powi(3))).tanh())
    }

    fn u32(_: u32) -> u32 {
        unimplemented!("no gelu function for u32")
    }

    fn i32(_: i32) -> i32 {
        unimplemented!("no gelu function for i32")
    }
}

impl UnaryOp for Abs {
    const NAME: &'static str = "abs";

    fn f32(v: f32) -> f32 {
        v.abs()
    }

    fn f64(v: f64) -> f64 {
        v.abs()
    }

    fn u32(v: u32) -> u32 {
        v
    }

    fn i32(v: i32) -> i32 {
        v.abs()
    }
}

impl UnaryOp for Neg {
    const NAME: &'static str = "abs";

    fn f32(v: f32) -> f32 {
        -v
    }

    fn f64(v: f64) -> f64 {
        -v
    }

    fn u32(_: u32) -> u32 {
        unimplemented!("no abs function for u32")
    }

    fn i32(v: i32) -> i32 {
        -v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exp_log() {
        let a = NdArray::new(&[0.0f32, 1.0, 2.0]).unwrap();
        let exp_a = a.exp().unwrap();
        let log_a = exp_a.log().unwrap();
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
    fn test_recip_sqr_sqrt() {
        let a = NdArray::new(&[1.0f32, 4.0, 9.0]).unwrap();
        let recip_a = a.recip().unwrap();
        let sqr_a = a.sqr().unwrap();
        let sqrt_a = a.sqrt().unwrap();

        let expected_recip = NdArray::new(&[1.0f32, 0.25, 1.0/9.0]).unwrap();
        let expected_sqr = NdArray::new(&[1.0f32, 16.0, 81.0]).unwrap();
        let expected_sqrt = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();

        assert!(recip_a.allclose(&expected_recip, 1e-6, 1e-6));
        assert!(sqr_a.allclose(&expected_sqr, 1e-6, 1e-6));
        assert!(sqrt_a.allclose(&expected_sqrt, 1e-6, 1e-6));
    }

    #[test]
    fn test_gelu_erf() {
        let a = NdArray::new(&[0.0f32, 1.0]).unwrap();
        let gelu_a = a.gelu().unwrap();
        let erf_a = a.erf().unwrap();

        // GELU(0) = 0, GELU(1) ~ 0.841
        let expected_gelu = NdArray::new(&[0.0f32, 0.841_3447]).unwrap();
        let expected_erf = NdArray::new(&[0.0f32, 0.842_7007]).unwrap();
        // println!("{:?}", gelu_a.iter().collect::<Vec<_>>());
        // println!("{:?}", erf_a.iter().collect::<Vec<_>>());

        assert!(gelu_a.allclose(&expected_gelu, 1e-4, 1e-3));
        assert!(erf_a.allclose(&expected_erf, 1e-4, 1e-3));
    }

    #[test]
    fn test_relu_silu_tanh() {
        let a = NdArray::new(&[-1.0f32, 0.0, 1.0]).unwrap();
        let relu_a = a.relu().unwrap();
        let tanh_a = a.tanh().unwrap();

        let expected_relu = NdArray::new(&[0.0f32, 0.0, 1.0]).unwrap();
        let expected_tanh = NdArray::new(&[-0.7615942f32, 0.0, 0.7615942]).unwrap();


        assert!(relu_a.allclose(&expected_relu, 1e-5, 1e-5));
        assert!(tanh_a.allclose(&expected_tanh, 1e-5, 1e-5));
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
