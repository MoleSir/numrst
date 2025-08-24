use crate::{Dim, Layout, Result, Storage, WithDType};

use super::NdArray;


macro_rules! reduce_impl {
    ($fn_name:ident, $reduce:ident) => {
        pub fn $fn_name<D: Dim>(&self, dim: D) -> Result<Self> {
            let dim = dim.to_index(self.shape(), stringify!($fn_name))?;
            let storage = Self::reduec_op::<$reduce>(&self.storage(), self.layout(), dim)?;
            let mut shape = self.dims().to_vec();
            shape.remove(dim);
            Ok(Self::from_storage(storage, shape))
        }
    };
}

impl NdArray {
    reduce_impl!(sum, ReduceSum);
    reduce_impl!(product, ReduceProduct);
    reduce_impl!(min, ReduceMin);
    reduce_impl!(max, ReduceMax);
    reduce_impl!(mean, ReduceMean);

    fn reduec_op<R: ReduceOp>(storage: &Storage, layout: &Layout, dim: usize) -> Result<Storage> {
        fn _reduce_op<R: ReduceOp, T: WithDType>(src: &[T], src_layout: &Layout, dim_index: usize) -> Result<Vec<T>> {
            assert!(dim_index < src_layout.dims().len());
            let reduce_stride = src_layout.stride()[dim_index];
            let reduce_dim = src_layout.dims()[dim_index];

            let dst_len = src_layout.element_count() / reduce_dim;
            // Init dst vector with `init` method 
            let mut dst = vec![R::init(); dst_len];

            for (index, src_index) in src_layout.to_index().enumerate() {
                let mut dst_index = index;

                let (pre, post) = (dst_index / reduce_stride, dst_index % reduce_stride);
                dst_index = (pre / reduce_dim) * reduce_stride + post;
                // Call accumlate method
                R::accumlate(&mut dst[dst_index], &src[src_index]);
            }

            R::finalize(&mut dst, reduce_dim);

            Ok(dst)
        }

        match storage {
            Storage::U32(vec) => {
                let output = _reduce_op::<R, u32>(vec, layout, dim)?;
                Ok(Storage::U32(output))
            }
            Storage::I32(vec) => {
                let output = _reduce_op::<R, i32>(vec, layout, dim)?;
                Ok(Storage::I32(output))
            }
            Storage::F32(vec) => {
                let output = _reduce_op::<R, f32>(vec, layout, dim)?;
                Ok(Storage::F32(output))
            }
            Storage::F64(vec) => {
                let output = _reduce_op::<R, f64>(vec, layout, dim)?;
                Ok(Storage::F64(output))
            }
        }
    }
}

trait ReduceOp {
    fn init<D: WithDType>() -> D;
    fn accumlate<D: WithDType>(dst: &mut D, src: &D);
    fn finalize<D: WithDType>(_dst: &mut [D], _count: usize) {}
}

struct ReduceSum;
impl ReduceOp for ReduceSum {
    fn init<D: WithDType>() -> D {
        D::zero()
    }

    fn accumlate<D: WithDType>(dst: &mut D, src: &D) {
        *dst += *src;
    }
}

struct ReduceProduct;
impl ReduceOp for ReduceProduct {
    fn init<D: WithDType>() -> D {
        D::one()
    }

    fn accumlate<D: WithDType>(dst: &mut D, src: &D) {
        *dst *= *src;
    }
}


struct ReduceMin;
impl ReduceOp for ReduceMin {
    fn init<D: WithDType>() -> D {
        D::max_value()
    }

    fn accumlate<D: WithDType>(dst: &mut D, src: &D) {
        if *src < *dst {
            *dst = *src;
        }
    }
}

struct ReduceMax;
impl ReduceOp for ReduceMax {
    fn init<D: WithDType>() -> D {
        D::min_value()
    }

    fn accumlate<D: WithDType>(dst: &mut D, src: &D) {
        if *src > *dst {
            *dst = *src;
        }
    }
}

struct ReduceMean;
impl ReduceOp for ReduceMean {
    fn init<D: WithDType>() -> D {
        D::zero()
    }

    fn accumlate<D: WithDType>(dst: &mut D, src: &D) {
        *dst += *src;
    }

    fn finalize<D: WithDType>(dst: &mut [D], count: usize) {
        let count = D::from_f64(count as f64);
        for d in dst {
            *d = *d / count;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::DType;
    use super::*;

    #[test]
    fn test_sum_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // sum(axis=0) -> [4, 6, 8]
        let arr = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let s = arr.sum(0).unwrap();
        let expected = NdArray::new(&[4, 6, 8]).unwrap();
        assert!(s.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_sum_matrix_axis1() {
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // sum(axis=1) -> [6, 12]
        let arr = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let s = arr.sum(1).unwrap();
        let expected = NdArray::new(&[6, 12]).unwrap();
        assert!(s.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_sum_ones_axis() {
        // ones( (2,3), dtype=I32 )
        // [[1,1,1],
        //  [1,1,1]]
        let arr = NdArray::ones((2, 3), DType::I32).unwrap();
        let s0 = arr.sum(0).unwrap(); // -> [2,2,2]
        let s1 = arr.sum(1).unwrap(); // -> [3,3]

        let expected0 = NdArray::new(&[2, 2, 2]).unwrap();
        let expected1 = NdArray::new(&[3, 3]).unwrap();

        assert!(s0.allclose(&expected0, 1e-5, 1e-8));
        assert!(s1.allclose(&expected1, 1e-5, 1e-8));
    }

    #[test]
    fn test_product_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // product(axis=0) -> [3, 8, 15]
        let arr = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let p = arr.product(0).unwrap();
        let expected = NdArray::new(&[3, 8, 15]).unwrap();
        assert!(p.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_min_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // min(axis=0) -> [1, 1, 0]
        let arr = NdArray::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.min(0).unwrap();
        let expected = NdArray::new(&[1, 1, 0]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_max_matrix_axis1() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // max(axis=1) -> [3, 3]
        let arr = NdArray::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.max(1).unwrap();
        let expected = NdArray::new(&[3, 3]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_mean_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // mean(axis=0) -> [2, 3, 4]
        let arr = NdArray::new(&[[1., 2., 3.], [3., 4., 5.]]).unwrap();
        let m = arr.mean(0).unwrap();
        let expected = NdArray::new(&[2.0, 3.0, 4.0]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_mean_matrix_axis1() {
        // [[1, 2, 3],  mean -> 2
        //  [3, 4, 5]]  mean -> 4
        let arr = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let m = arr.mean(1).unwrap();
        let expected = NdArray::new(&[2, 4]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }
}
