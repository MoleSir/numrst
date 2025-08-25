use num_traits::Pow;

use crate::{core::ndarray::construct::ToNdArray, Dim, Layout, Result, Storage, WithDType};
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
    reduce_impl!(argmin, ReduceArgMin);
    reduce_impl!(max, ReduceMax);
    reduce_impl!(argmax, ReduceArgMax);
    reduce_impl!(mean, ReduceMean);
    reduce_impl!(var, ReduceVar);

    fn reduec_op<R: ReduceOp>(storage: &Storage, layout: &Layout, dim: usize) -> Result<Storage> {
        fn _reduce_op<F, T: WithDType, R: WithDType>(src: &[T], src_layout: &Layout, reduce_dim: usize, f: F) -> Result<Vec<R>>  
        where 
            F: Fn(DimArray<'_, T>) -> R,

        {
            assert!(reduce_dim < src_layout.dims().len());
            let reduce_dim_stride = src_layout.stride()[reduce_dim];
            let reduce_dim_size = src_layout.dims()[reduce_dim];

            let dst_len = src_layout.element_count() / reduce_dim_size;
            let mut dst: Vec<R> = Vec::with_capacity(dst_len);
            let dst_to_set = dst.spare_capacity_mut();

            let layout = src_layout.narrow(reduce_dim, 0, 1)?;
            for (dst_index, src_index) in layout.to_index().enumerate() {
                let src = &src[src_index..];
                let arr: DimArray<'_, T> = DimArray {
                    src,
                    size: reduce_dim_size,
                    stride: reduce_dim_stride
                };
                dst_to_set[dst_index].write(f(arr));
            }

            unsafe { dst.set_len(dst_len) };
            Ok(dst)
        }

        match storage {
            Storage::U32(vec) => {
                let output = _reduce_op(vec, layout, dim, R::u32)?;
                Ok(output.to_storage()?)
            }
            Storage::I32(vec) => {
                let output = _reduce_op(vec, layout, dim, R::i32)?;
                Ok(output.to_storage()?)
            }
            Storage::F32(vec) => {
                let output = _reduce_op(vec, layout, dim, R::f32)?;
                Ok(output.to_storage()?)
            }
            Storage::F64(vec) => {
                let output = _reduce_op(vec, layout, dim,R::f64)?;
                Ok(output.to_storage()?)
            }
        }
    }

}

trait ReduceOp {
    type U32Ret: WithDType;
    type I32Ret: WithDType;
    type F32Ret: WithDType;
    type F64Ret: WithDType;

    fn u32<'a>(arr: DimArray<'a, u32>) -> Self::U32Ret;
    fn i32<'a>(arr: DimArray<'a, i32>) -> Self::I32Ret;
    fn f32<'a>(arr: DimArray<'a, f32>) -> Self::F32Ret;
    fn f64<'a>(arr: DimArray<'a, f64>) -> Self::F64Ret;
}

struct ReduceSum;
impl ReduceOp for ReduceSum {
    type I32Ret = i32;
    type U32Ret = u32;
    type F32Ret = f32;
    type F64Ret = f64;

    fn u32<'a>(arr: DimArray<'a, u32>) -> u32 {
        Self::sum(arr)
    }
    fn i32<'a>(arr: DimArray<'a, i32>) -> i32 {
        Self::sum(arr)
    }
    fn f32<'a>(arr: DimArray<'a, f32>) -> f32 {
        Self::sum(arr)
    }
    fn f64<'a>(arr: DimArray<'a, f64>) -> f64 {
        Self::sum(arr)
    }
}

impl ReduceSum {
    fn sum<'a, D: WithDType>(arr: DimArray<'a, D>) -> D {
        arr.into_iter().sum::<D>()
    }
}

struct ReduceProduct;
impl ReduceOp for ReduceProduct {
    type I32Ret = i32;
    type U32Ret = u32;
    type F32Ret = f32;
    type F64Ret = f64;

    fn u32<'a>(arr: DimArray<'a, u32>) -> u32 {
        Self::product(arr)
    }
    fn i32<'a>(arr: DimArray<'a, i32>) -> i32 {
        Self::product(arr)
    }
    fn f32<'a>(arr: DimArray<'a, f32>) -> f32 {
        Self::product(arr)
    }
    fn f64<'a>(arr: DimArray<'a, f64>) -> f64 {
        Self::product(arr)
    }
}

impl ReduceProduct {
    fn product<'a, D: WithDType>(arr: DimArray<'a, D>) -> D {
        arr.into_iter().product::<D>()
    }
}

struct ReduceMin;
impl ReduceOp for ReduceMin {
    type I32Ret = i32;
    type U32Ret = u32;
    type F32Ret = f32;
    type F64Ret = f64;

    fn u32<'a>(arr: DimArray<'a, u32>) -> u32 {
        Self::min(arr)
    }
    fn i32<'a>(arr: DimArray<'a, i32>) -> i32 {
        Self::min(arr)
    }
    fn f32<'a>(arr: DimArray<'a, f32>) -> f32 {
        Self::min(arr)
    }
    fn f64<'a>(arr: DimArray<'a, f64>) -> f64 {
        Self::min(arr)
    }
}

impl ReduceMin {
    fn min<'a, D: WithDType>(arr: DimArray<'a, D>) -> D {
        arr.into_iter()
            .reduce(|a, b| {
                if a.partial_cmp(&b) == Some(std::cmp::Ordering::Less) {
                    a
                } else {
                    b
                }
            }).unwrap()
    }
}

struct ReduceArgMin;
impl ReduceOp for ReduceArgMin {
    type I32Ret = u32;
    type U32Ret = u32;
    type F32Ret = u32;
    type F64Ret = u32;

    fn u32<'a>(arr: DimArray<'a, u32>) -> u32 {
        Self::argmin(arr)
    }
    fn i32<'a>(arr: DimArray<'a, i32>) -> u32 {
        Self::argmin(arr)
    }
    fn f32<'a>(arr: DimArray<'a, f32>) -> u32 {
        Self::argmin(arr)
    }
    fn f64<'a>(arr: DimArray<'a, f64>) -> u32 {
        Self::argmin(arr)
    }
}

impl ReduceArgMin {
    fn argmin<'a, D: WithDType>(arr: DimArray<'a, D>) -> u32 {
        assert!(arr.len() > 0);
        arr.into_iter()
            .enumerate()
            .reduce(|(ia, a), (ib, b)| {
                if a.partial_cmp(&b) == Some(std::cmp::Ordering::Less) {
                    (ia, a)
                } else {
                    (ib, b)
                }
            }).unwrap().0 as u32
    }
}

struct ReduceArgMax;
impl ReduceOp for ReduceArgMax {
    type I32Ret = u32;
    type U32Ret = u32;
    type F32Ret = u32;
    type F64Ret = u32;

    fn u32<'a>(arr: DimArray<'a, u32>) -> u32 {
        Self::argmax(arr)
    }
    fn i32<'a>(arr: DimArray<'a, i32>) -> u32 {
        Self::argmax(arr)
    }
    fn f32<'a>(arr: DimArray<'a, f32>) -> u32 {
        Self::argmax(arr)
    }
    fn f64<'a>(arr: DimArray<'a, f64>) -> u32 {
        Self::argmax(arr)
    }
}

impl ReduceArgMax {
    fn argmax<'a, D: WithDType>(arr: DimArray<'a, D>) -> u32 {
        assert!(arr.len() > 0);
        arr.into_iter()
            .enumerate()
            .reduce(|(ia, a), (ib, b)| {
                if a.partial_cmp(&b) == Some(std::cmp::Ordering::Greater) {
                    (ia, a)
                } else {
                    (ib, b)
                }
            }).unwrap().0 as u32
    }
}

struct ReduceMax;
impl ReduceOp for ReduceMax {
    type I32Ret = i32;
    type U32Ret = u32;
    type F32Ret = f32;
    type F64Ret = f64;

    fn u32<'a>(arr: DimArray<'a, u32>) -> u32 {
        Self::max(arr)
    }
    fn i32<'a>(arr: DimArray<'a, i32>) -> i32 {
        Self::max(arr)
    }
    fn f32<'a>(arr: DimArray<'a, f32>) -> f32 {
        Self::max(arr)
    }
    fn f64<'a>(arr: DimArray<'a, f64>) -> f64 {
        Self::max(arr)
    }
}

impl ReduceMax {
    fn max<'a, D: WithDType>(arr: DimArray<'a, D>) -> D {
        arr.into_iter()
            .reduce(|a, b| {
                if a.partial_cmp(&b) == Some(std::cmp::Ordering::Greater) {
                    a
                } else {
                    b
                }
            }).unwrap()
    }
}

struct ReduceMean;
impl ReduceOp for ReduceMean {
    type I32Ret = f64;
    type U32Ret = f64;
    type F32Ret = f32;
    type F64Ret = f64;

    fn i32<'a>(arr: DimArray<'a, i32>) -> Self::I32Ret {
        let len = arr.len();
        let sum = ReduceSum::i32(arr);
        sum as f64 / len as f64
    }

    fn u32<'a>(arr: DimArray<'a, u32>) -> Self::I32Ret {
        let len = arr.len();
        let sum = ReduceSum::u32(arr);
        sum as f64 / len as f64
    }

    fn f32<'a>(arr: DimArray<'a, f32>) -> f32 {
        let len = arr.len();
        let sum = ReduceSum::f32(arr);
        sum / len as f32
    }    

    fn f64<'a>(arr: DimArray<'a, f64>) -> f64 {
        let len = arr.len();
        let sum = ReduceSum::f64(arr);
        sum / len as f64
    }
}

struct ReduceVar;
impl ReduceOp for ReduceVar {
    type I32Ret = f64;
    type U32Ret = f64;
    type F32Ret = f32;
    type F64Ret = f64;

    fn i32<'a>(arr: DimArray<'a, i32>) -> Self::I32Ret {
        let len = arr.len();
        let mean = ReduceMean::i32(arr.clone());
        let sum = arr.into_iter()
            .map(|v| (v as f64 - mean).pow(2))
            .sum::<f64>();
        sum as f64 / len as f64
    }

    fn u32<'a>(arr: DimArray<'a, u32>) -> Self::I32Ret {
        let len = arr.len();
        let mean = ReduceMean::u32(arr.clone());
        let sum = arr.into_iter()
            .map(|v| (v as f64 - mean).pow(2))
            .sum::<f64>();
        sum as f64 / len as f64
    }

    fn f32<'a>(arr: DimArray<'a, f32>) -> f32 {
        let len = arr.len();
        let mean = ReduceMean::f32(arr.clone());
        let sum = arr.into_iter()
            .map(|v| (v - mean).pow(2))
            .sum::<f32>();
        sum / len as f32
    }    

    fn f64<'a>(arr: DimArray<'a, f64>) -> f64 {
        let len = arr.len();
        let mean = ReduceMean::f64(arr.clone());
        let sum = arr.into_iter()
            .map(|v| (v - mean).pow(2))
            .sum::<f64>();
        sum / len as f64
    }
}

#[derive(Debug, Clone)]
pub struct DimArray<'a, T> {
    src: &'a [T],
    size: usize,
    stride: usize
}

impl<'a, T: WithDType> DimArray<'a, T> {
    pub fn get(&self, index: usize) -> T {
        self.src[index * self.stride]
    }

    #[allow(unused)]
    pub fn to_vec(&self) -> Vec<T> {
        let mut v = vec![];
        for i in 0..self.size {
            v.push(self.get(i));
        }
        v
    }

    pub fn len(&self) -> usize {
        self.size
    }
}

impl<'a, T: WithDType> IntoIterator for DimArray<'a, T> {
    type IntoIter = DimArrayIter<'a, T>;
    type Item = T;
    fn into_iter(self) -> Self::IntoIter {
        DimArrayIter {
            array: self,
            index: 0,
        }
    }
}

pub struct DimArrayIter<'a, T> {
    array: DimArray<'a, T>,
    index: usize,
}

impl<'a, T: WithDType> Iterator for DimArrayIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if self.index >= self.array.size {
            None
        } else {
            let index = self.index;
            self.index += 1;
            Some(self.array.get(index))
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
    fn test_aragmin_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // min(axis=0) -> [1, 1, 0]
        let arr = NdArray::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.argmin(0).unwrap();
        let expected = NdArray::new(&[0u32, 1, 1]).unwrap();
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
    fn test_argmax_matrix_axis1() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // max(axis=1) -> [3, 3]
        let arr = NdArray::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.argmax(1).unwrap();
        let expected = NdArray::new(&[2u32, 0]).unwrap();
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
        let expected = NdArray::new(&[2., 4.]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_var_1d() {
        // NumPy: np.var([1, 2, 3]) = 2/3 = 0.666...
        let arr = NdArray::new(&[1.0f64, 2.0, 3.0]).unwrap();
        let v = arr.var(0).unwrap();
        let expected = NdArray::new(&[2.0 / 3.0]).unwrap();
        assert!(v.allclose(&expected, 1e-6, 1e-12));
    }

    #[test]
    fn test_var_2d_axis0() {
        // arr = [[1, 2, 3],
        //        [4, 5, 6]]
        // np.var(arr, axis=0) = [2.25, 2.25, 2.25]
        let arr = NdArray::new(&[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]]).unwrap();
        let v = arr.var(0).unwrap();
        let expected = NdArray::new(&[2.25, 2.25, 2.25]).unwrap();
        assert!(v.allclose(&expected, 1e-6, 1e-12));
    }

    #[test]
    fn test_var_2d_axis1() {
        // arr = [[1, 2, 3],
        //        [4, 5, 6]]
        // np.var(arr, axis=1) = [2/3, 2/3]
        let arr = NdArray::new(&[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]]).unwrap();
        let v = arr.var(1).unwrap();
        let expected = NdArray::new(&[2.0/3.0, 2.0/3.0]).unwrap();
        assert!(v.allclose(&expected, 1e-6, 1e-12));
    }
}
