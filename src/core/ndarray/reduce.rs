use num_traits::Pow;
use crate::{Dim, Layout, NumDType, Result, Storage};
use super::NdArray;

macro_rules! reduce_impl {
    ($fn_name:ident, $reduce:ident) => {
        pub fn $fn_name<D: Dim>(&self, dim: D) -> Result<NdArray<<$reduce as ReduceOp<T>>::Output>> {
            self.reduec_op(dim, $reduce::op, stringify!($fn_name))
        }
    };
}

impl<T: NumDType> NdArray<T> {
    reduce_impl!(sum, ReduceSum);
    reduce_impl!(product, ReduceProduct);
    reduce_impl!(min, ReduceMin);
    reduce_impl!(argmin, ReduceArgMin);
    reduce_impl!(max, ReduceMax);
    reduce_impl!(argmax, ReduceArgMax);
}

impl NdArray<i32> {
    fn mean_f(arr: DimArray<'_, i32>) -> f64 {
        let sum = ReduceSum::op(arr.clone());
        let len = arr.into_iter().count();
        sum as f64 / len as f64
    }

    fn var_f(arr: DimArray<'_, i32>) -> f64 {
        let mean = Self::mean_f(arr.clone());
        let len = arr.len();
        let sum = arr.into_iter()
            .map(|v| (v as f64 - mean).pow(2) )
            .sum::<f64>();
        sum / len as f64
    }

    pub fn mean<D: Dim>(&self, dim: D) -> Result<NdArray<f64>> {
        self.reduec_op(dim, Self::mean_f, "mean")
    }

    pub fn var<D: Dim>(&self, dim: D) -> Result<NdArray<f64>> {
        self.reduec_op(dim, Self::var_f, "var")
    }
}

impl NdArray<u32> {
    fn mean_f(arr: DimArray<'_, u32>) -> f64 {
        let sum = ReduceSum::op(arr.clone());
        let len = arr.into_iter().count();
        sum as f64 / len as f64
    }

    fn var_f(arr: DimArray<'_, u32>) -> f64 {
        let mean = Self::mean_f(arr.clone());
        let len = arr.len();
        let sum = arr.into_iter()
            .map(|v| (v as f64 - mean).pow(2) )
            .sum::<f64>();
        sum / len as f64
    }

    pub fn mean<D: Dim>(&self, dim: D) -> Result<NdArray<f64>> {
        self.reduec_op(dim, Self::mean_f, "mean")
    }

    pub fn var<D: Dim>(&self, dim: D) -> Result<NdArray<f64>> {
        self.reduec_op(dim, Self::var_f, "var")
    }
}

impl NdArray<f32> {
    fn mean_f(arr: DimArray<'_, f32>) -> f32 {
        let sum = ReduceSum::op(arr.clone());
        let len = arr.into_iter().count();
        sum / len as f32
    }

    fn var_f(arr: DimArray<'_, f32>) -> f32 {
        let mean = Self::mean_f(arr.clone());
        let len = arr.len();
        let sum = arr.into_iter()
            .map(|v| (v - mean).pow(2) )
            .sum::<f32>();
        sum / len as f32
    }

    pub fn mean<D: Dim>(&self, dim: D) -> Result<NdArray<f32>> {
        self.reduec_op(dim, Self::mean_f, "mean")
    }

    pub fn var<D: Dim>(&self, dim: D) -> Result<NdArray<f32>> {
        self.reduec_op(dim, Self::var_f, "var")
    }
}

impl NdArray<f64> {
    fn mean_f(arr: DimArray<'_, f64>) -> f64 {
        let sum = ReduceSum::op(arr.clone());
        let len = arr.into_iter().count();
        sum / len as f64
    }

    fn var_f(arr: DimArray<'_, f64>) -> f64 {
        let mean = Self::mean_f(arr.clone());
        let len = arr.len();
        let sum = arr.into_iter()
            .map(|v| (v - mean).pow(2) )
            .sum::<f64>();
        sum / len as f64
    }

    pub fn mean<D: Dim>(&self, dim: D) -> Result<NdArray<f64>> {
        self.reduec_op(dim, Self::mean_f, "mean")
    }

    pub fn var<D: Dim>(&self, dim: D) -> Result<NdArray<f64>> {
        self.reduec_op(dim, Self::var_f, "var")
    }
}

impl<T: NumDType> NdArray<T> {
    fn reduec_op<F, R: NumDType, D: Dim>(&self, dim: D, f: F, op_name: &'static str) -> Result<NdArray<R>> 
    where 
        F: Fn(DimArray<'_, T>) -> R
    {
        fn _reduec_op<F, T: NumDType, R: NumDType>(src_storage: &Storage<T>, src_layout: &Layout, reduce_dim: usize, f: F) -> Result<Storage<R>> 
        where 
            F: Fn(DimArray<'_, T>) -> R,
        {   
            let src = src_storage.data();
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
    
            Ok(Storage::new(dst))
        }

        let dim = dim.to_index(self.shape(), op_name)?;
        let storage = _reduec_op(&self.storage(), self.layout(), dim, f)?;
        let mut shape = self.dims().to_vec();
        shape.remove(dim);
        Ok(NdArray::<R>::from_storage(storage, shape))
    }
}

pub trait ReduceOp<D: NumDType> {
    type Output: NumDType;
    fn op(arr: DimArray<'_, D>) -> Self::Output;
}

pub struct ReduceSum;
impl<D: NumDType> ReduceOp<D> for ReduceSum {
    type Output = D;
    fn op(arr: DimArray<'_, D>) -> Self::Output {
        arr.into_iter().sum::<D>()
    }
} 

pub struct ReduceProduct;
impl<D: NumDType> ReduceOp<D> for ReduceProduct {
    type Output = D;
    fn op(arr: DimArray<'_, D>) -> Self::Output {
        arr.into_iter().product::<D>()
    }
} 

pub struct ReduceMin;
impl<D: NumDType> ReduceOp<D> for ReduceMin {
    type Output = D;
    fn op(arr: DimArray<'_, D>) -> Self::Output {
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

pub struct ReduceArgMin;
impl<D: NumDType> ReduceOp<D> for ReduceArgMin {
    type Output = u32;
    fn op(arr: DimArray<'_, D>) -> Self::Output {
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

pub struct ReduceMax;
impl<D: NumDType> ReduceOp<D> for ReduceMax {
    type Output = D;
    fn op(arr: DimArray<'_, D>) -> Self::Output {
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

pub struct ReduceArgMax;
impl<D: NumDType> ReduceOp<D> for ReduceArgMax {
    type Output = u32;
    fn op(arr: DimArray<'_, D>) -> Self::Output {
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

#[derive(Debug, Clone)]
pub struct DimArray<'a, T> {
    src: &'a [T],
    size: usize,
    stride: usize
}

impl<'a, T: NumDType> DimArray<'a, T> {
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

impl<'a, T: NumDType> IntoIterator for DimArray<'a, T> {
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

impl<'a, T: NumDType> Iterator for DimArrayIter<'a, T> {
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
        let arr = NdArray::ones((2, 3)).unwrap();
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
        let arr: NdArray<f32> = NdArray::new(&[[1., 2., 3.], [3., 4., 5.]]).unwrap();
        let m = arr.mean(0).unwrap();
        let expected = NdArray::new(&[2.0, 3.0, 4.0]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_mean_matrix_axis1() {
        // [[1, 2, 3],  mean -> 2
        //  [3, 4, 5]]  mean -> 4
        let arr: NdArray<i32> = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let m = arr.mean(1).unwrap();
        let expected = NdArray::new(&[2., 4.]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_var_1d() {
        // NumPy: np.var([1, 2, 3]) = 2/3 = 0.666...
        let arr: NdArray<f64> = NdArray::new(&[1.0f64, 2.0, 3.0]).unwrap();
        let v = arr.var(0).unwrap();
        let expected = NdArray::new(&[2.0 / 3.0]).unwrap();
        assert!(v.allclose(&expected, 1e-6, 1e-12));
    }

    #[test]
    fn test_var_2d_axis0() {
        // arr = [[1, 2, 3],
        //        [4, 5, 6]]
        // np.var(arr, axis=0) = [2.25, 2.25, 2.25]
        let arr: NdArray<f32> = NdArray::new(&[[1.0, 2.0, 3.0],
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
        let arr: NdArray<f32> = NdArray::new(&[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]]).unwrap();
        let v = arr.var(1).unwrap();
        let expected = NdArray::new(&[2.0/3.0, 2.0/3.0]).unwrap();
        assert!(v.allclose(&expected, 1e-6, 1e-12));
    }
}
