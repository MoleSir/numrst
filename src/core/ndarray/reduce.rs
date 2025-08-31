use crate::{Dim, FloatCategory, FloatDType, IntCategory, IntDType, NumCategory, NumDType, Result, Storage, WithDType};
use super::{iter::{NdArrayIter, ResettableIterator, StorageRef}, NdArray};

macro_rules! reduce_impl {
    ($fn_name:ident, $reduce:ident) => {
        pub fn $fn_name<'a, D: Dim>(&'a self, axis: D) -> Result<NdArray<<$reduce as ReduceOp<T, DimArrayIter<'a, T>>>::Output>> {
            self.reduec_axis_op(axis, $reduce::op, stringify!($fn_name))
        }
    };
}

macro_rules! reduce_category_impl {
    ($fn_name:ident, $reduce:ident) => {
        pub fn $fn_name<'a, D: Dim>(&'a self, axis: D) -> Result<NdArray<<$reduce as ReduceOp<T, DimArrayIter<'a, T>>>::Output>> 
        where $reduce: ReduceOpByCategory<T, DimArrayIter<'a, T>>
        {
            self.reduec_axis_op(axis, $reduce::op, stringify!($fn_name))
        }
    };
}

impl<T: NumDType> NdArray<T> {
    reduce_impl!(sum_axis, ReduceSum);
    reduce_impl!(product_axis, ReduceProduct);
    reduce_impl!(min_axis, ReduceMin);
    reduce_impl!(argmin_axis, ReduceArgMin);
    reduce_impl!(max_axis, ReduceMax);
    reduce_impl!(argmax_axis, ReduceArgMax);
    reduce_category_impl!(mean_axis, ReduceMean);
    reduce_category_impl!(var_axis, ReduceVar);
    reduce_category_impl!(std_axis, ReduceStd);
}

impl<T: NumDType> NdArray<T> {
    pub fn sum(&self) -> T {
        self.iter().sum::<T>()
    }

    pub fn product(&self) -> T {
        self.iter().product::<T>()
    }

    pub fn min(&self) -> T {
        self.iter().reduce(|acc, e| T::minimum(acc, e)).unwrap()
    }

    pub fn max(&self) -> T {
        self.iter().reduce(|acc, e| T::maximum(acc, e)).unwrap()
    }

    pub fn mean<'a>(&'a self) -> <ReduceMean as ReduceOp<T, NdArrayIter<'a, T>>>::Output 
    where ReduceMean: ReduceOpByCategory<T, NdArrayIter<'a, T>>
    {
        self.reduce_op(ReduceMean::op)
    }

    pub fn var<'a>(&'a self) -> <ReduceVar as ReduceOp<T, NdArrayIter<'a, T>>>::Output 
        where ReduceVar: ReduceOpByCategory<T, NdArrayIter<'a, T>>
    {
        self.reduce_op(ReduceVar::op)
    }

    pub fn std<'a>(&'a self) -> <ReduceStd as ReduceOp<T, NdArrayIter<'a, T>>>::Output 
        where ReduceStd: ReduceOpByCategory<T, NdArrayIter<'a, T>>
    {
        self.reduce_op(ReduceStd::op)
    }
}

impl NdArray<bool> {
    pub fn all(&self) -> bool {
        self.iter().all(|a| a)
    }

    pub fn any(&self) -> bool {
        self.iter().any(|a| a)
    } 

    pub fn all_axis<D: Dim>(&self, axis: D) -> Result<NdArray<bool>> {
        self.reduec_axis_op(axis, ReduceAll::op, "all")
    }

    pub fn any_axis<D: Dim>(&self, axis: D) -> Result<NdArray<bool>> {
        self.reduec_axis_op(axis, ReduceAny::op, "any")
    }
}

impl<T: WithDType> NdArray<T> {
    fn reduec_axis_op<'a, F, R: WithDType, D: Dim>(&'a self, reduce_dim: D, f: F, op_name: &'static str) -> Result<NdArray<R>> 
    where 
        F: Fn(&mut DimArrayIter<'a, T>) -> R
    {
        let reduce_dim = reduce_dim.to_index(self.shape(), op_name)?;
        assert!(reduce_dim < self.layout().dims().len());
        let reduce_dim_stride = self.layout().stride()[reduce_dim];
        let reduce_dim_size = self.layout().dims()[reduce_dim];

        let dst_len = self.layout().element_count() / reduce_dim_size;
        let mut dst: Vec<R> = Vec::with_capacity(dst_len);
        let dst_to_set = dst.spare_capacity_mut();

        let layout = self.layout().narrow(reduce_dim, 0, 1)?;
        for (dst_index, src_index) in layout.storage_indices().enumerate() {
            let arr: DimArray<'_, T> = DimArray {
                src: self.storage_ref(src_index),
                size: reduce_dim_size,
                stride: reduce_dim_stride
            };
            let mut iter: DimArrayIter<'_, T> = arr.into_iter();
            dst_to_set[dst_index].write(f(&mut iter));
        }
        unsafe { dst.set_len(dst_len) };

        let storage = Storage::new(dst);
        let mut shape = self.dims().to_vec();
        shape.remove(reduce_dim);

        Ok(NdArray::<R>::from_storage(storage, shape))
    }

    fn reduce_op<'a, F, R: WithDType>(&'a self, f: F) -> R 
    where 
        F: Fn(&mut NdArrayIter<'a, T>) -> R
    {
        let mut iter = self.iter();
        f(&mut iter)
    }
}

pub trait ExactSizeResetIterator : ExactSizeIterator + ResettableIterator {}

pub trait ReduceOp<D: WithDType, I: ExactSizeResetIterator<Item = D>> {
    type Output: WithDType;
    fn op(arr: &mut I) -> Self::Output;
}

/// Mean and Var are special reduce op
/// For float type, we need it return self
/// But for int type, it should be return bool, but you can't impl both NumDType and FloatDType for ReduceOp
/// Rust think maybe there are some type impl NumDType and FloatDType
/// So we use ReduceOpByCategory to solve this question
/// check https://geo-ant.github.io/blog/2021/mutually-exclusive-traits-rust/ for more detial :)
pub trait ReduceOpByCategory<T: NumDType, I: ExactSizeResetIterator<Item = T>, C: NumCategory = <T as NumDType>::Category> {
    type Output: NumDType;
    fn category_op(arr: &mut I) -> Self::Output;
}

pub struct ReduceMean;

impl<F: FloatDType, I: ExactSizeResetIterator<Item = F>> ReduceOpByCategory<F, I, FloatCategory> for ReduceMean {
    type Output = F;
    fn category_op(arr: &mut I) -> Self::Output {
        let sum = ReduceSum::op(arr);
        sum / F::from_usize(arr.len())
    }
}

impl<T: IntDType, I: ExactSizeResetIterator<Item = T>> ReduceOpByCategory<T, I, IntCategory> for ReduceMean {
    type Output = f64;
    fn category_op(arr: &mut I) -> Self::Output {
        let sum = ReduceSum::op(arr);
        sum.to_f64() / arr.len() as f64
    }
}

impl<D: NumDType, I: ExactSizeResetIterator<Item = D>> ReduceOp<D, I> for ReduceMean 
where
    Self: ReduceOpByCategory<D, I>
{
    type Output = <ReduceMean as ReduceOpByCategory<D, I>>::Output;
    fn op(arr: &mut I) -> Self::Output {
        ReduceMean::category_op(arr)
    }
} 

pub struct ReduceVar;

impl<F: FloatDType, I: ExactSizeResetIterator<Item = F>> ReduceOpByCategory<F, I, FloatCategory> for ReduceVar {
    type Output = F;
    fn category_op(arr: &mut I) -> Self::Output {
        let len = arr.len();
        let mean = ReduceMean::op(arr);
        arr.reset();
        let sum = arr   
            .map(|v| (v - mean).powi(2))
            .sum::<F>();
        sum / F::from_usize(len)
    }
}

impl<T: IntDType, I: ExactSizeResetIterator<Item = T>> ReduceOpByCategory<T, I, IntCategory> for ReduceVar {
    type Output = f64;
    fn category_op(arr: &mut I) -> Self::Output {
        let len = arr.len();
        let mean: f64 = ReduceMean::category_op(arr); 
        arr.reset();
        let sum = arr   
            .map(|v| (v.to_f64()- mean))
            .sum::<f64>();
        sum.to_f64() / len as f64
    }
}

impl<D: NumDType, I: ExactSizeResetIterator<Item = D>> ReduceOp<D, I> for ReduceVar 
where
    Self: ReduceOpByCategory<D, I>
{
    type Output = <ReduceVar as ReduceOpByCategory<D, I>>::Output;
    fn op(arr: &mut I) -> Self::Output {
        ReduceVar::category_op(arr)
    }
} 

pub struct ReduceStd;

impl<F: FloatDType, I: ExactSizeResetIterator<Item = F>> ReduceOpByCategory<F, I, FloatCategory> for ReduceStd {
    type Output = F;
    fn category_op(arr: &mut I) -> Self::Output {
        let var = ReduceVar::op(arr);
        var.sqrt()
    }
}

impl<T: IntDType, I: ExactSizeResetIterator<Item = T>> ReduceOpByCategory<T, I, IntCategory> for ReduceStd {
    type Output = f64;
    fn category_op(arr: &mut I) -> Self::Output {
        let var: f64 = ReduceMean::category_op(arr); 
        var.sqrt()
    }
}

impl<D: NumDType, I: ExactSizeResetIterator<Item = D>> ReduceOp<D, I> for ReduceStd 
where
    Self: ReduceOpByCategory<D, I>
{
    type Output = <ReduceStd as ReduceOpByCategory<D, I>>::Output;
    fn op(arr: &mut I) -> Self::Output {
        ReduceStd::category_op(arr)
    }
} 

pub struct ReduceAll;
impl<I: ExactSizeResetIterator<Item = bool>> ReduceOp<bool, I> for ReduceAll {
    type Output = bool;
    fn op(arr: &mut I) -> Self::Output {
        arr.into_iter().all(|b| b)
    }
}

pub struct ReduceAny;
impl<I: ExactSizeResetIterator<Item = bool>> ReduceOp<bool, I> for ReduceAny {
    type Output = bool;
    fn op(arr: &mut I) -> Self::Output {
        arr.into_iter().any(|b| b)
    }
}

pub struct ReduceSum;
impl<D: NumDType, I: ExactSizeResetIterator<Item = D>> ReduceOp<D, I> for ReduceSum {
    type Output = D;
    fn op(arr: &mut I) -> Self::Output {
        arr.into_iter().sum::<D>()
    }
} 

pub struct ReduceProduct;
impl<D: NumDType, I: ExactSizeResetIterator<Item = D>> ReduceOp<D, I> for ReduceProduct {
    type Output = D;
    fn op(arr: &mut I) -> Self::Output {
        arr.into_iter().product::<D>()
    }
} 

pub struct ReduceMin;
impl<D: NumDType, I: ExactSizeResetIterator<Item = D>> ReduceOp<D, I> for ReduceMin {
    type Output = D;
    fn op(arr: &mut I) -> Self::Output {
        arr.into_iter()
            .reduce(|a, b| D::minimum(a, b)).unwrap()
    }
} 

pub struct ReduceArgMin;
impl<D: NumDType, I: ExactSizeResetIterator<Item = D>> ReduceOp<D, I> for ReduceArgMin {
    type Output = u32;
    fn op(arr: &mut I) -> Self::Output {
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
impl<D: NumDType, I: ExactSizeResetIterator<Item = D>> ReduceOp<D, I> for ReduceMax {
    type Output = D;
    fn op(arr: &mut I) -> Self::Output {
        arr.into_iter()
            .reduce(|a, b| D::maximum(a, b)).unwrap()
    }
} 

pub struct ReduceArgMax;
impl<D: NumDType, I: ExactSizeResetIterator<Item = D>> ReduceOp<D, I> for ReduceArgMax {
    type Output = u32;
    fn op(arr: &mut I) -> Self::Output {
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

pub struct DimArray<'a, T> {
    src: StorageRef<'a, T>,
    size: usize,
    stride: usize
}

impl<'a, T: WithDType> DimArray<'a, T> {
    pub fn get(&self, index: usize) -> T {
        self.src.get_unchecked(index * self.stride)
    }

    #[allow(unused)]
    pub fn to_vec(&self) -> Vec<T> {
        let mut v = vec![];
        for i in 0..self.size {
            v.push(self.get(i));
        }
        v
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

impl<'a, T: WithDType> ExactSizeIterator for DimArrayIter<'a, T> {
    fn len(&self) -> usize {
        self.array.size
    }
}

impl<'a, T: WithDType> ResettableIterator for DimArrayIter<'a, T> {
    fn reset(&mut self) {
        self.index = 0
    }
}

impl<'a, T: WithDType> ExactSizeResetIterator for DimArrayIter<'a, T> {}
impl<'a, T: WithDType> ExactSizeResetIterator for NdArrayIter<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // sum(axis=0) -> [4, 6, 8]
        let arr = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let s = arr.sum_axis(0).unwrap();
        let expected = NdArray::new(&[4, 6, 8]).unwrap();
        assert!(s.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_sum_matrix_axis1() {
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // sum(axis=1) -> [6, 12]
        let arr = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let s = arr.sum_axis(1).unwrap();
        let expected = NdArray::new(&[6, 12]).unwrap();
        assert!(s.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_sum_ones_axis() {
        // ones( (2,3), dtype=I32 )
        // [[1,1,1],
        //  [1,1,1]]
        let arr = NdArray::ones((2, 3)).unwrap();
        let s0 = arr.sum_axis(0).unwrap(); // -> [2,2,2]
        let s1 = arr.sum_axis(1).unwrap(); // -> [3,3]

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
        let p = arr.product_axis(0).unwrap();
        let expected = NdArray::new(&[3, 8, 15]).unwrap();
        assert!(p.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_min_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // min(axis=0) -> [1, 1, 0]
        let arr = NdArray::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.min_axis(0).unwrap();
        let expected = NdArray::new(&[1, 1, 0]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_aragmin_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // min(axis=0) -> [1, 1, 0]
        let arr = NdArray::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.argmin_axis(0).unwrap();
        let expected = NdArray::new(&[0u32, 1, 1]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_max_matrix_axis1() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // max(axis=1) -> [3, 3]
        let arr = NdArray::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.max_axis(1).unwrap();
        let expected = NdArray::new(&[3, 3]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_argmax_matrix_axis1() {
        // [[1, 2, 3],
        //  [3, 1, 0]]
        // max(axis=1) -> [3, 3]
        let arr = NdArray::new(&[[1, 2, 3], [3, 1, 0]]).unwrap();
        let m = arr.argmax_axis(1).unwrap();
        let expected = NdArray::new(&[2u32, 0]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_mean_matrix_axis0() {
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // mean(axis=0) -> [2, 3, 4]
        let arr: NdArray<f32> = NdArray::new(&[[1., 2., 3.], [3., 4., 5.]]).unwrap();
        let m = arr.mean_axis(0).unwrap();
        let expected = NdArray::new(&[2.0, 3.0, 4.0]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_mean_matrix_axis1() {
        // [[1, 2, 3],  mean -> 2
        //  [3, 4, 5]]  mean -> 4
        let arr: NdArray<i32> = NdArray::new(&[[1, 2, 3], [3, 4, 5]]).unwrap();
        let m = arr.mean_axis(1).unwrap();
        let expected = NdArray::new(&[2., 4.]).unwrap();
        assert!(m.allclose(&expected, 1e-5, 1e-8));
    }

    #[test]
    fn test_var_1d() {
        // NumPy: np.var([1, 2, 3]) = 2/3 = 0.666...
        let arr: NdArray<f64> = NdArray::new(&[1.0f64, 2.0, 3.0]).unwrap();
        let v = arr.var_axis(0).unwrap();
        let expected = NdArray::new(2.0 / 3.0).unwrap();
        assert!(v.allclose(&expected, 1e-6, 1e-12));
    }

    #[test]
    fn test_var_2d_axis0() {
        // arr = [[1, 2, 3],
        //        [4, 5, 6]]
        // np.var(arr, axis=0) = [2.25, 2.25, 2.25]
        let arr: NdArray<f32> = NdArray::new(&[[1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0]]).unwrap();
        let v = arr.var_axis(0).unwrap();
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
        let v = arr.var_axis(1).unwrap();
        let expected = NdArray::new(&[2.0/3.0, 2.0/3.0]).unwrap();
        assert!(v.allclose(&expected, 1e-6, 1e-12));
    }


    #[test]
    fn test_sum_1d() {
        let arr = NdArray::new(&[1, 2, 3, 4]).unwrap();
        assert_eq!(arr.sum(), 10);
    }

    #[test]
    fn test_sum_2d() {
        let arr = NdArray::new(&[[1, 2], [3, 4]]).unwrap();
        assert_eq!(arr.sum(), 10);
    }

    #[test]
    fn test_product() {
        let arr = NdArray::new(&[1, 2, 3, 4]).unwrap();
        assert_eq!(arr.product(), 24);
    }

    #[test]
    fn test_min_max_1d() {
        let arr = NdArray::new(&[5, 2, 9, -1, 0]).unwrap();
        assert_eq!(arr.min(), -1);
        assert_eq!(arr.max(), 9);
    }

    #[test]
    fn test_min_max_2d() {
        let arr = NdArray::new(&[[3, 7, 1], [9, -2, 5]]).unwrap();
        assert_eq!(arr.min(), -2);
        assert_eq!(arr.max(), 9);
    }

    #[test]
    fn test_all_same() {
        let arr = NdArray::full((3, 3), 7).unwrap();
        assert_eq!(arr.sum(), 7 * 9);
        assert_eq!(arr.product(), 7_i32.pow(9));
        assert_eq!(arr.min(), 7);
        assert_eq!(arr.max(), 7);
    }

    #[test]
    fn test_float_array() {
        let arr = NdArray::new(&[1.0f32, -3.5, 2.5]).unwrap();
        assert!((arr.sum() - 0.0).abs() < 1e-6);
        assert_eq!(arr.min(), -3.5);
        assert_eq!(arr.max(), 2.5);
    }

    #[test]
    fn test_var_all_1d() {
        // NumPy: np.var([1, 2, 3]) = 2/3 ≈ 0.666...
        let arr: NdArray<f64> = NdArray::new(&[1.0f64, 2.0, 3.0]).unwrap();
        let v = arr.var();
        let expected = 2.0 / 3.0;
        assert!((v - expected).abs() < 1e-12);
    }
    
    #[test]
    fn test_std_all_1d() {
        // NumPy: np.std([1, 2, 3]) = sqrt(2/3) ≈ 0.8164965809
        let arr: NdArray<f64> = NdArray::new(&[1.0f64, 2.0, 3.0]).unwrap();
        let s = arr.std();
        let expected = (2.0 / 3.0f64).sqrt();
        assert!((s - expected).abs() < 1e-12);
    }
    
    #[test]
    fn test_var_all_2d() {
        // arr = [[1, 2, 3],
        //        [4, 5, 6]]
        // NumPy: np.var(arr) = 2.916666...
        let arr: NdArray<f32> = NdArray::new(&[[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0]]).unwrap();
        let v = arr.var();
        let expected = 2.9166667_f32;
        assert!((v - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_std_all_2d() {
        // arr = [[1, 2, 3],
        //        [4, 5, 6]]
        // NumPy: np.std(arr) = sqrt(2.916666...) ≈ 1.7078251
        let arr: NdArray<f32> = NdArray::new(&[[1.0, 2.0, 3.0],
                                     [4.0, 5.0, 6.0]]).unwrap();
        let s = arr.std();
        let expected = 1.7078251_f32;
        assert!((s - expected).abs() < 1e-6);
    }
}
