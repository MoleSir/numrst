use std::marker::PhantomData;

use crate::linalg::LinalgError;
use crate::{Error, FloatDType, Indexer, NdArray, NumDType, Result, Storage, WithDType};
use crate::{StorageMut, StorageRef};

pub trait AsVectorView<'a, T: WithDType> : Sized {
    fn len(&self) -> usize;
    fn stride(&self) -> usize;
    fn storage_get_uncheck(&self, storage_index: usize) -> T;
    fn from_ndarray(arr: &'a NdArray<T>) -> Result<Self>;

    fn get(&self, index: usize) -> Option<T> {
        if index >= self.len() {
            None
        } else {
            Some(self.storage_get_uncheck(self.storage_index(index)))
        }
    }

    #[inline]
    fn g(&self, index: usize) -> T {
        self.storage_get_uncheck(self.storage_index(index))
    }

    fn eqal(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            false 
        } else {
            for i in 0..self.len() {
                if self.g(i) != other.g(i) {
                    return false;
                }
            }
            true
        }
    }

    fn copy(&self) -> NdArray<T> {
        let data: Vec<_> = (0..self.len())
            .map(|index| self.g(index))
            .collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, self.len())
    }

    fn iter(&'a self) -> VectorViewIter<'a, T, Self> {
        VectorViewIter { 
            view: self, 
            index: 0,
            _marker: PhantomData,
        }
    }

    fn to_vec(&'a self) -> Vec<T> {
        self.iter().collect()
    }

    #[inline]
    fn storage_index(&self, index: usize) -> usize {
        self.stride() * index
    }
}

pub trait AsVectorViewNum<'a, T: NumDType> : AsVectorView<'a, T> {
    fn dot(&self, rhs: &Self) -> Result<T> {
        if self.len() != rhs.len() {
            Err(Error::LenMismatchVector { lhs: self.len(), rhs: rhs.len(), op: "dot" })?;
        }

        let v = (0..self.len())
            .map(|index| self.g(index) * rhs.g(index))
            .sum::<T>();

        Ok(v)
    }
}

pub trait AsVectorViewFloat<'a, T: FloatDType> : AsVectorViewNum<'a, T> {
    fn norm(&'a self) -> T {
        self.iter()
            .map(|v| v.powi(2))
            .sum::<T>()
            .sqrt()
    }
}

pub trait AsVectorViewMut<'a, T: WithDType> : AsVectorView<'a, T> {
    fn storage_set_uncheck(&mut self, storage_index: usize, value: T);
    fn from_ndarray_mut(arr: &'a mut NdArray<T>) -> Result<Self>;

    fn set(&mut self, index: usize, value: T) -> Option<()> {
        if index >= self.len() {
            None
        } else {
            self.storage_set_uncheck(self.storage_index(index), value);
            Some(())
        }
    }

    #[inline]
    fn s(&mut self, index: usize, value: T) {
        self.storage_set_uncheck(self.storage_index(index), value);
    }

    fn swap(&mut self, other: &mut Self) -> Result<()> {
        if self.len() != other.len() {
            return Err(Error::LenMismatchVector { lhs: self.len(), rhs: other.len(), op: "swap" });
        }
    
        let self_stride = self.stride();
        let other_stride = other.stride();
    
        for i in 0..self.len() {
            let si = self_stride * i;
            let oi = other_stride * i;
    
            let a = self.storage_get_uncheck(si);
            let b = other.storage_get_uncheck(oi);
    
            self.storage_set_uncheck(si, b);
            other.storage_set_uncheck(oi, a);
        }
    
        Ok(())
    }

    fn copy_from(&'a mut self, arr: &NdArray<T>) -> Result<()> {
        let view = arr.vector_view()?;
        self.copy_from_view(&view)
    }

    fn copy_from_view<'b>(&mut self, source: &impl AsVectorView<'b, T>) -> Result<()> {
        if self.len() != source.len() {
            Err(Error::LenMismatchVector { lhs: self.len(), rhs: source.len(), op: "copy_from_view" })?;
        }

        for i in 0..self.len() {
            self.s(i, source.g(i));
        }

        Ok(())
    }
}

pub trait AsVectorViewMutNum<'a, T: NumDType> : AsVectorViewMut<'a, T> {
    fn assign_op<F>(&mut self, rhs: T, f: F) 
    where 
        F: Fn(T, T) -> T
    {
        for i in 0..self.len() {
            let v = self.g(i);
            self.s(i, f(v, rhs));
        }
    } 

    #[inline]
    fn add_assign(&mut self, add: T) {
        self.assign_op(add, |a, b| a + b);
    }

    fn sub_assign(&mut self, sub: T) {
        self.assign_op(sub, |a, b| a - b);
    }

    fn mul_assign(&mut self, mul: T) {
        self.assign_op(mul, |a, b| a * b);
    }

    fn div_assign(&mut self, div: T) {
        self.assign_op(div, |a, b| a / b);
    }
}

pub struct VectorViewIter<'a, T: WithDType, V: AsVectorView<'a, T>> {
    view: &'a V,
    index: usize,
    _marker: PhantomData<T>,
}

impl<'a, T: WithDType, V: AsVectorView<'a, T>> Iterator for VectorViewIter<'a, T, V> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.len() {
            None
        } else {
            let value = self.view.g(self.index);
            self.index += 1;
            Some(value)
        }
    }
}

impl<'a, T: WithDType, V: AsVectorView<'a, T>> ExactSizeIterator for VectorViewIter<'a, T, V> {
    fn len(&self) -> usize {
        self.view.len()
    }
}

////////////////////////////////////////////////////////////////////////
///              VectorView
////////////////////////////////////////////////////////////////////////

pub struct VectorView<'a, T: WithDType> {
    pub(crate) storage: StorageRef<'a, T>,
    pub(crate) len: usize,
    pub(crate) stride: usize,
}

impl<'a, T: WithDType> AsVectorView<'a, T> for VectorView<'a, T> {
    fn from_ndarray(array: &'a NdArray<T>) -> Result<Self> {
        let _ = array.dims1()?;
        let len = array.layout().dims()[0];
        let stride = array.layout().stride()[0];
        let start_offset = array.layout().start_offset();
        let storage = array.storage_ref(start_offset);
        Ok(Self { len, stride, storage })
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    fn storage_get_uncheck(&self, storage_index: usize) -> T {
        self.storage.get_unchecked(storage_index)
    }
}

impl<'a, T: NumDType> AsVectorViewNum<'a, T> for VectorView<'a, T> {}
impl<'a, T: FloatDType> AsVectorViewFloat<'a, T> for VectorView<'a, T> {}

impl<'a, T: WithDType> VectorView<'a, T> {
    pub fn clone(&'a self) -> Self {
        Self {
            len: self.len,
            storage: self.storage.clone(),
            stride: self.stride
        }
    }

    pub fn take(&'a self, size: usize) -> Result<VectorView<'a, T>> {
        if size > self.len {
            Err(LinalgError::VectorIndexOutOfRange { index: size, len: self.len, op: "drop" })?;
        }
        Ok(Self {
            storage: self.storage.clone(),
            len: size,
            stride: self.stride
        })
    }  

    pub fn drop(&'a self, size: usize) -> Result<VectorView<'a, T>> {
        if size > self.len {
            Err(LinalgError::VectorIndexOutOfRange { index: size, len: self.len, op: "drop" })?;
        }
        let drop_size = self.stride * size;
        let new_len = self.len - size;
        Ok(Self {
            storage: self.storage.slice(drop_size),
            len: new_len,
            stride: self.stride
        })
    }  

    pub fn slice<I>(&'a self, index: I) -> Result<Self>
    where
        I: Into<Indexer>,
    {
        let index: Indexer = index.into();
        let (begin, step, size) = index.begin_step_size(self.len(), "slice_vector")?;
        let new_storage = self.storage.slice(begin * self.stride());
    
        Ok(Self {
            storage: new_storage,
            len: size,
            stride: self.stride() * step,
        })
    }
}

////////////////////////////////////////////////////////////////////////
///              VectorViewMut
////////////////////////////////////////////////////////////////////////

pub struct VectorViewMut<'a, T: WithDType> {
    pub(crate) storage: StorageMut<'a, T>,
    pub(crate) len: usize,
    pub(crate) stride: usize,
}

impl<'a, T: WithDType> AsVectorView<'a, T> for VectorViewMut<'a, T> {
    fn from_ndarray(array: &'a NdArray<T>) -> Result<Self> {
        let _ = array.dims1()?;
        let len = array.layout().dims()[0];
        let stride = array.layout().stride()[0];
        let start_offset = array.layout().start_offset();
        let storage = array.storage_mut(start_offset);
        Ok(Self { len, stride, storage })
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    fn storage_get_uncheck(&self, storage_index: usize) -> T {
        self.storage.get_unchecked(storage_index)
    }
}

impl<'a, T: NumDType> AsVectorViewNum<'a, T> for VectorViewMut<'a, T> {}
impl<'a, T: FloatDType> AsVectorViewFloat<'a, T> for VectorViewMut<'a, T> {}

impl<'a, T: WithDType> AsVectorViewMut<'a, T> for VectorViewMut<'a, T> {
    fn storage_set_uncheck(&mut self, storage_index: usize, value: T) {
        self.storage.set_unchecked(storage_index, value);
    }

    fn from_ndarray_mut(array: &'a mut NdArray<T>) -> Result<Self> {
        let _ = array.dims1()?;
        let len = array.layout().dims()[0];
        let stride = array.layout().stride()[0];
        let start_offset = array.layout().start_offset();
        let storage = array.storage_mut(start_offset);
        Ok(Self { len, stride, storage })
    }
}

impl<'a, T: NumDType> AsVectorViewMutNum<'a, T> for VectorViewMut<'a, T> {}

impl<'a, T: WithDType> VectorViewMut<'a, T> {
    pub fn clone(&self) -> VectorView<'_, T> {
        VectorView {
            len: self.len,
            storage: self.storage.clone(),
            stride: self.stride
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArray;

    #[test]
    fn test_get_and_iter() {
        let arr = NdArray::arange(0i32, 5).unwrap(); // [0,1,2,3,4]
        let view = VectorView::from_ndarray(&arr).unwrap();

        // get / g
        assert_eq!(view.get(0), Some(0));
        assert_eq!(view.g(4), 4);
        assert_eq!(view.get(5), None);

        // iter
        let collected: Vec<_> = view.iter().collect();
        assert_eq!(collected, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_dot_and_norm() {
        let arr1 = NdArray::new(&[1.0f32, 2.0, 3.0]).unwrap();
        let arr2 = NdArray::new(&[4.0f32, -5.0, 6.0]).unwrap();

        let v1 = VectorView::from_ndarray(&arr1).unwrap();
        let v2 = VectorView::from_ndarray(&arr2).unwrap();

        let dot = v1.dot(&v2).unwrap();
        assert!((dot - (1.0*4.0 + 2.0*-5.0 + 3.0*6.0)).abs() < 1e-6);

        let norm = v1.norm();
        assert!((norm - (1.0f32*1.0 + 2.0*2.0 + 3.0*3.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_drop_and_stride() {
        let arr = NdArray::arange(0i32, 6).unwrap(); // [0,1,2,3,4,5]
        let view = VectorView::from_ndarray(&arr).unwrap();
        let dropped = view.drop(2).unwrap(); // 从 index=2 开始

        assert_eq!(dropped.len(), 4);
        let collected: Vec<_> = dropped.iter().collect();
        assert_eq!(collected, [2, 3, 4, 5]);
    }

    #[test]
    fn test_set_and_assign_ops() {
        let mut arr = NdArray::new(&[1i32, 2, 3]).unwrap();
        let mut v = VectorViewMut::from_ndarray_mut(&mut arr).unwrap();

        // set
        v.set(1, 99);
        assert_eq!(v.to_vec(), [1, 99, 3]);

        // add_assign
        v.add_assign(1);
        assert_eq!(v.to_vec(), [2, 100, 4]);

        // mul_assign
        v.mul_assign(2);
        assert_eq!(v.to_vec(), [4, 200, 8]);

        drop(v);
        assert_eq!(arr.to_vec(), [4, 200, 8]);
    }

    #[test]
    fn test_swap_between_views() {
        let mut arr1 = NdArray::new(&[1i32, 2, 3]).unwrap();
        let mut arr2 = NdArray::new(&[4i32, 5, 6]).unwrap();

        {
            let mut v1 = VectorViewMut::from_ndarray_mut(&mut arr1).unwrap();
            let mut v2 = VectorViewMut::from_ndarray_mut(&mut arr2).unwrap();
            v1.swap(&mut v2).unwrap();
        }

        assert_eq!(arr1.to_vec(), [4, 5, 6]);
        assert_eq!(arr2.to_vec(), [1, 2, 3]);
    }

}
