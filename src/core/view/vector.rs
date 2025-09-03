use crate::linalg::LinalgError;
use crate::{FloatDType, NdArray, NumDType, Result, WithDType};
use crate::{StorageMut, StorageRef};
use super::{AsVectorView, AsVectorViewFloat, AsVectorViewMut, AsVectorViewMutNum, AsVectorViewNum};

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
