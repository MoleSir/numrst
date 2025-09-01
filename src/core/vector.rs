use std::sync::Arc;
use crate::{Error, NdArray, NumDType, Result, Storage, WithDType};
use super::{Layout, NdArrayId, NdArrayImpl, Shape, StorageArc, StorageRef};

pub struct VectorLayout {
    pub(crate) len: usize,
    pub(crate) stride: usize,
    pub(crate) start_offset: usize,
}

pub struct Vector<T: WithDType> {
    pub(crate) storage: StorageArc<T>,
    pub(crate) layout: VectorLayout,
}

pub struct VectorView<'a, T: WithDType> {
    pub(crate) storage: StorageRef<'a, T>,
    pub(crate) len: usize,
    pub(crate) stride: usize,
}

impl<T: NumDType> Vector<T> {
    pub fn zeros(len: usize) -> Result<Self> {
        let data = vec![T::zero(); len];
        let storage = Storage::new(data);
        Ok(Self::from_storage(storage, len))
    }

    pub fn add_assign(&self, index: usize, value: T) -> Result<()> {
        let index = self.layout.get_storage_index(index)?;
        let mut storage = self.storage.write();
        let data = storage.data_mut(); 
        data[index] += value;
        Ok(())
    }

    pub fn dot(&self, rhs: &Self) -> Result<T> {
        if self.len() != rhs.len() {
            return Err(Error::LenMismatchVectorDot { lhs: self.len(), rhs: rhs.len() });
        }
        let sum = self.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum::<T>();
        Ok(sum)
    }
}

impl<T: WithDType> Vector<T> {
    pub fn to_ndarray(&self) -> NdArray<T> {
        let shape: Shape = (self.layout.len).into();
        let stride = [self.layout.stride].to_vec();
        let layout = Layout {
            shape, stride, start_offset: self.layout.start_offset
        };
        let arr = NdArrayImpl {
            id: NdArrayId::new(),
            storage: self.storage.clone(),
            layout
        };
        NdArray(Arc::new(arr))
    }

    pub fn from_ndarray(array: &NdArray<T>) -> Result<Self> {
        let _ = array.dims1()?;

        let layout = VectorLayout {
            len: array.layout().dims()[0],
            stride: array.layout().stride()[0],
            start_offset: array.layout().start_offset(),
        };
        
        Ok(Self {
            storage: array.storage_clone(),
            layout
        })
    }

    pub fn take(&self, len: usize) -> Result<Self> {
        let layout = self.layout.take(len)?;
        Ok(Self {
            storage: self.storage.clone(),
            layout
        })
    }

    pub fn drop(&self, len: usize) -> Result<Self> {
        let layout = self.layout.drop(len)?;
        Ok(Self {
            storage: self.storage.clone(),
            layout
        })
    }

    pub fn to_view<'a>(&'a self) -> VectorView<'a, T> {
        VectorView {
            storage: self.storage.get_ref(self.layout.start_offset),
            len: self.layout.len,
            stride: self.layout.stride,
        }
    }

    pub fn get(&self, index: usize) -> Result<T> {
        let storage_index = self.layout.get_storage_index(index)?;
        Ok(self.storage.get_unchecked(storage_index))
    }

    /// Uncheck `get`
    #[inline]
    pub fn g(&self, index: usize) -> T {
        let storage_index = self.layout.get_storage_index_uncheck(index);
        self.storage.get_unchecked(storage_index)
    }

    pub fn set(&self, index: usize, value: T) -> Result<()> {
        let storage_index = self.layout.get_storage_index(index)?;
        self.storage.set_unchecked(storage_index, value);
        Ok(())
    }

    /// Uncheck `set`
    #[inline]
    pub fn s(&self, index: usize, value: T) {
        let storage_index = self.layout.get_storage_index_uncheck(index);
        self.storage.set_unchecked(storage_index, value);
    }

    pub fn copy(&self) -> Vector<T> {
        let data: Vec<_> = self.iter().collect();
        let storage = StorageArc::new(Storage::new(data));
        Self {
            storage,
            layout: VectorLayout::contiguous(self.len())
        }
    }

    pub fn swap(&self, other: &Self) -> Result<()> {
        if self.len() != other.len() {
            return Err(Error::Msg("swap vector with differnet len".into()));
        }

        if StorageArc::ptr_eq(&self.storage, &other.storage) {
            let mut storage = self.storage.write();
            let len = self.len();
            (0..len).into_iter()
                .map(|index| (self.layout.get_storage_index_uncheck(index), other.layout.get_storage_index_uncheck(index)))
                .for_each(|(self_index, other_index)| {
                    let self_value = storage.get_unchecked(self_index);
                    let other_value = storage.get_unchecked(other_index);
                    storage.set_unchecked(self_index, other_value);
                    storage.set_unchecked(other_index, self_value);
                });

        } else {
            let mut self_storage = self.storage.write();
            let mut other_storage = other.storage.write();
    
            let len = self.len();
            (0..len).into_iter()
                .map(|index| (self.layout.get_storage_index_uncheck(index), other.layout.get_storage_index_uncheck(index)))
                .for_each(|(self_index, other_index)| {
                    let self_value = self_storage.get_unchecked(self_index);
                    let other_value = other_storage.get_unchecked(other_index);
                    self_storage.set_unchecked(self_index, other_value);
                    other_storage.set_unchecked(other_index, self_value);
                });
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.layout.len
    }

    pub fn iter<'a>(&'a self) -> VectorIter<'a, T> {
        VectorIter {
            view: self.to_view(),
            index: 0
        }
    }

    pub(crate) fn from_storage(storage: Storage<T>, len: usize) -> Self {
        Self {
            storage: StorageArc::new(storage),
            layout: VectorLayout::contiguous(len),
        }
    }
}

impl<'a, T: WithDType> VectorView<'a, T> {
    pub fn from_ndarray(array: &'a NdArray<T>) -> Result<Self> {
        let _ = array.dims1()?;
        let len = array.layout().dims()[0];
        let stride = array.layout().stride()[0];
        let start_offset = array.layout().start_offset();
        let storage = array.storage_ref(start_offset);
        Ok(Self { len, stride, storage })
    }

    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            None
        } else {
            Some(self.storage.get_unchecked(self.storage_index(index)))
        }
    }

    /// Uncheck `get`
    pub fn g(&self, index: usize) -> T {
        self.storage.get_unchecked(self.storage_index(index))
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn copy(&self) -> Vector<T> {
        let data: Vec<_> = (0..self.len)
            .into_iter().map(|index| self.storage.get_unchecked(index * self.stride))
            .collect();
        let storage = StorageArc::new(Storage::new(data));
        Vector {
            storage,
            layout: VectorLayout::contiguous(self.len)
        }
    }

    #[inline]
    pub fn storage_index(&self, index: usize) -> usize {
        self.stride * index
    }
}

impl<'a, T: WithDType> IntoIterator for &'a Vector<T> {
    type Item = T;
    type IntoIter = VectorIter<'a, T>;
    fn into_iter(self) -> VectorIter<'a, T> {
        self.iter()
    }
}

impl<'a, T: WithDType> IntoIterator for VectorView<'a, T> {
    type Item = T;
    type IntoIter = VectorIter<'a, T>;
    fn into_iter(self) -> VectorIter<'a, T> {
        VectorIter {
            view: self,
            index: 0,
        }
    }
}

pub struct VectorIter<'a, T: WithDType> {
    view: VectorView<'a, T>,
    index: usize,
}

impl<'a, T: WithDType> Iterator for VectorIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.view.len {
            None
        } else {
            let value = self.view.g(self.index);
            self.index += 1;
            Some(value)
        }
    }
}

impl VectorLayout {
    pub fn contiguous(len: usize) -> Self {
        Self { len, stride: 1, start_offset: 0 }
    }

    pub fn take(&self, len: usize) -> Result<Self> {
        if len > self.len {
            return Err(Error::Msg("len out of range".into()));
        }

        Ok(Self {
            len,
            stride: self.stride,
            start_offset: self.start_offset,
        })
    }

    pub fn drop(&self, len: usize) -> Result<Self> {
        if len > self.len {
            return Err(Error::Msg("len out of range".into()));
        }

        let new_len = self.len - len;
        let start_offset = self.start_offset + self.stride * len;
        Ok(Self { len: new_len, stride: self.stride, start_offset })
    }

    #[inline]
    pub fn get_storage_index_uncheck(&self, index: usize) -> usize {
        self.start_offset + self.stride * index
    }

    pub fn get_storage_index(&self, index: usize) -> Result<usize> {
        if index >= self.len {
            Err(Error::VectorIndexOotOfRange { len: self.len, index })?
        }
        Ok(self.start_offset + self.stride * index)
    }
}

pub trait ToVectorView<T: WithDType> {
    fn to_vector_view(&self) -> Result<VectorView<'_, T>>;
}

impl<T: WithDType> ToVectorView<T> for NdArray<T> {
    fn to_vector_view(&self) -> Result<VectorView<'_, T>> {
        VectorView::from_ndarray(self)
    }
}

impl<T: WithDType> ToVectorView<T> for &NdArray<T> {
    fn to_vector_view(&self) -> Result<VectorView<'_, T>> {
        VectorView::from_ndarray(self)
    }
}

impl<T: WithDType> ToVectorView<T> for Vector<T> {
    fn to_vector_view(&self) -> Result<VectorView<'_, T>> {
        Ok(self.to_view())
    }
}

impl<T: WithDType> ToVectorView<T> for &Vector<T> {
    fn to_vector_view(&self) -> Result<VectorView<'_, T>> {
        Ok(self.to_view())
    }
}
