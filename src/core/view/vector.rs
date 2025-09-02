use crate::linalg::LinalgError;
use crate::{Error, FloatDType, NdArray, NumDType, Result, Storage, WithDType};
use crate::{StorageMut, StorageRef};

pub struct VectorView<'a, T: WithDType> {
    pub(crate) storage: StorageRef<'a, T>,
    pub(crate) len: usize,
    pub(crate) stride: usize,
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

    pub fn clone(&'a self) -> Self {
        Self {
            len: self.len,
            storage: self.storage.clone(),
            stride: self.stride
        }
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

    pub fn iter(&'a self) -> VectorIter<'a, T> {
        VectorIter { 
            view: self.clone(), 
            index: 0,
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

    pub fn copy(&self) -> NdArray<T> {
        let data: Vec<_> = (0..self.len)
            .into_iter().map(|index| self.storage.get_unchecked(index * self.stride))
            .collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, (self.len,))
    }

    #[inline]
    pub fn storage_index(&self, index: usize) -> usize {
        self.stride * index
    }
}

impl<'a, T: FloatDType> VectorView<'a, T> {
    pub fn norm(&self) -> T {
        let v = self.iter().map(|v| v.powi(2)).sum::<T>();
        v.sqrt()
    }

    pub fn dot(&self, rhs: &Self) -> Result<T> {
        if self.len() != rhs.len() {
            Err(Error::Msg("mis match size".into()))?;
        }
        Ok(self.iter().zip(rhs.iter())
            .map(|(a, b)| a * b)
            .sum::<T>())
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

pub struct VectorViewMut<'a, T: WithDType> {
    pub(crate) storage: StorageMut<'a, T>,
    pub(crate) len: usize,
    pub(crate) stride: usize,
}

impl<'a, T: WithDType> VectorViewMut<'a, T> {
    pub fn from_ndarray(array: &'a NdArray<T>) -> Result<Self> {
        let _ = array.dims1()?;
        let len = array.layout().dims()[0];
        let stride = array.layout().stride()[0];
        let start_offset = array.layout().start_offset();
        let storage = array.storage_mut(start_offset);
        Ok(Self { len, stride, storage })
    }

    pub fn clone(&'a self) -> VectorView<'a, T> {
        VectorView {
            len: self.len,
            storage: self.storage.clone(),
            stride: self.stride
        }
    }

    pub fn clone_mut(&'a mut self) -> VectorViewMut<'a, T> {
        Self {
            len: self.len,
            storage: self.storage.clone_mut(),
            stride: self.stride
        }
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

    pub fn set(&mut self, index: usize, value: T) -> Option<()> {
        if index >= self.len {
            None
        } else {
            self.storage.set_unchecked(self.storage_index(index), value);
            Some(())
        }
    }

    /// Uncheck `get`
    pub fn s(&mut self, index: usize, value: T) {
        self.storage.set_unchecked(self.storage_index(index), value);
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn copy(&self) -> NdArray<T> {
        let data: Vec<_> = (0..self.len)
            .into_iter().map(|index| self.storage.get_unchecked(index * self.stride))
            .collect();
        let storage = Storage::new(data);
        NdArray::from_storage(storage, (self.len,))
    }

    pub fn swap(&mut self, other: &mut Self) -> Result<()> {
        if self.len() != other.len() {
            return Err(Error::Msg("len mismatch".into()))?;
        }

        let stride = self.stride;
        let storage_index = |index: usize| {
            stride * index
        };

        let len = self.len();
        (0..len).into_iter()
            .map(|index| (storage_index(index), storage_index(index)))
            .for_each(|(self_index, other_index)| {
                let self_value = self.storage.get_unchecked(self_index);
                let other_value = other.storage.get_unchecked(other_index);
                self.storage.set_unchecked(self_index, other_value);
                other.storage.set_unchecked(other_index, self_value);
            });

        Ok(())
    }

    pub fn iter(&'a self) -> VectorIter<'a, T> {
        VectorIter { 
            view: self.clone(), 
            index: 0,
        }
    }

    #[inline]
    pub fn storage_index(&self, index: usize) -> usize {
        self.stride * index
    }
}


impl<'a, T: FloatDType> VectorViewMut<'a, T> {
    pub fn norm(&self) -> T {
        let v = self.iter().map(|v| v.powi(2)).sum::<T>();
        v.sqrt()
    }

    pub fn dot(&self, rhs: &Self) -> Result<T> {
        if self.len() != rhs.len() {
            Err(Error::Msg("mis match size".into()))?;
        }
        Ok(self.iter().zip(rhs.iter())
            .map(|(a, b)| a * b)
            .sum::<T>())
    }
}

impl<'a, T: NumDType> VectorViewMut<'a, T> {
    pub fn mul_assign(&'a mut self, mul: T) {
        // TODO: fast
        for i in 0..self.len {
            let v = self.g(i);
            self.s(i, v * mul);
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
