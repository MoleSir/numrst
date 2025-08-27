use std::sync::{Arc, RwLock, RwLockReadGuard};
use crate::{Error, NdArray, NumDType, Result, Storage, WithDType};

pub struct VectorLayout {
    pub(crate) len: usize,
    pub(crate) stride: usize,
    pub(crate) start_offset: usize,
}

pub struct Vector<T: WithDType> {
    pub(crate) storage: Arc<RwLock<Storage<T>>>,
    pub(crate) layout: VectorLayout,
}

impl<T: WithDType> Vector<T> {
    pub fn from_ndarray(array: &NdArray<T>) -> Result<Self> {
        let _ = array.dims1()?;
        Ok(Self::from_ndarray_impl(array, 0))
    }

    fn from_ndarray_impl(array: &NdArray<T>, axis: usize) -> Self {
        let layout = VectorLayout {
            len: array.layout().dims()[axis],
            stride: array.layout().stride()[axis],
            start_offset: array.layout().start_offset(),
        };
        Self {
            storage: array.storage_clone(),
            layout
        }
    }

    pub fn get(&self, index: usize) -> Result<T> {
        let index = self.layout.get_storage_index(index)?;
        Ok(self.storage.read().unwrap().data()[index])
    }

    pub fn set(&self, index: usize, value: T) -> Result<()> {
        let index = self.layout.get_storage_index(index)?;
        let mut storage = self.storage.write().unwrap();
        let data = storage.data_mut(); 
        data[index] = value;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.layout.len
    }

    pub fn iter<'a>(&'a self) -> VectorIter<'a, T> {
        let storage = self.storage.read().unwrap();
        let start_offset = self.layout.start_offset;
        let data: std::sync::MappedRwLockReadGuard<'_, [T]> = RwLockReadGuard::map(storage, |s| &s.data()[start_offset..]);
        VectorIter {
            data,
            len: self.len(),
            stride: self.layout.stride,
            index: 0
        }
    }
}

impl<T: NumDType> Vector<T> {
    pub fn dot(&self, rhs: &Self) -> Result<T> {
        if self.len() != rhs.len() {
            return Err(Error::Msg("dot with diff len vector".into()));
        }
        let sum = self.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum::<T>();
        Ok(sum)
    }
}

impl<'a, T: WithDType> IntoIterator for &'a Vector<T> {
    type Item = T;
    type IntoIter = VectorIter<'a, T>;
    fn into_iter(self) -> VectorIter<'a, T> {
        self.iter()
    }
}

pub struct VectorIter<'a, T: WithDType> {
    data: std::sync::MappedRwLockReadGuard<'a, [T]>,
    len: usize,
    stride: usize,
    index: usize,
}

impl<'a, T: WithDType> Iterator for VectorIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.len {
            None
        } else {
            let index = self.index;
            let storage_index = self.stride * index;
            self.index += 1;
            Some(self.data[storage_index])
        }
    }
}

impl VectorLayout {
    pub fn get_storage_index(&self, index: usize) -> Result<usize> {
        if index >= self.len {
            Err(Error::Msg("Vector index out of range".into()))?
        }
        Ok(self.start_offset + self.stride * index)
    }
}

